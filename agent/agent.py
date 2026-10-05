import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types

from agent.tools import Toolbox, TOOL_SCHEMAS

load_dotenv()


SYSTEM_PROMPT = """
You are DealLens AI, an evidence-first M&A
due-diligence investigation agent.

Your job is to investigate financial and operational
risk using the tools provided.

CORE RULES:

1. Never invent evidence.
2. Prefer tool results over assumptions.
3. Distinguish facts from interpretations.
4. An anomaly is not proof of fraud.
5. Use language such as:
   - potential risk
   - unusual pattern
   - requires investigation
   - financial exposure
6. Cite document source and page whenever available.
7. Investigate before giving a conclusion.
8. Keep the final answer structured and concise.

When appropriate, use:
- document search
- financial analysis
- risk findings

Return a professional analyst-style investigation.
"""


def _build_tools():

    declarations = []

    for schema in TOOL_SCHEMAS:

        declarations.append(
            types.FunctionDeclaration(
                name=schema["name"],
                description=schema.get(
                    "description",
                    "",
                ),
                parameters=schema.get(
                    "parameters",
                    {
                        "type": "OBJECT",
                        "properties": {},
                    },
                ),
            )
        )

    return [
        types.Tool(
            function_declarations=declarations
        )
    ]


def _extract_function_calls(response):

    calls = []

    try:

        candidates = response.candidates or []

        for candidate in candidates:

            if not candidate.content:
                continue

            for part in candidate.content.parts or []:

                function_call = getattr(
                    part,
                    "function_call",
                    None,
                )

                if function_call:
                    calls.append(function_call)

    except Exception:
        pass

    return calls


def _function_response_part(
    name,
    result,
):

    if isinstance(result, dict):
        response_data = result

    elif isinstance(result, str):
        response_data = {
            "result": result
        }

    else:
        response_data = {
            "result": str(result)
        }

    return types.Part.from_function_response(
        name=name,
        response=response_data,
    )


def investigate(
    question,
    toolbox: Toolbox,
    max_steps=8,
):

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not configured."
        )

    model = os.getenv(
        "DEALLENS_MODEL",
        "gemini-3.8-flash",
    )

    client = genai.Client(
        api_key=api_key
    )

    tools = _build_tools()

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,

        tools=tools,

        # We execute the tools ourselves.
        automatic_function_calling={
            "disable": True
        },

        temperature=0.2,
        max_output_tokens=3500,
    )

    # ---------------------------------------------------------
    # IMPORTANT:
    # Use Chat instead of client.models.generate_content()
    # when tools are involved.
    # ---------------------------------------------------------

    chat = client.chats.create(
        model=model,
        config=config,
    )

    response = chat.send_message(
        question
    )

    for step in range(max_steps):

        function_calls = _extract_function_calls(
            response
        )

        # -----------------------------------------------------
        # GEMINI HAS FINISHED
        # -----------------------------------------------------

        if not function_calls:

            answer = getattr(
                response,
                "text",
                None,
            )

            if answer and answer.strip():
                return answer.strip()

            return (
                "The investigation completed, but "
                "Gemini did not return a textual conclusion."
            )

        # -----------------------------------------------------
        # EXECUTE TOOL CALLS
        # -----------------------------------------------------

        tool_parts = []

        for call in function_calls:

            tool_name = call.name

            raw_args = call.args

            if raw_args is None:
                arguments = {}

            elif isinstance(raw_args, dict):
                arguments = raw_args

            else:

                try:
                    arguments = dict(raw_args)
                except Exception:
                    arguments = {}

            try:

                result = toolbox.run(
                    tool_name,
                    arguments,
                )

            except Exception as exc:

                result = {
                    "error": (
                        f"Tool '{tool_name}' failed: "
                        f"{exc}"
                    )
                }

            tool_parts.append(
                _function_response_part(
                    tool_name,
                    result,
                )
            )

        # -----------------------------------------------------
        # SEND TOOL RESULTS BACK THROUGH CHAT
        # -----------------------------------------------------

        response = chat.send_message(
            tool_parts
        )

        if step < max_steps - 1:
            time.sleep(0.2)

    return (
        "The investigation reached the maximum "
        "number of reasoning/tool steps. "
        "Please narrow the question and try again."
    )