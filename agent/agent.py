"""Gemini tool-calling investigation agent with an explicit tool loop."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from agent.tools import Toolbox, TOOL_SCHEMAS
import json

load_dotenv()

SYSTEM_PROMPT = """You are DealLens AI, an evidence-first M&A due-diligence investigation agent.
Never invent evidence. Use tools before stating computed facts. Distinguish facts from interpretations.
An anomaly is not proof of fraud. Cite file/page/location whenever available. If evidence is missing, say so.
Use verification for important claims. End with concrete management-investigation questions."""


def _build_tools():
    return [types.Tool(function_declarations=[types.FunctionDeclaration(
        name=s["name"], description=s["description"], parameters=s["parameters"]
    ) for s in TOOL_SCHEMAS])]


def _calls(response):
    out=[]
    for candidate in getattr(response, "candidates", []) or []:
        content=getattr(candidate, "content", None)
        for part in getattr(content, "parts", []) or []:
            fc=getattr(part, "function_call", None)
            if fc: out.append(fc)
    return out


def investigate(question: str, toolbox: Toolbox, max_steps: int = 8) -> str:
    key=os.getenv("GEMINI_API_KEY")
    if not key: raise RuntimeError("GEMINI_API_KEY is not configured.")
    client=genai.Client(api_key=key)
    model=os.getenv("DEALLENS_MODEL", "gemini-3.8-flash")
    config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, tools=_build_tools(), automatic_function_calling={"disable": True}, temperature=0.2, max_output_tokens=3500)
    chat=client.chats.create(model=model, config=config)
    response=chat.send_message(question)
    for _ in range(max_steps):
        calls=_calls(response)
        if not calls:
            return (getattr(response, "text", None) or "No textual conclusion was returned.").strip()
        parts=[]
        for call in calls:
            args=dict(call.args) if call.args else {}
            result=toolbox.run(call.name, args)
            try:
                data=json.loads(result)
            except Exception:
                data={"result":result}
            parts.append(types.Part.from_function_response(name=call.name, response=data))
        response=chat.send_message(parts)
    return "The investigation reached the tool-step limit. Narrow the question and retry."

