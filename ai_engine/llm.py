import os
import time
import random

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiLLM:
    """
    Small provider layer around the Gemini API.

    Features:
    - Environment-based API key
    - Configurable primary model
    - Configurable fallback model
    - Retry with exponential backoff
    - Handles temporary 429/5xx/503 failures
    """

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not configured. "
                "Add it to your .env file."
            )

        self.client = genai.Client(api_key=api_key)

        self.model = os.getenv(
            "DEALLENS_MODEL",
            "gemini-3.8-flash",
        )

        self.fallback_model = os.getenv(
            "DEALLENS_FALLBACK_MODEL",
            "gemini-3.5-flash",
        )

        self.max_retries = int(
            os.getenv("GEMINI_MAX_RETRIES", "3")
        )

    # ---------------------------------------------------------
    # INTERNAL REQUEST
    # ---------------------------------------------------------

    def _request(
        self,
        model,
        prompt,
        system_instruction=None,
        max_tokens=3000,
        temperature=0.2,
    ):
        config = types.GenerateContentConfig(
            system_instruction=(
                system_instruction
                if system_instruction
                else None
            ),
            max_output_tokens=max_tokens,
            temperature=temperature,
        )

        return self.client.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )

    # ---------------------------------------------------------
    # GENERATE
    # ---------------------------------------------------------

    def generate(
        self,
        prompt,
        system_instruction=None,
        max_tokens=3000,
        temperature=0.2,
    ):
        """
        Generate text with retry + fallback model.

        Primary model is tried first.
        If it repeatedly fails with a transient server/rate
        problem, the fallback model is attempted.
        """

        models_to_try = [self.model]

        if (
            self.fallback_model
            and self.fallback_model != self.model
        ):
            models_to_try.append(self.fallback_model)

        last_error = None

        for model in models_to_try:

            for attempt in range(self.max_retries):

                try:

                    response = self._request(
                        model=model,
                        prompt=prompt,
                        system_instruction=system_instruction,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                    text = getattr(
                        response,
                        "text",
                        None,
                    )

                    if text and text.strip():
                        return text.strip()

                    raise RuntimeError(
                        "Gemini returned an empty response."
                    )

                except Exception as exc:

                    last_error = exc

                    error_text = str(exc).lower()

                    transient = any(
                        token in error_text
                        for token in [
                            "503",
                            "429",
                            "500",
                            "502",
                            "504",
                            "unavailable",
                            "high demand",
                            "resource exhausted",
                            "temporarily",
                            "timeout",
                        ]
                    )

                    # Don't retry permanent client errors.
                    if not transient:
                        raise

                    if attempt < self.max_retries - 1:

                        delay = (
                            2 ** attempt
                            + random.uniform(0, 0.5)
                        )

                        time.sleep(delay)

            # Move to fallback model.

        raise RuntimeError(
            "Gemini is temporarily unavailable after "
            f"trying {len(models_to_try)} model(s). "
            f"Last error: {last_error}"
        )