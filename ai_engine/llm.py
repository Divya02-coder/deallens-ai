"""Central Gemini client used by DealLens AI features."""
from __future__ import annotations
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()


class GeminiLLM:
    def __init__(self):
        key = os.getenv("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not configured. Add it to .env.")
        self.client = genai.Client(api_key=key)
        self.model = os.getenv("DEALLENS_MODEL", "gemini-3.8-flash")

    def generate(self, prompt: str, system_instruction: str = "", max_tokens: int = 3000) -> str:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction or None,
            max_output_tokens=max_tokens,
            temperature=0.2,
        )
        response = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        return (getattr(response, "text", None) or "").strip()
