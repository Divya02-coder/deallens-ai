from __future__ import annotations
import json
from ai_engine.llm import GeminiLLM

class FinancialCopilot:
    def answer(self, question: str, state: dict) -> str:
        context = {
            "ratios": state.get("ratios").to_dict("records") if hasattr(state.get("ratios"), "to_dict") else state.get("ratios"),
            "findings": state.get("findings", [])[:20],
            "latest_financials": state.get("fin").tail(3).to_dict("records") if hasattr(state.get("fin"), "tail") else state.get("fin"),
        }
        prompt = ("Answer the analyst's financial question using only this computed DealLens context. "
                  "Do not invent numbers. If the context is insufficient, say so. Include the relevant figures "
                  "and explain the business implication.\n\nQUESTION:\n" + question + "\n\nCONTEXT:\n" + json.dumps(context, default=str))
        return GeminiLLM().generate(prompt, "You are a cautious M&A financial analyst. Deterministic outputs are authoritative.", 2400)
