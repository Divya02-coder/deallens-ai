from __future__ import annotations
import json
from ai_engine.llm import GeminiLLM

class EvidenceVerifier:
    def verify(self, claim: str, evidence: list[dict]) -> dict:
        compact = [{k: e.get(k) for k in ("source", "page", "location", "text", "score") if k in e} for e in evidence]
        prompt = ("Verify the claim only against the supplied evidence. Return JSON with keys: "
                  "verdict (supported|partially_supported|unsupported), confidence (0-1), reasoning, "
                  "supporting_evidence, missing_evidence. Never invent a citation.\n\nCLAIM:\n" + claim +
                  "\n\nEVIDENCE:\n" + json.dumps(compact, default=str))
        text = GeminiLLM().generate(prompt, "You are a strict evidence verifier. JSON only.", 1800)
        try:
            return json.loads(text[text.find("{"):text.rfind("}")+1])
        except Exception:
            return {"verdict":"unsupported","confidence":0.0,"reasoning":text,"supporting_evidence":[],"missing_evidence":["Structured verifier output could not be parsed."]}
