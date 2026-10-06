from __future__ import annotations
import json
import pandas as pd
from ai_engine.llm import GeminiLLM

class AnomalyExplainer:
    def explain(self, anomalies: pd.DataFrame | list[dict], context: str = "") -> str:
        data = anomalies.to_dict("records") if isinstance(anomalies, pd.DataFrame) else anomalies
        prompt = ("Explain these machine-detected unusual observations for an M&A analyst. "
                  "Do not call them fraud. Distinguish observed facts from hypotheses. "
                  "Provide: Summary, Why flagged, Investigation questions, Evidence limitations.\n\n"
                  "ANOMALIES:\n" + json.dumps(data[:30], default=str) + "\n\nCONTEXT:\n" + context[:6000])
        return GeminiLLM().generate(prompt, "You are an evidence-first financial anomaly analyst.", 2200)
