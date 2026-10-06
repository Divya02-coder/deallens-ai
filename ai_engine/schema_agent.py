"""AI-assisted data schema interpretation with deterministic profiling."""
from __future__ import annotations
import json
import pandas as pd
from ai_engine.llm import GeminiLLM


class SchemaAgent:
    def profile(self, df: pd.DataFrame) -> dict:
        return {
            "rows": int(len(df)),
            "columns": int(len(df.columns)),
            "columns_detail": [
                {"name": str(c), "dtype": str(df[c].dtype), "missing": int(df[c].isna().sum()), "unique": int(df[c].nunique(dropna=True))}
                for c in df.columns
            ],
        }

    def infer(self, df: pd.DataFrame) -> dict:
        profile = self.profile(df)
        llm = GeminiLLM()
        prompt = """Analyze this tabular dataset for an M&A due-diligence platform. Return valid JSON only with keys: dataset_purpose, key_entities, important_columns, data_quality_risks, recommended_checks. Do not invent facts; base suggestions on the supplied schema.

PROFILE:
""" + json.dumps(profile, default=str)
        text = llm.generate(prompt, "You are a data schema analyst. Return JSON only.", 1800)
        try:
            parsed = json.loads(text[text.find("{"):text.rfind("}")+1])
        except Exception:
            parsed = {"raw_analysis": text}
        parsed["profile"] = profile
        return parsed
