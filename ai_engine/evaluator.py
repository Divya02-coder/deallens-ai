"""Small evaluation harness for recruiter-facing AI quality checks."""
from __future__ import annotations
from ai_engine.llm import GeminiLLM


CASES = [
    {"question": "What is the top financial risk?", "must_include": ["risk"]},
    {"question": "Are anomalies proof of fraud?", "must_include": ["not", "fraud"]},
    {"question": "Should an unsupported claim be treated as fact?", "must_include": ["evidence"]},
]


def evaluate(answer_fn) -> list[dict]:
    results = []
    for case in CASES:
        try:
            answer = answer_fn(case["question"]) or ""
            low = answer.lower()
            hits = [term for term in case["must_include"] if term in low]
            results.append({"question": case["question"], "passed": len(hits) == len(case["must_include"]), "matched": hits, "answer": answer})
        except Exception as exc:
            results.append({"question": case["question"], "passed": False, "matched": [], "answer": f"ERROR: {exc}"})
    return results
