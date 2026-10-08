# DealLens AI — Advanced Agent Upgrade

## Preserved evaluation

The deterministic risk engine and benchmark evaluator were intentionally left unchanged. The independent 30-case holdout was rerun after the upgrade:

- Precision: **100.00%**
- Recall: **100.00%**
- F1: **100.00%**
- Accuracy: **100.00%**
- Evidence coverage: **100.00%**
- Average runtime: **0.355s** in the packaging environment
- Failure analysis: **0 false positives / 0 false negatives**

Runtime is hardware/environment dependent; the classification metrics are the important regression checks.

## New architecture

The new `advanced_agent/` package adds an optional LangGraph workflow:

`Planner → Specialist Investigations → Evidence Verification → LLM Synthesis`

It reuses the existing deterministic engines rather than replacing them. Specialist domains are:

- Financial
- Customer / supplier dependencies
- Contracts
- Transaction anomalies
- Document evidence

The planner uses Gemini when available and falls back to deterministic domain routing if the model/API is unavailable.

## Installation

```cmd
pip install -r requirements.txt
```

This now includes `langgraph`.

## Run

```cmd
python -m streamlit run frontend/app.py
```

Use the **Advanced Agent** tab for the new workflow.

## Benchmark

From the project root:

```cmd
python evaluation/benchmark/evaluate_benchmark.py
```

Do not tune the risk engine against the holdout set. The holdout is intended as an independent regression/evaluation set.

## CUAD

The large CUAD dataset is intentionally not bundled in this source ZIP. Keep the separately cloned `cuad/` directory locally if you are continuing the real-contract evaluation work. This keeps the project archive focused on source code and evaluation assets rather than duplicating hundreds of contract PDFs.
