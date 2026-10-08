# DealLens Advanced Agent

DealLens now has an optional LangGraph orchestration layer on top of the existing deterministic engines.

## Workflow

`Planner -> Specialist Investigations -> Evidence Verification -> LLM Synthesis`

The specialist layer reuses the existing financial, concentration, contract, transaction-anomaly and RAG components. The benchmark never imports this module, so the deterministic evaluation remains isolated from LLM variability.

## Run

```cmd
pip install -r requirements.txt
python -m streamlit run frontend/app.py
```

Then open **Advanced Agent** in the dashboard.

If Gemini is unavailable, planning falls back to keyword routing and synthesis returns the verified structured evidence instead of fabricating an answer.
