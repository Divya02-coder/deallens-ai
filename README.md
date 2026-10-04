# DealLens AI — M&A Due-Diligence & Risk Intelligence

DealLens AI is an evidence-first financial investigation workspace for first-pass M&A due diligence.

## What this version demonstrates

- Deterministic financial analysis: revenue growth, margins, liquidity, leverage, working-capital metrics
- Explainable earnings-quality and revenue-quality checks
- Rule-based transaction screening + Isolation Forest anomaly detection
- Benford first-digit screening (signal only; never treated as proof of fraud)
- Customer and supplier concentration analysis
- Debt maturity exposure
- Contract renewal + change-of-control cross-document reasoning
- NetworkX dependency graph
- PDF parsing and page-cited TF-IDF search
- Scenario simulation and EBITDA sensitivity
- Optional LLM investigation agent with controlled read-only tools
- Offline investigation assistant when no API key is available
- FastAPI API + Streamlit dashboard
- Automated tests

## Run locally (Windows PowerShell)

```powershell
cd deallens-ai

python -m venv .venv
.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

python data/generate_sample_data.py
python -m pytest -q tests

streamlit run frontend/app.py
```

Open the URL Streamlit prints, normally:
`http://localhost:8501`

## Run the FastAPI backend

Open a second terminal:

```powershell
cd deallens-ai
.venv\Scripts\Activate.ps1
uvicorn backend.main:app --reload
```

API docs:
`http://127.0.0.1:8000/docs`

## Optional AI agent

The project works without an LLM key. To activate the multi-step Claude investigation agent:

```powershell
$env:ANTHROPIC_API_KEY="YOUR_KEY"
streamlit run frontend/app.py
```

If your Anthropic account uses a different model name, set:

```powershell
$env:DEALLENS_MODEL="YOUR_MODEL_NAME"
```

## Recruiter demo flow

1. Open **Executive View** and explain why there is no opaque overall deal score.
2. Show the **Risk Register** and open the highest-risk finding.
3. Explain that every major finding has evidence + management questions.
4. Open **Financials** and show deterministic ratio calculations.
5. Open **Dependencies** and explain customer/supplier concentration plus the knowledge graph.
6. Open **Transactions** and show Isolation Forest + threshold clustering + Benford screening.
7. Open **Scenario** and simulate revenue decline / cost increase / customer loss / synergies.
8. Upload a PDF and use **Documents / AI** for page-cited retrieval.
9. Ask the investigation assistant for the top risks.
10. Download the Markdown due-diligence report.

## Architecture

User → Streamlit workspace → shared analysis pipeline → deterministic finance/anomaly/dependency/contract engines → evidence-backed risk register → graph + scenario simulator → optional tool-calling LLM agent.

The LLM is deliberately not responsible for financial arithmetic. It can interpret results and call controlled read-only tools.

## Important design principles

- Findings are potential risks requiring investigation, not proof of wrongdoing.
- Deterministic calculations, ML signals and LLM interpretation remain separate.
- Evidence and source/page references should accompany important claims.
- Human analyst review remains required for material financial decisions.

## Project structure

```text
deallens-ai/
├── frontend/
├── backend/
├── data/
├── document_engine/
├── finance_engine/
├── anomaly_engine/
├── dependency_engine/
├── contract_engine/
├── graph_engine/
├── risk_engine/
├── agent/
├── tests/
├── requirements.txt
└── README.md
```
