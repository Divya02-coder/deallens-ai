# DealLens AI — Evidence-Linked M&A Due Diligence

DealLens is a document-intelligence and risk-analytics system for M&A diligence. It combines deterministic financial/transaction checks with contract/document retrieval, traceable evidence snippets, and an optional LLM investigation workflow. It is intentionally distinct from a web-research agent: DealLens works on a deal room's documents and structured financial records, not autonomous internet search.

> **Important:** outputs are investigative signals, not investment, legal, accounting, or compliance advice. Retrieved passages are evidence candidates; the application does not claim that semantic matching independently proves a finding.

## What it does

- **Structured risk analytics:** financial ratios/trends, customer and supplier concentration, debt maturity, contract exposure, payment anomalies and transaction outliers.
- **Document ingestion and RAG:** PDF, CSV, Excel, TXT, Markdown, JSON and DOCX ingestion with source/page/sheet/row metadata.
- **Evidence fusion:** risk findings can carry relevant document passages or table-row ranges and a clear status that distinguishes retrieval matches from independently verified evidence.
- **REST API:** document upload, asynchronous analysis jobs, status polling, and findings retrieval.
- **Background processing:** local FastAPI background-task fallback or Celery workers backed by Redis, with retries in the worker configuration.
- **Privacy helpers:** regex redaction for email, phone-like values, Indian PAN and Aadhaar-like numbers. Optional spaCy PERSON-entity redaction can be enabled with a locally installed model.
- **Optional local model:** call an Ollama-compatible local model such as Qwen through a provider adapter, without requiring Gemini for that endpoint.
- **Evaluation:** existing synthetic holdout regression suite plus a labelled-case evaluator for category-level precision, recall and F1.
- **UI:** existing Streamlit investigation dashboard.
- **Quality and delivery:** pytest tests, GitHub Actions, Docker and Docker Compose.

## Architecture

```text
Streamlit UI / REST client
          |
          v
   FastAPI (backend.main)
    | upload + jobs + findings
    v
 SQLite job/document registry ---- Redis/Celery (optional)
    |                                  |
    +--------------------------> worker
                                       |
                                       v
                        deterministic risk engines
                        + document ingestion
                        + evidence fusion/redaction
                                       |
                                       v
                          persisted job findings
```

The existing `data/raw/*.csv` files are the default structured demo case. Uploaded documents are supplied explicitly to an analysis job using their returned document IDs. They are not silently treated as replacement financial statements.

## Quick start — local

Use Python 3.11+.

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env` for local secrets/configuration. Put a real `GEMINI_API_KEY` in your local `.env` only if using Gemini features; never commit it.

Run the dashboard:

```bash
python -m streamlit run frontend/app.py
```

Run the API in another terminal:

```bash
uvicorn backend.main:app --reload
```

- Health check: `GET http://127.0.0.1:8000/health`
- Interactive API docs: `http://127.0.0.1:8000/docs`

## REST API workflow

### 1. Upload one or more documents

```bash
curl -X POST http://127.0.0.1:8000/api/v1/documents \
  -F "file=@sample_contract.pdf"
```

Example response:

```json
{"document_id":"<returned-uuid>","filename":"sample_contract.pdf","bytes":12345,"status":"uploaded"}
```

Supported extensions: `.pdf`, `.csv`, `.xlsx`, `.xls`, `.txt`, `.md`, `.json`, `.docx`. Uploads are capped at 25 MiB by default (`DEALLENS_MAX_UPLOAD_BYTES`).

### 2. Start an analysis job

Use the real `document_id` returned by upload:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/analyses \
  -H "Content-Type: application/json" \
  -d '{"document_ids":["<returned-uuid>"],"data_dir":"data/raw"}'
```

The response is `202 Accepted` with `analysis_id`, `status_url` and `findings_url`.

### 3. Poll status and retrieve findings

```bash
curl http://127.0.0.1:8000/api/v1/analyses/<analysis_id>
curl http://127.0.0.1:8000/api/v1/analyses/<analysis_id>/findings
```

Possible job statuses: `queued`, `running`, `succeeded`, `failed`. A failed job exposes a bounded error string in its status response. Do not expose the API publicly without authentication, rate limits, access control, TLS and an upload/storage policy.

### Privacy helper

```bash
curl -X POST http://127.0.0.1:8000/api/v1/privacy/redact \
  -H "Content-Type: application/json" \
  -d '{"text":"Contact jane@example.com"}'
```

Redaction is best-effort, not a guarantee of anonymization. The optional spaCy NER path is enabled by setting `DEALLENS_SPACY_MODEL` to an installed model name. The app redacts extracted document chunks before persisting job results; original uploaded files are retained for analysis and must be protected/deleted according to your own retention policy.

## Celery + Redis

For a local API-only run, background tasks run in the API process. For a worker-backed deployment:

```bash
docker compose up --build
```

Compose starts the API, Redis and a Celery worker. The job database and `data/` are mounted into both app containers. Configure real API credentials in a local environment file before enabling any provider that requires them. For production, replace the demo secret handling with a proper secret manager and use a shared production database rather than SQLite across multiple replicas.

To run the worker manually after installing dependencies:

```bash
celery -A backend.workers.celery_app.celery_app worker --loglevel=INFO
```

The API uses Celery only when `DEALLENS_USE_CELERY=1`; otherwise it uses FastAPI background tasks. Worker tasks retry exceptions up to three times with exponential backoff.

## Optional local open-weight model (Ollama)

1. Install Ollama separately and pull a model compatible with your hardware, for example `ollama pull qwen2.5:7b`.
2. Ensure the Ollama server is reachable from the API process.
3. Configure `OLLAMA_BASE_URL` (default `http://localhost:11434`) and `OLLAMA_MODEL` (default `qwen2.5:7b`).
4. Call:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/models/ollama/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt":"List possible risks in this excerpt: ...","system_instruction":"Be evidence-bound."}'
```

Benchmark repeated prompts:

```bash
python evaluation/compare_ollama.py --model qwen2.5:7b --prompts evaluation/model_prompts.json
```

Create `evaluation/model_prompts.json` as a JSON array of prompt objects. Compare models on the same held-out prompts and report latency, token counts, human-reviewed quality, hardware, RAM/VRAM and model quantization. Do not claim a model is more accurate from latency alone.

## Tests and CI

```bash
pytest -q
```

GitHub Actions runs tests on pushes and pull requests. The tests cover API health, upload validation, redaction, evidence metadata and evaluation metrics. Integration tests requiring a live Gemini/Ollama/Redis service should be marked and run separately.

## Evaluation: do not confuse synthetic results with real-world accuracy

Run the included regression benchmark:

```bash
python evaluation/benchmark/evaluate_benchmark.py
```

The repository includes a synthetic holdout for deterministic risk-engine regression. It is useful for checking planted scenarios and preventing regressions, but does **not** establish production accuracy.

For manually labelled cases, create a JSONL file where each line includes `case_id`, `expected_categories`, and `predicted_categories`, then run:

```bash
python evaluation/metrics.py evaluation/labelled_cases.jsonl --output evaluation/results.json
```

This reports per-category precision, recall and F1. For citation correctness, label whether each finding's source/page/span truly supports its claim and report that metric separately. See [`evaluation/README.md`](evaluation/README.md).

CUAD is a contract-review dataset, not a full M&A diligence benchmark. Use it for relevant contract clause categories after carefully mapping its annotation schema; evaluate financial anomalies and customer/supplier concentration on separately labelled financial cases. Do not report CUAD metrics until the dataset is downloaded, the mapping is validated, and the evaluation has actually been run.

## Resume positioning

**DealLens AI — Enterprise Due Diligence & Risk Intelligence**
- FastAPI, Celery/Redis, document ingestion, structured + unstructured evidence fusion, privacy-aware processing, explainable risk analytics and evaluation.

Keep your **SerpAPI Research Agent** positioned around web-search orchestration, source discovery and synthesis. Avoid describing DealLens as another general-purpose web research agent.

Only list deployment, benchmark scores, or model-comparison results on a resume after running and verifying them in your environment.
