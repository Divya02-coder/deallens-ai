# DealLens AI - first-pass M&A due-diligence platform

## Quick start
    pip install -r requirements.txt
    python data/generate_sample_data.py      # synthetic target with planted risks
    pytest -q tests                          # 6 tests
    streamlit run frontend/app.py            # dashboard
    uvicorn backend.main:app --reload        # API (docs at /docs)
    export ANTHROPIC_API_KEY=...             # optional: enables the investigation agent + LLM contract extraction

## Build order (each step is one file)
1  data/generate_sample_data.py     test data with known risks
2  finance_engine/ratios.py         deterministic ratios + statement validation
3  finance_engine/trends.py         earnings-quality red flags
4  finance_engine/scenarios.py      scenario simulator + sensitivity
5  anomaly_engine/                  threshold clustering, duplicates, Benford, Isolation Forest
6  dependency_engine/               customer / supplier concentration
7  contract_engine/extractor.py     clause scan, LLM extraction, cross-document exposure
8  graph_engine/knowledge_graph.py  NetworkX graph
9  document_engine/pdf_parser.py    PDF text + tables with page numbers
10 document_engine/search.py        page-cited search (TF-IDF -> swap for embeddings)
11 risk_engine/findings.py          merge, rank, report
12 agent/                           tool-calling investigation agent
13 backend/main.py                  FastAPI
14 frontend/app.py                 Streamlit

## Design rules
- Numbers come from code, never from the LLM.
- Every finding carries evidence + questions for management.
- Findings are flags for investigation, not conclusions.
- Document text is untrusted input (prompt-injection risk); tools are read-only.
