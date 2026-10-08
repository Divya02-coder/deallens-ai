# DealLens AI

### Agentic AI-Powered M&A Due Diligence & Risk Intelligence

DealLens AI is an **agentic M&A due-diligence platform** that analyzes financial data, commercial contracts, transactions, customer/supplier concentration, debt schedules, and supporting documents to identify potential deal risks and produce **evidence-backed investment intelligence**.

Instead of treating an LLM as a simple chatbot, DealLens combines:

- **Gemini LLMs**
- **Retrieval-Augmented Generation (RAG)**
- **ChromaDB**
- **Sentence Transformers**
- **LangGraph**
- **specialized investigation agents**
- **deterministic risk detection**
- **evidence verification**
- **NetworkX relationship analysis**
- **FastAPI**
- **Streamlit**

The result is a system designed to answer a practical M&A question:

> **"What could go wrong with this deal, and can we prove it from the underlying evidence?"**

---

##  Why DealLens?

Traditional 
document-analysis systems often stop at:

> Upload documents → ask an LLM questions → receive an answer.

That approach can produce useful summaries, but M&A due diligence requires more than summarization.

A serious due-diligence workflow needs to:

1. Find relevant evidence.
2. Understand relationships between entities.
3. Detect financial and operational anomalies.
4. Analyze contractual exposure.
5. Identify concentration risks.
6. Verify findings against source evidence.
7. Prioritize risks.
8. Explain why a finding matters.
9. Produce an actionable investment-level summary.

DealLens is designed around this workflow.

---

#  Core Capabilities

## 1. Agentic M&A Investigation

DealLens uses a structured investigation workflow rather than relying on a single LLM prompt.

The investigation can delegate analysis to specialized components for areas such as:

- Financial analysis
- Contract analysis
- Customer concentration
- Supplier concentration
- Transaction anomalies
- Debt maturity
- Evidence verification
- Risk scoring
- Investment memo generation

The goal is to transform raw deal information into a structured risk investigation.

---

## 2. Retrieval-Augmented Generation

DealLens uses a RAG pipeline to retrieve relevant evidence from documents before generating conclusions.

### Pipeline

```text
Documents
    │
    ▼
Document Parsing
    │
    ▼
Chunking
    │
    ▼
Sentence Transformer Embeddings
    │
    ▼
ChromaDB Vector Store
    │
    ▼
Semantic Retrieval
    │
    ▼
Relevant Evidence
    │
    ▼
Gemini