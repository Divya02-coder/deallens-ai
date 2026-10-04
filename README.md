# 🔎 DealLens AI

### AI-Powered M&A Due Diligence & Risk Intelligence Platform

DealLens AI is an evidence-first financial investigation platform designed to accelerate **M&A due diligence** by automatically analyzing financial, transaction, customer, supplier, debt, and contract data.

Instead of simply generating an AI summary, DealLens combines **deterministic financial analysis, anomaly detection, risk scoring, dependency analysis, and knowledge-graph relationships** to surface potential deal risks and provide evidence for every important finding.

> **AI-assisted investigation. Evidence first. Human analyst in the loop.**

---

## 📌 Problem

M&A due diligence requires analysts to investigate large amounts of:

- Financial statements
- Transaction records
- Customer data
- Supplier data
- Debt schedules
- Contracts
- Operational dependencies

Important risks can easily be missed when information is distributed across multiple files.

Common examples include:

- Revenue growing while cash generation deteriorates
- Receivables increasing faster than revenue
- Excessive customer concentration
- Supplier dependency
- Suspicious or duplicate transactions
- Upcoming debt maturities
- Contract renewal or change-of-control risks
- Hidden relationships between entities

Traditional analysis is often manual, time-consuming, and difficult to reproduce.

---

# 💡 Solution

DealLens AI provides a centralized investigation workspace that:

1. Ingests structured business data
2. Validates and normalizes the data
3. Calculates deterministic financial metrics
4. Detects transaction anomalies
5. Identifies customer and supplier concentration
6. Analyzes debt and contract exposure
7. Builds relationships between entities
8. Assigns risk severity
9. Presents evidence-backed findings
10. Allows analysts to investigate and interpret the results

The system separates **calculated facts and ML signals from optional LLM interpretation**, reducing the risk of unsupported AI-generated claims.

---

# 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │   Source Documents  │
                    │                     │
                    │ Financials          │
                    │ Transactions        │
                    │ Customers           │
                    │ Suppliers           │
                    │ Debt                 │
                    │ Contracts            │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Document / Data    │
                    │      Engine         │
                    └──────────┬──────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
       Finance Engine    Anomaly Engine    Risk Engine
              │                │                │
              └────────────────┼────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
              ▼                ▼                ▼
      Dependency Engine   Contract Engine   Graph Engine
              │                │                │
              └────────────────┼────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Investigation /     │
                    │ Findings Layer      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    │                     │
                    │ Executive View      │
                    │ Risk Register       │
                    │ Financials          │
                    │ Dependencies        │
                    │ Transactions        │
                    │ Scenario Analysis   │
                    │ Documents / AI      │
                    └─────────────────────┘