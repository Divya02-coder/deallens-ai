# 🔎 DealLens AI

## AI-Powered M&A Due Diligence & Risk Intelligence Platform

> **Evidence-first financial investigation • Automated risk detection • Human analyst in the loop**

DealLens AI is an AI-assisted M&A due-diligence platform that analyzes financial, transaction, customer, supplier, debt, and contract data to identify potential business risks and generate evidence-backed investigation findings.

Instead of relying only on an LLM to summarize documents, DealLens combines **deterministic financial analysis, machine-learning-based anomaly detection, dependency analysis, contract intelligence, risk scoring, and optional AI interpretation**.

---

# 📋 Table of Contents

- [Problem](#-problem)
- [Solution](#-solution)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Dataset](#-dataset)
- [Installation](#-installation)
- [Usage](#-usage)
- [Screenshots](#-screenshots)
- [Results](#-results)
- [API Documentation](#-api-documentation)
- [Future Improvements](#-future-improvements)

---

# ❗ Problem

Mergers and Acquisitions (M&A) require extensive due diligence before an investment or acquisition decision is made.

Analysts may need to examine:

- Financial statements
- Revenue and profitability
- Operating cash flow
- Customer concentration
- Supplier dependency
- Transaction records
- Debt schedules
- Contracts
- Operational dependencies

When this information is spread across multiple datasets and documents, manually identifying relationships and hidden risks can be time-consuming.

Important patterns can easily be overlooked, such as:

- Revenue growing while cash generation deteriorates
- Receivables growing faster than revenue
- Excessive customer concentration
- Supplier dependency
- Duplicate transactions
- Unusual transaction amounts
- Upcoming debt maturities
- Contract renewal risks
- Potential change-of-control issues

### The core problem

> **How can analysts investigate large amounts of business data faster while keeping risk findings explainable and traceable to evidence?**

---

# 💡 Solution

DealLens AI provides a centralized investigation workspace that combines multiple analytical engines into a single platform.

The system:

1. Ingests structured business data
2. Validates and processes the data
3. Calculates financial metrics
4. Detects transaction anomalies
5. Analyzes customer concentration
6. Analyzes supplier dependency
7. Evaluates debt exposure
8. Analyzes contract-related risks
9. Builds relationships between entities
10. Aggregates findings into a risk register
11. Presents evidence through an interactive dashboard
12. Allows optional AI-assisted interpretation

### Evidence-first approach

DealLens separates **calculated evidence** from **AI-generated interpretation**.

For example:

```text
Revenue increased
        ↓
Operating Cash Flow decreased
        ↓
Cash conversion weakened
        ↓
Potential financial risk
        ↓
Analyst investigation