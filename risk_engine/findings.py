"""STEP 11 - Risk engine: run every engine, merge, dedupe, score, rank."""
from pathlib import Path
import pandas as pd
from finance_engine.ratios import compute_ratios, validate_statements
from finance_engine.trends import quality_of_earnings_flags
from anomaly_engine.rules import threshold_clustering, duplicate_payments, benford_first_digit
from anomaly_engine.isolation_forest import score_transactions, outlier_findings
from dependency_engine.concentration import concentration, concentration_findings
from contract_engine.extractor import contract_findings

SEV = {"High": 3, "Medium": 2, "Low": 1}


def debt_findings(debt: pd.DataFrame, horizon_days: int = 365) -> list[dict]:
    d = debt.copy()
    d["maturity_date"] = pd.to_datetime(d["maturity_date"])
    due = d[(d["maturity_date"] - pd.Timestamp.today()).dt.days <= horizon_days]
    if due.empty:
        return []
    total, share = due["principal"].sum(), due["principal"].sum() / d["principal"].sum()
    return [{
        "id": "debt_maturity", "category": "Liquidity", "severity": "High" if share > 0.4 else "Medium",
        "title": "Debt maturity exposure",
        "detail": f"Rs {total/1e7:.1f} Cr ({share:.0%} of debt) matures within {horizon_days} days: "
                  f"{', '.join(due['lender'])}.",
        "evidence": [{"source": "debt_schedule.csv"}],
        "questions": ["Refinancing plans or committed facilities?", "Any covenant triggers on change of control?"],
    }]


def run_all(data_dir: str = "data/raw") -> dict:
    p = Path(data_dir)
    fin = pd.read_csv(p / "financials.csv")
    cust, sup = pd.read_csv(p / "customers.csv"), pd.read_csv(p / "suppliers.csv")
    debt, contracts = pd.read_csv(p / "debt_schedule.csv"), pd.read_csv(p / "contracts.csv")
    tx = pd.read_csv(p / "transactions.csv")

    ratios = compute_ratios(fin)
    shares = concentration(cust, "customer", "revenue")["shares"]
    scored = score_transactions(tx)
    findings = (quality_of_earnings_flags(fin, ratios) + concentration_findings(cust, sup)
                + debt_findings(debt) + contract_findings(contracts, shares)
                + threshold_clustering(tx) + duplicate_payments(tx) + outlier_findings(scored))
    bf = benford_first_digit(tx["amount"])
    if bf["suspicious"]:
        findings.append({
            "id": "benford",
            "category": "Transactions",
            "severity": "Medium",
            "title": "Transaction amounts deviate from Benford's-law expectations",
            "detail": f"First-digit analysis produced a chi-square statistic of {bf['chi_square']:.1f} across {bf['n']:,} positive payments. This is a screening signal only and is not evidence of fraud.",
            "evidence": [{"source": "transactions.csv", "rows": bf["n"], "method": "Benford first-digit test"}],
            "questions": ["Review the transaction population and business mix before interpreting the deviation.",
                          "Are there legitimate pricing, batching, or threshold effects affecting the first-digit distribution?"],
        })
    for issue in validate_statements(fin):
        findings.append({"id": "validation", "category": "Data integrity", "severity": "High",
                         "title": f"Statement check failed: {issue['check']}", "detail": f"FY{issue['year']}",
                         "evidence": [{"source": "financials.csv"}], "questions": ["Verify extraction / restated figures."]})
    findings.sort(key=lambda f: -SEV[f["severity"]])
    for i, f in enumerate(findings, 1):
        f["rank"] = i
    return {"findings": findings, "ratios": ratios, "benford": bf, "scored_tx": scored,
            "fin": fin, "cust": cust, "sup": sup, "debt": debt, "contracts": contracts}


def to_markdown(findings: list[dict], company: str = "Target") -> str:
    lines = [f"# Due-Diligence Risk Report: {company}", "",
             "_First-pass AI analysis. Findings are flags for human investigation, not conclusions._", ""]
    for f in findings:
        lines += [f"## Risk #{f['rank']} [{f['severity']}] {f['title']}", f"**Category:** {f['category']}", "",
                  f["detail"], "", f"**Evidence:** {f['evidence']}", "**Questions for management:**"]
        lines += [f"- {q}" for q in f["questions"]] + [""]
    return "\n".join(lines)
