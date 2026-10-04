"""STEP 2 - Deterministic financial ratios. NO LLM math here."""
import pandas as pd


def _div(a, b):
    return float(a) / float(b) if b not in (0, None) and pd.notna(b) else float("nan")


def compute_ratios(fin: pd.DataFrame) -> pd.DataFrame:
    """fin: one row per year (see data/raw/financials.csv). Returns ratios per year."""
    f = fin.sort_values("year").reset_index(drop=True).copy()
    out = pd.DataFrame({"year": f["year"]})

    ebitda = f["revenue"] - f["cogs"] - f["operating_expenses"]
    ebit = ebitda - f["depreciation"]
    out["ebitda"] = ebitda
    out["revenue_growth"] = f["revenue"].pct_change()
    out["gross_margin"] = (f["revenue"] - f["cogs"]) / f["revenue"]
    out["operating_margin"] = ebit / f["revenue"]
    out["ebitda_margin"] = ebitda / f["revenue"]
    out["net_margin"] = f["net_income"] / f["revenue"]
    out["current_ratio"] = f["current_assets"] / f["current_liabilities"]
    out["quick_ratio"] = (f["current_assets"] - f["inventory"]) / f["current_liabilities"]
    out["free_cash_flow"] = f["operating_cash_flow"] - f["capex"]
    out["ocf_growth"] = f["operating_cash_flow"].pct_change()
    out["cash_conversion"] = f["operating_cash_flow"] / ebitda          # OCF / EBITDA
    out["debt_to_equity"] = f["total_debt"] / f["total_equity"]
    out["debt_to_ebitda"] = f["total_debt"] / ebitda
    out["interest_coverage"] = ebit / f["interest_expense"]
    out["dso"] = f["receivables"] / f["revenue"] * 365
    out["dio"] = f["inventory"] / f["cogs"] * 365
    out["dpo"] = f["payables"] / f["cogs"] * 365
    out["inventory_turnover"] = f["cogs"] / f["inventory"]
    out["asset_turnover"] = f["revenue"] / f["total_assets"]
    out["receivables_growth"] = f["receivables"].pct_change()
    return out


def validate_statements(fin: pd.DataFrame, tol: float = 0.01) -> list[dict]:
    """Arithmetic sanity checks; failures = extraction error OR a real issue. Always route to review."""
    issues = []
    for _, r in fin.iterrows():
        if r["current_assets"] > r["total_assets"]:
            issues.append({"year": int(r["year"]), "check": "current_assets <= total_assets", "severity": "high"})
        implied_liab = r["total_assets"] - r["total_equity"]
        if implied_liab < r["total_debt"] * (1 - tol):
            issues.append({"year": int(r["year"]), "check": "debt exceeds implied liabilities", "severity": "high"})
    return issues
