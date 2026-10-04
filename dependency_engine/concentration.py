"""STEP 6 - Customer / supplier concentration (deterministic)."""
import pandas as pd

RESIDUAL = {"others", "other", "misc", "rest"}


def concentration(df: pd.DataFrame, name_col: str, value_col: str, top_n: int = 5) -> dict:
    total = df[value_col].sum()
    named = df[~df[name_col].str.lower().isin(RESIDUAL)].sort_values(value_col, ascending=False)
    shares = (named.set_index(name_col)[value_col] / total)
    return {
        "total": float(total),
        "shares": shares.to_dict(),
        "top1": float(shares.iloc[0]) if len(shares) else 0.0,
        "top_n": float(shares.head(top_n).sum()),
        "top1_name": shares.index[0] if len(shares) else None,
        "hhi": float((shares ** 2).sum() * 10_000),
    }


def concentration_findings(cust: pd.DataFrame, sup: pd.DataFrame) -> list[dict]:
    out = []
    c = concentration(cust, "customer", "revenue")
    if c["top1"] >= 0.20 or c["top_n"] >= 0.50:
        out.append({
            "id": "cust_conc", "category": "Dependency",
            "severity": "High" if c["top1"] >= 0.25 else "Medium",
            "title": "Customer concentration",
            "detail": f"Top customer {c['top1_name']} = {c['top1']:.0%} of revenue; top 5 = {c['top_n']:.0%}.",
            "evidence": [{"source": "customers.csv", "field": "revenue"}],
            "questions": ["Contract expiry, renewal history and churn risk for the top accounts?"],
        })
    s = concentration(sup, "supplier", "procurement")
    if s["top1"] >= 0.25:
        out.append({
            "id": "sup_conc", "category": "Dependency", "severity": "High" if s["top1"] >= 0.35 else "Medium",
            "title": "Supplier concentration",
            "detail": f"{s['top1_name']} = {s['top1']:.0%} of procurement.",
            "evidence": [{"source": "suppliers.csv", "field": "procurement"}],
            "questions": ["Are there qualified alternate suppliers? What are switching costs and lead times?"],
        })
    return out
