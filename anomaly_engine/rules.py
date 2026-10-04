"""STEP 5a - Rule-based transaction checks. Flags patterns for investigation; never claims fraud."""
import pandas as pd


def threshold_clustering(tx: pd.DataFrame, threshold: float = 50_000, band: float = 0.05, min_count: int = 10) -> list[dict]:
    """Payments repeatedly just below an approval threshold, grouped by vendor."""
    lo = threshold * (1 - band)
    near = tx[(tx["amount"] >= lo) & (tx["amount"] < threshold)]
    out = []
    for vendor, g in near.groupby("vendor"):
        if len(g) >= min_count:
            out.append({
                "id": f"threshold_{vendor}", "category": "Transactions", "severity": "High",
                "title": f"Payments clustered just below approval threshold ({vendor})",
                "detail": f"{len(g)} payments between {lo:,.0f} and {threshold:,.0f}, total {g['amount'].sum():,.0f}. "
                          f"Pattern warrants review; it does not prove wrongdoing.",
                "evidence": [{"source": "transactions.csv", "vendor": vendor, "rows": int(len(g))}],
                "questions": [f"Who approved payments to {vendor}? Is there a contract or PO for each?",
                              "Why were payments split below the approval limit?"],
            })
    return out


def duplicate_payments(tx: pd.DataFrame) -> list[dict]:
    d = tx[tx.duplicated(["vendor", "amount"], keep=False)]
    if len(d) >= 5:
        return [{
            "id": "duplicates", "category": "Transactions", "severity": "Medium",
            "title": "Repeated identical payments to the same vendor",
            "detail": f"{len(d)} rows share vendor+amount with another row.",
            "evidence": [{"source": "transactions.csv", "rows": int(len(d))}],
            "questions": ["Are these legitimate recurring payments or duplicates?"],
        }]
    return []


def benford_first_digit(amounts: pd.Series) -> dict:
    """Chi-square-style deviation from Benford's law. Returns statistic (higher = more unusual)."""
    import numpy as np
    a = amounts[amounts > 0]
    first = a.astype(str).str.replace(r"[^1-9]", "", regex=True).str[0].astype(int)
    obs = first.value_counts(normalize=True).reindex(range(1, 10), fill_value=0)
    exp = pd.Series([np.log10(1 + 1 / d) for d in range(1, 10)], index=range(1, 10))
    chi = float((((obs - exp) ** 2) / exp).sum() * len(a))
    return {"chi_square": chi, "n": int(len(a)), "suspicious": chi > 15.5}  # 8 dof, p~0.05
