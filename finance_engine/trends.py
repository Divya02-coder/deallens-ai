"""STEP 3 - Trend / quality-of-earnings red flags (rule based, explainable)."""
import pandas as pd


def quality_of_earnings_flags(fin: pd.DataFrame, ratios: pd.DataFrame) -> list[dict]:
    flags = []
    f = fin.sort_values("year").reset_index(drop=True)
    r = ratios.sort_values("year").reset_index(drop=True)
    last, prev = len(f) - 1, len(f) - 2
    if prev < 0:
        return flags
    yr = int(f.loc[last, "year"])

    rev_g = r.loc[last, "revenue_growth"]
    ocf_g = r.loc[last, "ocf_growth"]
    if rev_g > 0.05 and ocf_g < 0:
        flags.append({
            "id": "cash_conversion", "category": "Earnings quality", "severity": "High",
            "title": "Weakening cash conversion",
            "detail": f"FY{yr}: revenue {rev_g:+.1%} but operating cash flow {ocf_g:+.1%}.",
            "evidence": [{"source": "financials.csv", "field": "revenue, operating_cash_flow", "year": yr}],
            "questions": ["Break down working-capital movements behind the OCF decline.",
                          "Is any revenue recognised ahead of cash collection?"],
        })

    rec_g = r.loc[last, "receivables_growth"]
    if rec_g > rev_g + 0.10:
        flags.append({
            "id": "receivables", "category": "Revenue quality", "severity": "High",
            "title": "Receivables growing much faster than revenue",
            "detail": f"FY{yr}: receivables {rec_g:+.1%} vs revenue {rev_g:+.1%}; "
                      f"DSO {r.loc[prev, 'dso']:.0f} -> {r.loc[last, 'dso']:.0f} days.",
            "evidence": [{"source": "financials.csv", "field": "receivables, revenue", "year": yr}],
            "questions": ["Request the AR ageing schedule.", "Any extended credit terms or channel stuffing?"],
        })

    if r.loc[last, "interest_coverage"] < 3:
        flags.append({
            "id": "interest_cov", "category": "Leverage", "severity": "Medium",
            "title": "Thin interest coverage",
            "detail": f"FY{yr} interest coverage is {r.loc[last, 'interest_coverage']:.1f}x.",
            "evidence": [{"source": "financials.csv", "field": "ebit, interest_expense", "year": yr}],
            "questions": ["What is the covenant headroom on existing facilities?"],
        })

    inv_g = f["inventory"].pct_change().iloc[last]
    if inv_g > rev_g + 0.10:
        flags.append({
            "id": "inventory", "category": "Working capital", "severity": "Medium",
            "title": "Inventory building faster than sales",
            "detail": f"FY{yr}: inventory {inv_g:+.1%} vs revenue {rev_g:+.1%}.",
            "evidence": [{"source": "financials.csv", "field": "inventory, revenue", "year": yr}],
            "questions": ["Provide inventory ageing and obsolescence provisions."],
        })
    return flags
