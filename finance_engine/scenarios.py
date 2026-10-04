"""STEP 4 - Scenario simulator: change assumptions, see financial impact."""
from dataclasses import dataclass, field


@dataclass
class Base:
    revenue: float
    cogs: float
    opex: float
    depreciation: float
    interest: float
    capex: float
    debt: float
    debt_service: float           # annual principal + interest
    tax_rate: float = 0.25
    dwc_pct_revenue: float = 0.0  # change in working capital as % of revenue change


@dataclass
class Scenario:
    revenue_change: float = 0.0        # -0.15 = -15%
    opex_change: float = 0.0           # +0.08 = +8%
    cogs_pct_change: float = 0.0       # change in COGS as % of revenue (points)
    lost_customer_revenue: float = 0.0 # absolute revenue lost from a churned account
    lost_customer_margin: float = 0.4  # contribution margin on lost revenue
    synergy: float = 0.0               # absolute annual cost synergy


def _pnl(revenue, cogs, opex, dep, interest, tax):
    ebitda = revenue - cogs - opex
    ebit = ebitda - dep
    ni = (ebit - interest) * (1 - tax)
    return ebitda, ni


def run(base: Base, s: Scenario) -> dict:
    b_ebitda, b_ni = _pnl(base.revenue, base.cogs, base.opex, base.depreciation, base.interest, base.tax_rate)
    cogs_ratio = base.cogs / base.revenue + s.cogs_pct_change
    rev = base.revenue * (1 + s.revenue_change) - s.lost_customer_revenue
    cogs = rev * cogs_ratio
    opex = base.opex * (1 + s.opex_change) - s.synergy
    ebitda, ni = _pnl(rev, cogs, opex, base.depreciation, base.interest, base.tax_rate)
    ocf = ni + base.depreciation
    fcf = ocf - base.capex
    b_fcf = b_ni + base.depreciation - base.capex
    return {
        "revenue": rev, "ebitda": ebitda, "net_income": ni, "free_cash_flow": fcf,
        "delta_revenue": rev - base.revenue, "delta_ebitda": ebitda - b_ebitda,
        "delta_fcf": fcf - b_fcf,
        "ebitda_margin": ebitda / rev if rev else float("nan"),
        "debt_to_ebitda": base.debt / ebitda if ebitda > 0 else float("inf"),
        "dscr": (ebitda / base.debt_service) if base.debt_service else float("nan"),
        "base_ebitda": b_ebitda,
    }


def sensitivity(base: Base, s: Scenario, field_name: str, deltas: list[float]) -> list[dict]:
    """One-at-a-time sensitivity of EBITDA to a scenario field."""
    rows = []
    for d in deltas:
        sc = Scenario(**{**s.__dict__, field_name: d})
        rows.append({field_name: d, "ebitda": run(base, sc)["ebitda"]})
    return rows
