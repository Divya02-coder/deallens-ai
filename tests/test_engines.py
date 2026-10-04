import pandas as pd, pytest
from finance_engine.ratios import compute_ratios
from finance_engine.scenarios import Base, Scenario, run
from dependency_engine.concentration import concentration
from anomaly_engine.rules import threshold_clustering
from contract_engine.extractor import regex_clause_scan
from risk_engine.findings import run_all


def test_ratios_known_values():
    f = pd.DataFrame({"year": [2023, 2024], "revenue": [100., 120.], "cogs": [60., 70.], "operating_expenses": [20., 25.],
                      "depreciation": [5., 5.], "interest_expense": [2., 2.], "net_income": [10., 12.],
                      "operating_cash_flow": [15., 14.], "capex": [5., 5.], "current_assets": [50., 60.], "inventory": [10., 12.],
                      "receivables": [20., 30.], "current_liabilities": [25., 30.], "payables": [10., 12.],
                      "total_assets": [100., 110.], "total_debt": [20., 25.], "total_equity": [50., 55.]})
    r = compute_ratios(f)
    assert r.loc[1, "revenue_growth"] == pytest.approx(0.2)
    assert r.loc[1, "ebitda"] == 25
    assert r.loc[1, "dso"] == pytest.approx(30 / 120 * 365)


def test_concentration_excludes_others():
    df = pd.DataFrame({"c": ["A", "B", "Others"], "v": [50., 30., 20.]})
    assert concentration(df, "c", "v")["top1"] == pytest.approx(0.5)


def test_threshold_cluster_detected():
    tx = pd.DataFrame({"vendor": ["X"] * 12, "amount": [49000.] * 12})
    assert threshold_clustering(tx)


def test_scenario_revenue_drop_reduces_ebitda():
    b = Base(300, 190, 58, 10, 7, 14, 70, 20)
    assert run(b, Scenario(revenue_change=-0.15))["delta_ebitda"] < 0


def test_clause_scan_requires_quote():
    hits = regex_clause_scan("Any change of control requires consent.")
    assert hits and "change of control" in hits[0]["quote"].lower()


def test_planted_risks_found():
    titles = " ".join(f["title"] for f in run_all()["findings"])
    for k in ["cash conversion", "Customer concentration", "Supplier concentration", "Debt maturity", "threshold"]:
        assert k.lower() in titles.lower()
