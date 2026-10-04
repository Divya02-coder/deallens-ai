"""STEP 1 - Create a synthetic target company ("XYZ Technologies") with planted risks.
Run:  python data/generate_sample_data.py
Planted risks (so you can verify the platform finds them):
  - Revenue up ~19% but operating cash flow down
  - Customer A = 30%+ of revenue, and its contract expires in ~4 months
  - Supplier X = ~40% of procurement
  - Debt due within 12 months is large
  - Payments clustered just below an approval threshold (Rs 50,000)
"""
import numpy as np, pandas as pd
from datetime import date, timedelta
from pathlib import Path

RAW = Path(__file__).parent / "raw"
RAW.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
CR = 1e7  # 1 crore

# --- financial statements (3 years, INR) ---
fin = pd.DataFrame({
    "year": [2022, 2023, 2024],
    "revenue": [210*CR, 252*CR, 300*CR],
    "cogs": [130*CR, 158*CR, 192*CR],
    "operating_expenses": [40*CR, 47*CR, 58*CR],
    "depreciation": [8*CR, 9*CR, 10*CR],
    "interest_expense": [4*CR, 5*CR, 7*CR],
    "net_income": [20*CR, 24*CR, 25*CR],
    "operating_cash_flow": [26*CR, 30*CR, 27*CR],   # falling while revenue rises
    "capex": [10*CR, 12*CR, 14*CR],
    "current_assets": [90*CR, 110*CR, 135*CR],
    "inventory": [22*CR, 30*CR, 41*CR],
    "receivables": [38*CR, 52*CR, 75*CR],           # growing faster than revenue
    "current_liabilities": [60*CR, 72*CR, 96*CR],
    "payables": [24*CR, 28*CR, 35*CR],
    "total_assets": [220*CR, 260*CR, 310*CR],
    "total_debt": [45*CR, 55*CR, 70*CR],
    "total_equity": [95*CR, 115*CR, 135*CR],
})
fin.to_csv(RAW / "financials.csv", index=False)

# --- customers ---
cust = pd.DataFrame({
    "customer": ["ABC Corp", "Beta Retail", "Gamma Infra", "Delta Foods", "Epsilon Labs", "Others"],
    "revenue": np.array([92, 41, 27, 21, 15, 104]) * CR,
})
cust.to_csv(RAW / "customers.csv", index=False)

# --- suppliers ---
sup = pd.DataFrame({
    "supplier": ["DEF Ltd", "Omega Metals", "Sigma Parts", "Others"],
    "procurement": np.array([77, 40, 25, 50]) * CR,
})
sup.to_csv(RAW / "suppliers.csv", index=False)

# --- debt schedule ---
today = date.today()
debt = pd.DataFrame({
    "lender": ["Bank X", "Bank Y", "NBFC Z"],
    "principal": [25*CR, 30*CR, 15*CR],
    "maturity_date": [today + timedelta(days=200), today + timedelta(days=900), today + timedelta(days=150)],
    "interest_rate": [0.095, 0.105, 0.135],
})
debt.to_csv(RAW / "debt_schedule.csv", index=False)

# --- contracts ---
contracts = pd.DataFrame({
    "contract_id": ["A17", "B02", "C09", "D11"],
    "counterparty": ["ABC Corp", "Beta Retail", "Gamma Infra", "DEF Ltd"],
    "type": ["customer", "customer", "customer", "supplier"],
    "value": [92*CR, 41*CR, 27*CR, 77*CR],
    "end_date": [today + timedelta(days=120), today + timedelta(days=700),
                 today + timedelta(days=400), today + timedelta(days=500)],
    "change_of_control": [True, False, False, True],
})
contracts.to_csv(RAW / "contracts.csv", index=False)

# --- transactions (payments) with a cluster just under the 50,000 threshold ---
n = 1500
tx = pd.DataFrame({
    "date": [today - timedelta(days=int(d)) for d in rng.integers(0, 365, n)],
    "vendor": rng.choice(["V-%02d" % i for i in range(1, 31)], n),
    "amount": np.round(rng.lognormal(mean=10, sigma=0.9, size=n), 2),
})
sus = pd.DataFrame({
    "date": [today - timedelta(days=int(d)) for d in rng.integers(0, 120, 45)],
    "vendor": "V-99",
    "amount": np.round(rng.uniform(48500, 49950, 45), 2),
})
tx = pd.concat([tx, sus], ignore_index=True).sample(frac=1, random_state=1)
tx.to_csv(RAW / "transactions.csv", index=False)
print("Sample data written to", RAW)
