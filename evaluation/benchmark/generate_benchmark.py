from pathlib import Path
import json
import shutil
import random
from datetime import date, timedelta

import pandas as pd
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "evaluation" / "benchmark"

DEV_DIR = BENCHMARK / "data_dev"
HOLDOUT_DIR = BENCHMARK / "data_holdout"

RISK_TYPES = [
    "cash_conversion",
    "customer_concentration",
    "supplier_concentration",
    "debt_maturity",
    "contract_expiry",
    "threshold_clustering",
]

DEV_CASES = 40
HOLDOUT_CASES = 30

# Different seeds deliberately separate development and holdout data.
DEV_SEED = 123
HOLDOUT_SEED = 987654


# ------------------------------------------------------------
# Financial data
# ------------------------------------------------------------

def make_financials(risk_type=None, rng=None):

    rng = rng or random.Random()

    revenue_2023 = rng.uniform(180, 350) * 1e7
    revenue_2024 = revenue_2023 * rng.uniform(1.08, 1.32)

    if risk_type == "cash_conversion":
        ocf_2023 = rng.uniform(25, 45) * 1e7
        ocf_2024 = ocf_2023 * rng.uniform(0.55, 0.88)
    else:
        ocf_2023 = rng.uniform(25, 45) * 1e7
        ocf_2024 = ocf_2023 * rng.uniform(1.05, 1.25)

    return pd.DataFrame({
        "year": [2023, 2024],
        "revenue": [revenue_2023, revenue_2024],
        "cogs": [
            revenue_2023 * 0.62,
            revenue_2024 * 0.62,
        ],
        "operating_expenses": [
            revenue_2023 * 0.20,
            revenue_2024 * 0.20,
        ],
        "depreciation": [
            revenue_2023 * 0.03,
            revenue_2024 * 0.03,
        ],
        "interest_expense": [
            revenue_2023 * 0.02,
            revenue_2024 * 0.02,
        ],
        "net_income": [
            revenue_2023 * 0.10,
            revenue_2024 * 0.09,
        ],
        "operating_cash_flow": [
            ocf_2023,
            ocf_2024,
        ],
        "capex": [
            revenue_2023 * 0.04,
            revenue_2024 * 0.04,
        ],
        "current_assets": [
            revenue_2023 * 0.40,
            revenue_2024 * 0.40,
        ],
        "inventory": [
            revenue_2023 * 0.08,
            revenue_2024 * 0.08,
        ],
        "receivables": [
            revenue_2023 * 0.18,
            revenue_2024 * 0.18,
        ],
        "current_liabilities": [
            revenue_2023 * 0.25,
            revenue_2024 * 0.25,
        ],
        "payables": [
            revenue_2023 * 0.10,
            revenue_2024 * 0.10,
        ],
        "total_assets": [
            revenue_2023,
            revenue_2024,
        ],
        "total_debt": [
            revenue_2023 * 0.20,
            revenue_2024 * 0.20,
        ],
        "total_equity": [
            revenue_2023 * 0.40,
            revenue_2024 * 0.40,
        ],
    })


# ------------------------------------------------------------
# Customers
# ------------------------------------------------------------

def make_customers(risk_type=None, rng=None):

    rng = rng or random.Random()

    total = rng.uniform(250, 400) * 1e7

    if risk_type == "customer_concentration":

        top = rng.uniform(0.32, 0.48)

        remaining = 1 - top

        shares = [
            top,
            remaining * 0.30,
            remaining * 0.25,
            remaining * 0.20,
            remaining * 0.15,
            remaining * 0.10,
        ]

    elif risk_type == "contract_expiry":

    # Customer A = 15%.
    # Remaining 85% distributed evenly across 10 customers.
    # Each remaining customer = 8.5%.
    #
    # Top 1 = 15%
    # Top 5 = 15 + 8.5 + 8.5 + 8.5 + 8.5 = 49%
    #
    # Therefore:
    # - Contract-expiry detector sees Customer A (>10%)
    # - Customer concentration detector does NOT trigger.

        shares = [
        0.15,
        0.085,
        0.085,
        0.085,
        0.085,
        0.085,
        0.085,
        0.085,
        0.085,
        0.085,
        0.085,
    ]

    else:

        # Clean customer distribution.
        #
        # 10 customers at 8.5% each = 85%
        # Final customer = 15%
        #
        # Top 1 = 15%  -> below 20% threshold
        # Top 5 = 42.5% -> below 50% threshold
        #
        # Therefore this is safely below the
        # customer concentration detector threshold.

        shares = [
            0.085,
            0.085,
            0.085,
            0.085,
            0.085,
            0.085,
            0.085,
            0.085,
            0.085,
            0.085,
            0.15,
        ]

    customers = [
        "Customer A",
        "Customer B",
        "Customer C",
        "Customer D",
        "Customer E",
        "Customer F",
        "Customer G",
        "Customer H",
        "Customer I",
        "Customer J",
        "Customer K",
    ]

    return pd.DataFrame({
        "customer": customers[:len(shares)],
        "revenue": [
            total * share for share in shares
        ],
    })


# ------------------------------------------------------------
# Suppliers
# ------------------------------------------------------------

def make_suppliers(risk_type=None, rng=None):

    rng = rng or random.Random()

    total = rng.uniform(180, 280) * 1e7

    if risk_type == "supplier_concentration":
        main_share = rng.uniform(0.35, 0.50)
    else:
        main_share = rng.uniform(0.10, 0.20)

    remaining = total * (1 - main_share)

    return pd.DataFrame({
        "supplier": [
            "Supplier A",
            "Supplier B",
            "Supplier C",
            "Supplier D",
            "Others",
        ],
        "procurement": [
            total * main_share,
            remaining * 0.25,
            remaining * 0.20,
            remaining * 0.15,
            remaining * 0.40,
        ],
    })


# ------------------------------------------------------------
# Debt
# ------------------------------------------------------------

def make_debt(risk_type=None, rng=None):

    rng = rng or random.Random()

    today = date.today()

    if risk_type == "debt_maturity":

        first_days = rng.randint(60, 300)
        second_days = rng.randint(500, 900)

    else:

        first_days = rng.randint(500, 800)
        second_days = rng.randint(900, 1500)

    return pd.DataFrame({
        "lender": ["Bank A", "Bank B"],
        "principal": [
            rng.uniform(20, 35) * 1e7,
            rng.uniform(15, 30) * 1e7,
        ],
        "maturity_date": [
            today + timedelta(days=first_days),
            today + timedelta(days=second_days),
        ],
        "interest_rate": [
            rng.uniform(0.085, 0.105),
            rng.uniform(0.100, 0.125),
        ],
    })


# ------------------------------------------------------------
# Contracts
# ------------------------------------------------------------

def make_contracts(risk_type=None, rng=None):

    rng = rng or random.Random()

    today = date.today()

    if risk_type == "contract_expiry":

        expiry = today + timedelta(
            days=rng.randint(60, 180)
        )

    else:

        expiry = today + timedelta(
            days=rng.randint(500, 1000)
        )

    return pd.DataFrame({
        "contract_id": ["A01", "B02"],
        "counterparty": [
            "Customer A",
            "Supplier A",
        ],
        "type": [
            "customer",
            "supplier",
        ],
        "value": [
            rng.uniform(70, 130) * 1e7,
            rng.uniform(30, 70) * 1e7,
        ],
        "end_date": [
            expiry,
            today + timedelta(days=rng.randint(600, 900)),
        ],
        "change_of_control": [
            False,
            False,
        ],
    })


# ------------------------------------------------------------
# Transactions
# ------------------------------------------------------------

def make_transactions(risk_type=None, rng=None, np_rng=None):

    rng = rng or random.Random()
    np_rng = np_rng or np.random.default_rng()

    today = date.today()

    n = rng.randint(450, 700)

    vendors = [
        f"V-{i:02d}"
        for i in range(1, 16)
    ]

    tx = pd.DataFrame({
        "date": [
            today - timedelta(days=rng.randint(0, 365))
            for _ in range(n)
        ],
        "vendor": np_rng.choice(vendors, n),
        "amount": np.round(
            np_rng.lognormal(
                mean=10,
                sigma=0.9,
                size=n,
            ),
            2,
        ),
    })

    if risk_type == "threshold_clustering":

        suspicious = pd.DataFrame({
            "date": [
                today - timedelta(days=rng.randint(0, 120))
                for _ in range(30)
            ],
            "vendor": ["V-99"] * 30,
            "amount": np.round(
                np_rng.uniform(
                    48500,
                    49950,
                    30,
                ),
                2,
            ),
        })

        tx = pd.concat(
            [tx, suspicious],
            ignore_index=True,
        )

    return tx


# ------------------------------------------------------------
# Create one case
# ------------------------------------------------------------

def create_case(
    base_dir,
    case_id,
    risk_type=None,
    seed=123,
):

    case_dir = base_dir / case_id

    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    make_financials(
        risk_type,
        rng,
    ).to_csv(
        case_dir / "financials.csv",
        index=False,
    )

    make_customers(
        risk_type,
        rng,
    ).to_csv(
        case_dir / "customers.csv",
        index=False,
    )

    make_suppliers(
        risk_type,
        rng,
    ).to_csv(
        case_dir / "suppliers.csv",
        index=False,
    )

    make_debt(
        risk_type,
        rng,
    ).to_csv(
        case_dir / "debt_schedule.csv",
        index=False,
    )

    make_contracts(
        risk_type,
        rng,
    ).to_csv(
        case_dir / "contracts.csv",
        index=False,
    )

    make_transactions(
        risk_type,
        rng,
        np_rng,
    ).to_csv(
        case_dir / "transactions.csv",
        index=False,
    )


# ------------------------------------------------------------
# Generate benchmark split
# ------------------------------------------------------------

def generate_split(base_dir, total_cases, seed, split_name):

    if base_dir.exists():
        shutil.rmtree(base_dir)

    base_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    ground_truth = []

    # Positive cases:
    # evenly distributed across risk types.
    positive_cases = total_cases * 4 // 5

    for i in range(positive_cases):

        risk_type = RISK_TYPES[
            i % len(RISK_TYPES)
        ]

        case_id = f"{split_name.upper()}_POS_{i + 1:02d}"

        case_seed = seed + i * 7919

        create_case(
            base_dir,
            case_id,
            risk_type,
            case_seed,
        )

        ground_truth.append({
            "case_id": case_id,
            "risk_present": True,
            "risk_type": risk_type,
        })

    # Negative / clean cases
    negative_cases = total_cases - positive_cases

    for i in range(negative_cases):

        case_id = f"{split_name.upper()}_NEG_{i + 1:02d}"

        case_seed = seed + 10000 + i * 7919

        create_case(
            base_dir,
            case_id,
            None,
            case_seed,
        )

        ground_truth.append({
            "case_id": case_id,
            "risk_present": False,
            "risk_type": None,
        })

    return ground_truth


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("DEALLENS BENCHMARK GENERATION")
    print("=" * 60)

    dev_ground_truth = generate_split(
        DEV_DIR,
        DEV_CASES,
        DEV_SEED,
        "dev",
    )

    holdout_ground_truth = generate_split(
        HOLDOUT_DIR,
        HOLDOUT_CASES,
        HOLDOUT_SEED,
        "holdout",
    )

    with open(
        BENCHMARK / "ground_truth_dev.json",
        "w",
    ) as f:
        json.dump(
            dev_ground_truth,
            f,
            indent=2,
        )

    with open(
        BENCHMARK / "ground_truth_holdout.json",
        "w",
    ) as f:
        json.dump(
            holdout_ground_truth,
            f,
            indent=2,
        )

    print()
    print("Development set :")
    print(f"  Cases : {DEV_CASES}")
    print(f"  Seed  : {DEV_SEED}")

    print()
    print("Independent holdout :")
    print(f"  Cases : {HOLDOUT_CASES}")
    print(f"  Seed  : {HOLDOUT_SEED}")

    print()
    print(f"Development data : {DEV_DIR}")
    print(f"Holdout data     : {HOLDOUT_DIR}")


if __name__ == "__main__":
    main()