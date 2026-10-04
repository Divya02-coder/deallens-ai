
from pathlib import Path
import pandas as pd
import zipfile
import shutil

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "stress_test_pack"
OUT.mkdir(parents=True, exist_ok=True)

def save(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"Created: {path}")

# ------------------------------------------------------------
# 1. FINANCIALS
# Preserve the project's exact existing schema.
# ------------------------------------------------------------
src = RAW / "financials.csv"
if src.exists():
    df = pd.read_csv(src)

    # Modify only columns that actually exist.
    if "revenue" in df.columns:
        last = df.index[-1]
        df.loc[last, "revenue"] *= 0.88

    if "operating_cash_flow" in df.columns:
        last = df.index[-1]
        df.loc[last, "operating_cash_flow"] *= 0.68

    if "receivables" in df.columns:
        last = df.index[-1]
        df.loc[last, "receivables"] *= 1.25

    if "inventory" in df.columns:
        last = df.index[-1]
        df.loc[last, "inventory"] *= 1.20

    save(df, "financials.csv")
else:
    print("WARNING: financials.csv not found")

# ------------------------------------------------------------
# 2. CUSTOMERS
# Create a stronger concentration case while preserving schema.
# ------------------------------------------------------------
src = RAW / "customers.csv"
if src.exists():
    df = pd.read_csv(src)

    if "revenue" in df.columns and len(df) >= 3:
        total = df["revenue"].sum()

        # Make the first customer materially dominant.
        df.loc[df.index[0], "revenue"] = total * 0.42

        remaining = total * 0.58
        other_idx = list(df.index[1:])
        if other_idx:
            weights = pd.Series(
                range(len(other_idx), 0, -1),
                index=other_idx,
                dtype=float,
            )
            weights = weights / weights.sum()
            df.loc[other_idx, "revenue"] = remaining * weights

    save(df, "customers.csv")
else:
    print("WARNING: customers.csv not found")

# ------------------------------------------------------------
# 3. SUPPLIERS
# Create a high supplier-dependency case.
# ------------------------------------------------------------
src = RAW / "suppliers.csv"
if src.exists():
    df = pd.read_csv(src)

    if "procurement" in df.columns and len(df) >= 3:
        total = df["procurement"].sum()

        df.loc[df.index[0], "procurement"] = total * 0.48

        remaining = total * 0.52
        other_idx = list(df.index[1:])
        if other_idx:
            weights = pd.Series(
                range(len(other_idx), 0, -1),
                index=other_idx,
                dtype=float,
            )
            weights = weights / weights.sum()
            df.loc[other_idx, "procurement"] = remaining * weights

    save(df, "suppliers.csv")
else:
    print("WARNING: suppliers.csv not found")

# ------------------------------------------------------------
# 4. DEBT SCHEDULE
# Move the earliest maturity dates closer to today.
# ------------------------------------------------------------
src = RAW / "debt_schedule.csv"
if src.exists():
    df = pd.read_csv(src)

    if "maturity_date" in df.columns and len(df) >= 2:
        dates = pd.to_datetime(
            df["maturity_date"],
            errors="coerce"
        )

        valid = dates.dropna().sort_values().index

        if len(valid) >= 1:
            df.loc[valid[0], "maturity_date"] = (
                pd.Timestamp.today()
                + pd.Timedelta(days=90)
            ).strftime("%Y-%m-%d")

        if len(valid) >= 2:
            df.loc[valid[1], "maturity_date"] = (
                pd.Timestamp.today()
                + pd.Timedelta(days=180)
            ).strftime("%Y-%m-%d")

    save(df, "debt_schedule.csv")
else:
    print("WARNING: debt_schedule.csv not found")

# ------------------------------------------------------------
# 5. CONTRACTS
# Bring an existing contract expiry closer and preserve schema.
# ------------------------------------------------------------
src = RAW / "contracts.csv"
if src.exists():
    df = pd.read_csv(src)

    date_candidates = [
        "end_date",
        "expiry_date",
        "contract_end",
        "renewal_date",
    ]

    changed = False

    for col in date_candidates:
        if col in df.columns and len(df) > 0:
            df.loc[df.index[0], col] = (
                pd.Timestamp.today()
                + pd.Timedelta(days=75)
            ).strftime("%Y-%m-%d")
            changed = True
            break

    # If there is a change-of-control field, make the first
    # contract explicitly exposed.
    for col in [
        "change_of_control",
        "change_of_control_clause",
        "change_of_control_rights",
    ]:
        if col in df.columns and len(df) > 0:
            df.loc[df.index[0], col] = "Yes"
            changed = True
            break

    save(df, "contracts.csv")
else:
    print("WARNING: contracts.csv not found")

# ------------------------------------------------------------
# Create ZIP
# ------------------------------------------------------------
zip_path = ROOT / "DealLens_Stress_Test_Pack.zip"

with zipfile.ZipFile(
    zip_path,
    "w",
    compression=zipfile.ZIP_DEFLATED
) as z:
    for csv_file in OUT.glob("*.csv"):
        z.write(
            csv_file,
            arcname=csv_file.name
        )

print()
print("=" * 60)
print("STRESS TEST PACK READY")
print("=" * 60)
print(f"Folder: {OUT}")
print(f"ZIP:    {zip_path}")
print()
print("Upload these CSVs to DealLens AI:")
print("  financials.csv")
print("  customers.csv")
print("  suppliers.csv")
print("  debt_schedule.csv")
print("  contracts.csv")
print()
print("Keep your existing transactions.csv.")
