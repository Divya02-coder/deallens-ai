"""STEP 5b - Isolation Forest on transactions (unsupervised outlier scoring)."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest


def score_transactions(tx: pd.DataFrame, contamination: float = 0.03, seed: int = 42) -> pd.DataFrame:
    df = tx.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["log_amount"] = np.log1p(df["amount"])
    df["dow"] = df["date"].dt.dayofweek
    df["vendor_txn_count"] = df.groupby("vendor")["amount"].transform("count")
    df["vendor_amt_z"] = df.groupby("vendor")["amount"].transform(lambda s: (s - s.mean()) / (s.std() or 1))
    df["near_round"] = (df["amount"] % 1000 < 1).astype(int)
    feats = ["log_amount", "dow", "vendor_txn_count", "vendor_amt_z", "near_round"]
    model = IsolationForest(contamination=contamination, random_state=seed, n_estimators=200)
    model.fit(df[feats])
    df["anomaly_score"] = -model.score_samples(df[feats])   # higher = more anomalous
    df["is_outlier"] = model.predict(df[feats]) == -1
    return df.sort_values("anomaly_score", ascending=False)


def outlier_findings(scored: pd.DataFrame, top_n: int = 10) -> list[dict]:
    out = scored[scored["is_outlier"]]
    if out.empty:
        return []
    top = out.head(top_n)
    return [{
        "id": "iforest", "category": "Transactions", "severity": "Medium",
        "title": f"{len(out)} statistically unusual transactions",
        "detail": f"Isolation Forest flagged {len(out)} of {len(scored)} payments. Highest-scoring vendor(s): "
                  f"{', '.join(top['vendor'].astype(str).unique()[:3])}.",
        "evidence": [{"source": "transactions.csv", "rows": int(len(out))}],
        "questions": ["Review the top-scored transactions with the finance team."],
    }]
