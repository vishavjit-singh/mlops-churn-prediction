"""The input contract: training, the prepare-stage checks and the API all use this one definition."""
from __future__ import annotations

import pandas as pd

TARGET = "churned"
ID = "customer_id"

NUMERIC = {
    # column: (min, max) allowed
    "tenure_months": (0, 72),
    "monthly_charges": (0, 400),
    "total_charges": (0, 30000),
    "support_calls_90d": (0, 30),
    "data_usage_gb": (0, 500),
    "num_addons": (0, 6),
    "late_payments_12m": (0, 12),
    "senior": (0, 1),
    "has_partner": (0, 1),
}
CATEGORICAL = {
    "contract": ("month-to-month", "one-year", "two-year"),
    "internet_service": ("fiber", "dsl", "none"),
    "payment_method": ("card_autopay", "bank_autopay", "upi", "cash"),
    "region": ("north", "south", "east", "west"),
}
FEATURES = list(NUMERIC) + list(CATEGORICAL)


class SchemaError(ValueError):
    pass


def validate(df: pd.DataFrame, require_target: bool = True) -> dict:
    """Raise SchemaError on contract breaks; return soft findings (nulls, unseen categories) as a report."""
    needed = [ID, *FEATURES] + ([TARGET] if require_target else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if df[ID].duplicated().any():
        raise SchemaError(f"{int(df[ID].duplicated().sum())} duplicate {ID} values")
    report: dict = {"rows": int(len(df)), "nulls": {}, "out_of_range": {}, "unseen_categories": {}}
    for col, (lo, hi) in NUMERIC.items():
        s = pd.to_numeric(df[col], errors="coerce")
        if s.isna().any():
            report["nulls"][col] = int(s.isna().sum())
        bad = int(((s < lo) | (s > hi)).sum())
        if bad:
            report["out_of_range"][col] = bad
    for col, allowed in CATEGORICAL.items():
        unseen = sorted(set(df[col].dropna().unique()) - set(allowed))
        if unseen:
            report["unseen_categories"][col] = unseen
    if require_target:
        labels = set(df[TARGET].dropna().unique())
        if not labels <= {0, 1}:
            raise SchemaError(f"{TARGET} must be 0/1, found {sorted(labels)}")
        report["churn_rate"] = round(float(df[TARGET].mean()), 4)
    if report["out_of_range"]:
        raise SchemaError(f"values out of range: {report['out_of_range']}")
    return report
