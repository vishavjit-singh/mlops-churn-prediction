"""SYNTHETIC telecom customers with known churn drivers (seeded, reproducible).

The label is drawn from a logistic model whose coefficients are written below, so we know what a
good model should find: contract type, tenure, support calls, price and autopay matter; region
does not. ``drifted=True`` produces a later batch after a price rise and a network outage (data drift)
in which a rival fibre offer also makes fibre and price-sensitive customers leave faster (concept drift).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_EFFECTS = {
    "contract=month-to-month": 1.3, "contract=two-year": -1.3, "tenure_months (per month)": -0.035,
    "internet_service=fiber": 0.45, "support_calls_90d (per call)": 0.32, "monthly_charges (per ₹100)": 0.9,
    "autopay": -0.55, "late_payments_12m (per payment)": 0.25, "num_addons (per add-on)": -0.18,
    "senior": 0.25, "region": 0.0,
}


def generate(n: int, seed: int, drifted: bool = False, id_offset: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n,
                          p=[0.62, 0.22, 0.16] if drifted else [0.52, 0.26, 0.22])
    tenure = np.where(contract == "month-to-month", rng.gamma(1.6, 9, n), rng.gamma(3.0, 11, n))
    tenure = np.clip(np.round(tenure), 0, 72).astype(int)
    internet = rng.choice(["fiber", "dsl", "none"], n, p=[0.45, 0.38, 0.17])
    addons = np.where(internet == "none", 0, rng.binomial(5, 0.35, n))
    base = np.select([internet == "fiber", internet == "dsl"], [95.0, 65.0], 25.0)
    monthly = base + addons * rng.uniform(8, 15, n) + rng.normal(0, 6, n)
    if drifted:
        monthly *= 1.12                                       # price rise
    monthly = np.clip(np.round(monthly, 2), 10, 380)
    payment = rng.choice(["card_autopay", "bank_autopay", "upi", "cash"], n, p=[0.28, 0.22, 0.32, 0.18])
    support = rng.poisson(2.6 if drifted else 1.2, n)        # outage → more calls
    late = rng.binomial(12, np.where(payment == "cash", 0.12, 0.04))
    senior = rng.binomial(1, 0.16, n)
    partner = rng.binomial(1, 0.48, n)
    region = rng.choice(["north", "south", "east", "west"], n)
    usage = np.where(internet == "none", 0.0, rng.gamma(2.2, 45, n))
    usage = np.round(np.clip(usage, 0, 500), 1)
    total = np.round(np.clip(monthly * tenure * rng.uniform(0.95, 1.02, n), 0, 30000), 2)

    autopay = np.isin(payment, ["card_autopay", "bank_autopay"]).astype(float)
    logit = (-1.6 + 1.3 * (contract == "month-to-month") - 1.3 * (contract == "two-year")
             - 0.035 * tenure + (1.1 if drifted else 0.45) * (internet == "fiber") + 0.32 * support
             + (1.6 if drifted else 0.9) * (monthly - 70) / 100
             - 0.55 * autopay + 0.25 * late - 0.18 * addons + 0.25 * senior)
    churned = rng.binomial(1, 1 / (1 + np.exp(-logit)))

    df = pd.DataFrame({
        "customer_id": [f"C{id_offset + i:06d}" for i in range(n)],
        "tenure_months": tenure, "contract": contract, "internet_service": internet, "payment_method": payment,
        "monthly_charges": monthly, "total_charges": total, "support_calls_90d": support,
        "data_usage_gb": usage, "num_addons": addons, "late_payments_12m": late, "senior": senior,
        "has_partner": partner, "region": region, "churned": churned,
    })
    # realistic gaps: usage meter missing for ~2%, total charges blank for brand-new customers
    df.loc[rng.random(n) < 0.02, "data_usage_gb"] = np.nan
    df.loc[df["tenure_months"] == 0, "total_charges"] = np.nan
    return df
