"""Drift monitoring: PSI per feature and on the predictions, plus a retrain rule.

PSI (population stability index) = Σ (cur% − ref%) · ln(cur% / ref%) over bins. Common reading:
< 0.10 stable, 0.10–0.25 moderate shift, > 0.25 significant shift.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .schema import FEATURES, NUMERIC, TARGET

EPS = 1e-4


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().to_numpy(float), cur.dropna().to_numpy(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:                       # (almost) constant reference: compare value shares instead
        return psi_categorical(pd.Series(ref), pd.Series(cur))
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    cats = sorted(set(ref.dropna().unique()) | set(cur.dropna().unique()), key=str)
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(model, ref: pd.DataFrame, test: pd.DataFrame, batch: pd.DataFrame, p: dict) -> dict:
    feats = []
    for col in FEATURES:
        value = psi_numeric(ref[col], batch[col]) if col in NUMERIC else psi_categorical(ref[col], batch[col])
        status = "alert" if value > p["psi_retrain"] else "warn" if value > p["psi_warn"] else "ok"
        feats.append({"feature": col, "kind": "numeric" if col in NUMERIC else "categorical", "psi": round(value, 4),
                      "status": status})
    ref_pred = model.predict_proba(ref[FEATURES])[:, 1]
    cur_pred = model.predict_proba(batch[FEATURES])[:, 1]
    pred_psi = psi_numeric(pd.Series(ref_pred), pd.Series(cur_pred))
    test_auc = roc_auc_score(test[TARGET], model.predict_proba(test[FEATURES])[:, 1])
    batch_auc = roc_auc_score(batch[TARGET], cur_pred) if TARGET in batch else None
    drifted = [f["feature"] for f in feats if f["status"] == "alert"]
    reasons = []
    if len(drifted) > p["max_features_drifted"]:
        reasons.append(f"{len(drifted)} features with PSI > {p['psi_retrain']}: {', '.join(drifted)}")
    if pred_psi > p["psi_retrain"]:
        reasons.append(f"prediction PSI {pred_psi:.3f} > {p['psi_retrain']}")
    if batch_auc is not None and test_auc - batch_auc > p["max_auc_drop"]:
        reasons.append(f"ROC AUC fell from {test_auc:.3f} to {batch_auc:.3f} once outcomes arrived")
    if TARGET in batch and abs(batch[TARGET].mean() - cur_pred.mean()) > p["max_calibration_gap"]:
        reasons.append(f"observed churn {batch[TARGET].mean():.1%} vs predicted {cur_pred.mean():.1%}")
    return {
        "features": feats, "prediction_psi": round(pred_psi, 4),
        "mean_predicted_churn_reference": round(float(ref_pred.mean()), 4),
        "mean_predicted_churn_batch": round(float(cur_pred.mean()), 4),
        "observed_churn_batch": round(float(batch[TARGET].mean()), 4) if TARGET in batch else None,
        "roc_auc_test": round(float(test_auc), 4), "roc_auc_batch": round(float(batch_auc), 4) if batch_auc else None,
        "batch_rows": int(len(batch)), "retrain": bool(reasons), "reasons": reasons,
        "prediction_hist": {"reference": np.histogram(ref_pred, bins=20, range=(0, 1))[0].tolist(),
                            "batch": np.histogram(cur_pred, bins=20, range=(0, 1))[0].tolist()},
    }
