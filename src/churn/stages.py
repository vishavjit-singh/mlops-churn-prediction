"""The DVC stages. Each one reads files, writes files, and is declared in dvc.yaml with its deps and outs."""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import (average_precision_score, brier_score_loss, log_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from . import config
from .features import build
from .generate import generate as make_customers
from .monitoring import drift_report
from .schema import FEATURES, ID, TARGET, validate
from .tracking import make_tracker

log = logging.getLogger("churn")


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------- generate
def stage_generate(params: dict) -> None:
    p = params["generate"]
    out = config.DATA / "raw" / "customers.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    make_customers(p["n_customers"], p["seed"]).to_csv(out, index=False)
    log.info("wrote %s (SYNTHETIC, seed %s)", out, p["seed"])


# ----------------------------------------------------------------------------------------------- prepare
def stage_prepare(params: dict) -> None:
    p = params["prepare"]
    df = pd.read_csv(config.DATA / "raw" / "customers.csv")
    report = validate(df)
    train, test = train_test_split(df, test_size=p["test_size"], random_state=p["seed"], stratify=df[TARGET])
    out = config.DATA / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    report.update(train_rows=len(train), test_rows=len(test), train_churn_rate=round(float(train[TARGET].mean()), 4),
                  test_churn_rate=round(float(test[TARGET].mean()), 4))
    _write_json(config.REPORTS / "data_report.json", report)
    # DVC metrics: flat numbers only
    _write_json(config.METRICS / "data.json", {k: report[k] for k in ("rows", "churn_rate", "train_rows", "test_rows",
                                                                      "train_churn_rate", "test_churn_rate")}
                | {f"nulls_{k}": v for k, v in report["nulls"].items()})
    log.info("prepared: %s", report)


# ------------------------------------------------------------------------------------------------- train
def stage_train(params: dict) -> None:
    p = params["train"]
    train_path = config.DATA / "prepared" / "train.csv"
    train = pd.read_csv(train_path)
    X, y = train[FEATURES], train[TARGET]
    cv = StratifiedKFold(n_splits=p["cv_folds"], shuffle=True, random_state=p["seed"])
    tracker = make_tracker(config.ROOT)
    results = {}
    for name, hp in p["candidates"].items():
        with tracker.start_run(f"cv-{name}"):
            scores = cross_validate(build(name, hp, p["seed"]), X, y, cv=cv,
                                    scoring={"roc_auc": "roc_auc", "pr_auc": "average_precision",
                                             "brier": "neg_brier_score"})
            res = {"roc_auc": float(np.mean(scores["test_roc_auc"])),
                   "roc_auc_std": float(np.std(scores["test_roc_auc"])),
                   "pr_auc": float(np.mean(scores["test_pr_auc"])),
                   "brier": float(-np.mean(scores["test_brier"])),
                   "fit_seconds": float(np.mean(scores["fit_time"]))}
            tracker.log_params({"model": name, **hp, "cv_folds": p["cv_folds"]})
            tracker.log_metrics({f"cv_{k}": v for k, v in res.items()})
            results[name] = res
            log.info("%-24s cv roc_auc %.4f ± %.4f  pr_auc %.4f", name, res["roc_auc"], res["roc_auc_std"],
                     res["pr_auc"])
    best = max(results, key=lambda n: results[n][p["primary_metric"]])
    model = build(best, p["candidates"][best], p["seed"]).fit(X, y)
    config.MODELS.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS / "model.joblib"
    joblib.dump(model, model_path)
    with tracker.start_run(f"champion-{best}"):
        tracker.log_params({"model": best, **p["candidates"][best], "train_rows": len(train),
                            "train_data_sha": _sha(train_path)})
        tracker.log_metrics({f"cv_{k}": v for k, v in results[best].items()})
        version = tracker.log_model(model_path, "churn-classifier", "champion", model=model,
                                    input_example=X.head(3))
    meta = {"model_name": "churn-classifier", "version": version, "alias": "champion", "algorithm": best,
            "hyperparameters": p["candidates"][best], "features": FEATURES, "train_rows": len(train),
            "train_data_sha": _sha(train_path), "tracker": tracker.kind,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "cv": results[best]}
    _write_json(config.MODELS / "metadata.json", meta)
    _write_json(config.METRICS / "train.json", {"best": best, "candidates": results})
    log.info("champion: %s (registered version %s via %s tracker)", best, version, tracker.kind)


# ---------------------------------------------------------------------------------------------- evaluate
def best_threshold(y: np.ndarray, prob: np.ndarray, cost_fn: float, cost_fp: float) -> tuple[float, pd.DataFrame]:
    """Threshold that minimises expected cost per customer; returns it and the full cost curve."""
    rows = []
    for t in np.round(np.arange(0.02, 0.99, 0.01), 2):
        pred = prob >= t
        fn = int(((~pred) & (y == 1)).sum())
        fp = int((pred & (y == 0)).sum())
        tp = int((pred & (y == 1)).sum())
        rows.append({"threshold": float(t), "cost_per_customer": (fn * cost_fn + fp * cost_fp) / len(y),
                     "recall": tp / max(1, int((y == 1).sum())),
                     "precision": tp / int(pred.sum()) if pred.sum() else float("nan"),
                     "flagged_share": float(pred.mean())})
    curve = pd.DataFrame(rows)
    best = curve.loc[curve["cost_per_customer"].idxmin()]
    return float(best["threshold"]), curve


def stage_evaluate(params: dict) -> None:
    p = params["evaluate"]
    test = pd.read_csv(config.DATA / "prepared" / "test.csv")
    model = joblib.load(config.MODELS / "model.joblib")
    X, y = test[FEATURES], test[TARGET].to_numpy()
    prob = model.predict_proba(X)[:, 1]
    threshold, curve = best_threshold(y, prob, p["cost_false_negative"], p["cost_false_positive"])
    at = curve.set_index("threshold")
    default_cost = float(at.loc[0.5, "cost_per_customer"])
    order = np.argsort(-prob)
    top = order[: len(order) // 10]
    metrics = {
        "roc_auc": roc_auc_score(y, prob), "pr_auc": average_precision_score(y, prob),
        "brier": brier_score_loss(y, prob), "log_loss": log_loss(y, prob), "base_rate": float(y.mean()),
        "threshold": threshold, "precision_at_threshold": float(at.loc[threshold, "precision"]),
        "recall_at_threshold": float(at.loc[threshold, "recall"]),
        "flagged_share_at_threshold": float(at.loc[threshold, "flagged_share"]),
        "cost_per_customer_at_threshold": float(at.loc[threshold, "cost_per_customer"]),
        "cost_per_customer_at_0_5": default_cost,
        "cost_per_customer_no_model": float(y.mean() * p["cost_false_negative"]),
        "top_decile_lift": float(y[top].mean() / y.mean()),
        "test_rows": int(len(y)),
    }
    metrics = {k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()}
    _write_json(config.METRICS / "eval.json", metrics)

    config.PLOTS.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y, prob)
    pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(config.PLOTS / "roc.csv", index=False)
    prec, rec, _ = precision_recall_curve(y, prob)
    pd.DataFrame({"recall": rec, "precision": prec}).to_csv(config.PLOTS / "pr.csv", index=False)
    frac, mean_pred = calibration_curve(y, prob, n_bins=p["calibration_bins"], strategy="quantile")
    pd.DataFrame({"mean_predicted": mean_pred, "fraction_churned": frac}).to_csv(config.PLOTS / "calibration.csv",
                                                                                 index=False)
    curve.to_csv(config.PLOTS / "threshold_cost.csv", index=False)
    imp = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=5, random_state=0, n_jobs=-1)
    pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean, "std": imp.importances_std}) \
        .sort_values("importance", ascending=False).to_csv(config.PLOTS / "importance.csv", index=False)

    # a separate file, so each DVC output has exactly one producing stage
    _write_json(config.MODELS / "evaluation.json", {"threshold": threshold, "test_metrics": metrics,
                                                    "costs": {"false_negative": p["cost_false_negative"],
                                                              "false_positive": p["cost_false_positive"]}})
    log.info("test roc_auc %.4f pr_auc %.4f threshold %.2f (cost %.1f vs %.1f at 0.5)", metrics["roc_auc"],
             metrics["pr_auc"], threshold, metrics["cost_per_customer_at_threshold"], default_cost)


# ----------------------------------------------------------------------------------------------- monitor
def stage_monitor(params: dict) -> None:
    p = params["monitor"]
    ref = pd.read_csv(config.DATA / "prepared" / "train.csv")
    test = pd.read_csv(config.DATA / "prepared" / "test.csv")
    batch = make_customers(p["n_customers"], p["seed"], drifted=True, id_offset=900_000)
    out = config.DATA / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    batch.drop(columns=[TARGET]).to_csv(out / "batch.csv", index=False)
    batch[[ID, TARGET]].to_csv(out / "batch_outcomes.csv", index=False)
    model = joblib.load(config.MODELS / "model.joblib")
    report = drift_report(model, ref, test, batch, p)
    _write_json(config.REPORTS / "drift_report.json", report)
    _write_json(config.METRICS / "drift.json", {
        "retrain": int(report["retrain"]), "prediction_psi": report["prediction_psi"],
        "max_feature_psi": max(f["psi"] for f in report["features"]),
        "features_alerted": sum(f["status"] == "alert" for f in report["features"]),
        "roc_auc_batch": report["roc_auc_batch"], "observed_churn_batch": report["observed_churn_batch"],
        "mean_predicted_churn_batch": report["mean_predicted_churn_batch"]})
    pd.DataFrame(report["features"]).to_csv(config.PLOTS / "drift.csv", index=False)
    log.info("drift: retrain=%s reasons=%s", report["retrain"], report["reasons"])
