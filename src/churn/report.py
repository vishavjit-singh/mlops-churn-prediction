"""README images, rendered from metrics/*.json and plots/*.csv written by the DVC stages."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import config  # noqa: E402
from .generate import TRUE_EFFECTS  # noqa: E402

INK, MUTED, GRID = "#0e1a2b", "#5b6676", "#e3e7ee"
ACCENT, SIGNAL, BAD, GOOD, PURPLE = "#2a78d6", "#eda100", "#e34948", "#1baf7a", "#6250d6"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.titleweight": "bold", "axes.titlesize": 12.5, "axes.titlecolor": INK,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150,
})


def _j(name: str) -> dict:
    if name == "drift":
        return json.loads((config.REPORTS / "drift_report.json").read_text())
    return json.loads((config.METRICS / f"{name}.json").read_text())


def _save(fig, out: Path, name: str) -> Path:
    fig.tight_layout()
    p = out / name
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


def architecture(out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(13, 5.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color=INK):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc="white",
                                    ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.7, title, ha="center", va="center", fontsize=10.5, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", fontsize=8, color=MUTED, linespacing=1.4)

    def arrow(x1, y1, x2, y2, color=MUTED):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, lw=1.5, color=color))

    ax.text(0.15, 5.7, "mlops-churn-prediction: versioned data → tracked experiments → registered model → API → "
            "drift checks", fontsize=13, weight="bold", color=INK)
    ax.text(0.2, 5.15, "dvc.yaml  (params.yaml drives every stage; `dvc repro` re-runs only what changed)",
            fontsize=8.6, color=PURPLE)
    stages = [("generate", "synthetic customers\nseeded"), ("prepare", "schema contract\nstratified split"),
              ("train", "3 candidates × 5-fold CV\nbest refit on train"),
              ("evaluate", "ROC/PR · calibration\ncost threshold · importance"),
              ("monitor", "PSI per feature +\npredictions · retrain rule")]
    for i, (t, s) in enumerate(stages):
        box(0.2 + i * 2.55, 3.35, 2.2, 1.45, t, s, PURPLE)
        if i:
            arrow(0.2 + i * 2.55 - 0.35, 4.07, 0.2 + i * 2.55, 4.07)
    box(5.3, 0.5, 2.6, 1.6, "MLflow", "params · metrics · artifacts\nRegistry: churn-classifier\nalias @champion",
        ACCENT)
    box(0.2, 0.5, 2.6, 1.6, "metrics/ · plots/",
        "JSON + CSV declared as\nDVC metrics and plots\n(dvc metrics diff)", INK)
    box(8.4, 0.5, 2.0, 1.6, "Prediction API", "Starlette + pydantic\n/predict · /model\n/health", GOOD)
    box(10.85, 0.5, 2.0, 1.6, "Drift report", "metrics/drift.json\nretrain: true/false\n+ reasons", SIGNAL)
    arrow(3.4, 3.35, 2.2, 2.1)
    ax.text(2.95, 2.55, "every stage writes\nmetrics + plots", fontsize=7.6, color=MUTED)
    arrow(6.3, 3.35, 6.6, 2.1, ACCENT)
    arrow(8.6, 3.35, 9.3, 2.1, GOOD)
    ax.text(9.05, 2.75, "model + threshold", fontsize=7.6, color=GOOD)
    arrow(11.5, 3.35, 11.85, 2.1, SIGNAL)
    ax.text(5.3, 0.25, "local JSON tracker when MLflow is not installed (logged as a warning)", fontsize=7.6,
            color=MUTED)
    return _save(fig, out, "architecture.png")


def model_comparison(out: Path) -> Path:
    t = _j("train")
    names = list(t["candidates"])
    c = t["candidates"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, (key, label, better) in zip(axes, [("roc_auc", "CV ROC AUC (± 1 sd)", "higher"),
                                               ("pr_auc", "CV PR AUC", "higher"), ("brier", "CV Brier score",
                                                                                   "lower")]):
        vals = [c[n][key] for n in names]
        err = [c[n]["roc_auc_std"] for n in names] if key == "roc_auc" else None
        colors = [GOOD if n == t["best"] else GRID for n in names]
        ax.barh(names, vals, xerr=err, color=colors, height=0.55, capsize=3, ecolor=MUTED)
        for i, v in enumerate(vals):
            ax.text(v + (err[i] if err else 0), i, f"  {v:.4f}", va="center", fontsize=9, color=INK)
        lo = min(vals) - (max(vals) - min(vals)) * 3 - 0.01
        ax.set_xlim(max(0, lo), max(vals) + (max(vals) - lo) * 0.45)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID)
        ax.set_title(f"{label} · {better} is better", fontsize=11)
        if ax is not axes[0]:
            ax.set_yticks([])
    fig.suptitle(f"Candidate models, 5-fold stratified CV on the training split · champion: {t['best']}",
                 x=0.01, ha="left", fontsize=13, weight="bold", color=INK)
    return _save(fig, out, "model_comparison.png")


def roc_pr_calibration(out: Path) -> Path:
    e = _j("eval")
    roc, pr, cal = (pd.read_csv(config.PLOTS / f) for f in ("roc.csv", "pr.csv", "calibration.csv"))
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(13, 4.1))
    a1.plot(roc["fpr"], roc["tpr"], color=ACCENT, lw=2)
    a1.plot([0, 1], [0, 1], color=GRID, ls="--")
    a1.set_title(f"ROC · AUC {e['roc_auc']:.3f}")
    a1.set_xlabel("false positive rate")
    a1.set_ylabel("true positive rate")
    a2.plot(pr["recall"], pr["precision"], color=PURPLE, lw=2)
    a2.axhline(e["base_rate"], color=GRID, ls="--")
    a2.text(0.02, e["base_rate"] + 0.02, f"base rate {e['base_rate']:.1%}", fontsize=8, color=MUTED)
    a2.set_title(f"Precision–recall · AP {e['pr_auc']:.3f}")
    a2.set_xlabel("recall")
    a2.set_ylabel("precision")
    a3.plot([0, 1], [0, 1], color=GRID, ls="--", label="perfect")
    a3.plot(cal["mean_predicted"], cal["fraction_churned"], color=GOOD, marker="o", lw=2, label="model (deciles)")
    a3.set_title(f"Calibration · Brier {e['brier']:.3f}")
    a3.set_xlabel("mean predicted probability")
    a3.set_ylabel("observed churn rate")
    a3.legend(frameon=False, fontsize=8)
    for a in (a1, a2, a3):
        a.grid(color=GRID)
    fig.suptitle(f"Held-out test set ({e['test_rows']:,} synthetic customers)", x=0.01, ha="left", fontsize=13,
                 weight="bold", color=INK)
    return _save(fig, out, "roc_pr_calibration.png")


def threshold_cost(out: Path, params: dict) -> Path:
    e = _j("eval")
    c = pd.read_csv(config.PLOTS / "threshold_cost.csv")
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(c["threshold"], c["cost_per_customer"], color=INK, lw=2.2, label="expected cost per customer (₹)")
    ax.axvline(e["threshold"], color=GOOD, lw=1.5)
    ax.axvline(0.5, color=BAD, lw=1.2, ls="--")
    ax.scatter([e["threshold"], 0.5], [e["cost_per_customer_at_threshold"], e["cost_per_customer_at_0_5"]],
               color=[GOOD, BAD], zorder=3)
    ax.text(e["threshold"] + 0.01, e["cost_per_customer_at_threshold"] - 4,
            f"chosen {e['threshold']:.2f}: ₹{e['cost_per_customer_at_threshold']:.1f}", color=GOOD, fontsize=9)
    ax.text(0.51, e["cost_per_customer_at_0_5"] + 2, f"default 0.5: ₹{e['cost_per_customer_at_0_5']:.1f}",
            color=BAD, fontsize=9)
    ax.axhline(e["cost_per_customer_no_model"], color=MUTED, ls=":", lw=1)
    ax.text(0.62, e["cost_per_customer_no_model"] - 4, f"no model (nobody gets an offer): "
            f"₹{e['cost_per_customer_no_model']:.1f}", color=MUTED, fontsize=8.5)
    ax2 = ax.twinx()
    ax2.plot(c["threshold"], c["recall"], color=PURPLE, lw=1.3, alpha=0.8, label="recall")
    ax2.plot(c["threshold"], c["precision"], color=SIGNAL, lw=1.3, alpha=0.8, label="precision")
    ax2.set_ylim(0, 1.05)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylabel("recall / precision")
    ax.set_xlabel("decision threshold")
    ax.set_ylabel("₹ per customer")
    ax.grid(color=GRID)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center right")
    ev = params["evaluate"]
    ax.set_title(f"Threshold from business costs (missed churner ₹{ev['cost_false_negative']}, retention offer "
                 f"₹{ev['cost_false_positive']}; illustrative)")
    return _save(fig, out, "threshold_cost.png")


def importance(out: Path) -> Path:
    imp = pd.read_csv(config.PLOTS / "importance.csv").sort_values("importance")
    fig, ax = plt.subplots(figsize=(10, 4.6))
    colors = [MUTED if f == "region" else ACCENT for f in imp["feature"]]
    ax.barh(imp["feature"], imp["importance"], xerr=imp["std"], color=colors, height=0.6, ecolor=MUTED, capsize=2)
    for i, (f, v) in enumerate(zip(imp["feature"], imp["importance"])):
        ax.text(max(v, 0) + 0.002, i, f"{v:.3f}", va="center", fontsize=8.5)
    ax.grid(axis="x", color=GRID)
    ax.set_xlabel("drop in test ROC AUC when the feature is shuffled (5 repeats)")
    ax.set_title("Permutation importance on the test set (region has no effect in the generator)")
    return _save(fig, out, "feature_importance.png")


def drift(out: Path, params: dict) -> Path:
    d = _j("drift")
    f = pd.DataFrame(d["features"]).sort_values("psi")
    p = params["monitor"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    colors = [BAD if s == "alert" else SIGNAL if s == "warn" else GRID for s in f["status"]]
    a1.barh(f["feature"], f["psi"], color=colors, height=0.6)
    for i, v in enumerate(f["psi"]):
        a1.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8.5)
    a1.axvline(p["psi_warn"], color=SIGNAL, ls="--", lw=1)
    a1.axvline(p["psi_retrain"], color=BAD, ls="--", lw=1)
    a1.set_xlabel("PSI vs training data")
    a1.grid(axis="x", color=GRID)
    a1.set_title("Data drift in the next batch")
    bins = np.linspace(0, 1, 21)
    ref = np.array(d["prediction_hist"]["reference"], float)
    cur = np.array(d["prediction_hist"]["batch"], float)
    a2.bar(bins[:-1], ref / ref.sum(), width=0.05, align="edge", color=ACCENT, alpha=0.55, label="training data")
    a2.bar(bins[:-1], cur / cur.sum(), width=0.05, align="edge", color=SIGNAL, alpha=0.55, label="new batch")
    a2.set_xlabel("predicted churn probability")
    a2.set_ylabel("share of customers")
    a2.legend(frameon=False, fontsize=8.5)
    a2.grid(axis="y", color=GRID)
    a2.set_title(f"Predictions · PSI {d['prediction_psi']:.3f}")
    verdict = "RETRAIN" if d["retrain"] else "no retrain"
    a2.text(0.98, 0.95, f"{verdict}\n" + "\n".join(d["reasons"]) + f"\nbatch ROC AUC {d['roc_auc_batch']:.3f} "
            f"(test {d['roc_auc_test']:.3f})", transform=a2.transAxes, ha="right", va="top", fontsize=8.5,
            color=BAD if d["retrain"] else GOOD)
    fig.suptitle("Monitoring a later synthetic batch: price rise + outage (data drift) and a rival offer (concept "
                 "drift)", x=0.01, ha="left", fontsize=13, weight="bold", color=INK)
    return _save(fig, out, "drift.png")


def render_all(out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    params = config.load_params()
    return [architecture(out), model_comparison(out), roc_pr_calibration(out), threshold_cost(out, params),
            importance(out), drift(out, params)]


__all__ = ["render_all", "TRUE_EFFECTS"]
