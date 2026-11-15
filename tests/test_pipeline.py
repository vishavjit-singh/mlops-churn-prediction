import json

import joblib
import pandas as pd

from churn.schema import FEATURES


def read(root, path):
    return json.loads((root / path).read_text())


def test_every_declared_output_exists(pipeline):
    for path in ("data/raw/customers.csv", "data/prepared/train.csv", "data/prepared/test.csv", "models/model.joblib",
                 "models/metadata.json", "models/evaluation.json", "metrics/data.json", "metrics/train.json",
                 "metrics/eval.json", "metrics/drift.json", "plots/roc.csv", "plots/pr.csv", "plots/calibration.csv",
                 "plots/threshold_cost.csv", "plots/importance.csv", "plots/drift.csv", "reports/drift_report.json",
                 "reports/data_report.json", "data/monitor/batch.csv"):
        assert (pipeline / path).exists(), path


def test_dvc_metrics_files_are_flat_numbers(pipeline):
    for name in ("data", "eval", "drift"):
        for k, v in read(pipeline, f"metrics/{name}.json").items():
            assert isinstance(v, (int, float)), (name, k, v)


def test_split_is_stratified_and_disjoint(pipeline):
    train = pd.read_csv(pipeline / "data/prepared/train.csv")
    test = pd.read_csv(pipeline / "data/prepared/test.csv")
    assert not set(train["customer_id"]) & set(test["customer_id"])
    assert abs(train["churned"].mean() - test["churned"].mean()) < 0.01


def test_champion_is_the_best_cv_candidate(pipeline):
    t = read(pipeline, "metrics/train.json")
    assert t["best"] == max(t["candidates"], key=lambda n: t["candidates"][n]["roc_auc"])
    meta = read(pipeline, "models/metadata.json")
    assert meta["algorithm"] == t["best"] and meta["features"] == FEATURES
    assert meta["tracker"] == "local" and meta["version"] == "1" and meta["alias"] == "champion"


def test_model_quality_and_threshold(pipeline):
    e = read(pipeline, "metrics/eval.json")
    assert e["roc_auc"] > 0.75 and e["pr_auc"] > 2 * e["base_rate"]
    assert e["threshold"] < 0.5                      # missed churners cost more than offers
    assert e["cost_per_customer_at_threshold"] <= e["cost_per_customer_at_0_5"] < e["cost_per_customer_no_model"]
    assert e["top_decile_lift"] > 2


def test_permutation_importance_finds_the_true_drivers(pipeline):
    imp = pd.read_csv(pipeline / "plots/importance.csv").set_index("feature")["importance"]
    assert imp.idxmax() in ("contract", "tenure_months")
    assert abs(imp["region"]) < 0.005                # no effect in the generator


def test_model_artifact_scores_raw_rows(pipeline):
    model = joblib.load(pipeline / "models/model.joblib")
    test = pd.read_csv(pipeline / "data/prepared/test.csv")
    p = model.predict_proba(test[FEATURES].head(5))[:, 1]
    assert ((p >= 0) & (p <= 1)).all()


def test_monitor_flags_the_drifted_batch(pipeline):
    d = read(pipeline, "reports/drift_report.json")
    psi = {f["feature"]: f["psi"] for f in d["features"]}
    assert psi["support_calls_90d"] > 0.25 and psi["region"] < 0.05
    assert d["retrain"] and d["reasons"]
