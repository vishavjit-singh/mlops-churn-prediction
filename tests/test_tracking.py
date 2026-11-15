import json
import sys
import types
from contextlib import contextmanager

import pytest

from churn import tracking
from churn.tracking import LocalTracker, MLflowTracker, make_tracker


def test_local_tracker_registry_versions_and_alias(tmp_path):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"m1")
    t = LocalTracker(tmp_path / "mlruns_local", "exp")
    with t.start_run("a"):
        t.log_params({"C": 1})
        t.log_metrics({"auc": 0.8})
        v1 = t.log_model(model, "churn-classifier", "champion")
    with t.start_run("b"):
        v2 = t.log_model(model, "churn-classifier", "champion")
    reg = tmp_path / "mlruns_local" / "registry" / "churn-classifier"
    assert (v1, v2) == ("1", "2")
    assert json.loads((reg / "aliases.json").read_text()) == {"champion": "2"}
    run = next((tmp_path / "mlruns_local" / "exp").glob("a-*"))
    assert json.loads((run / "metrics.json").read_text()) == {"auc": 0.8}


class FakeMlflow(types.ModuleType):
    """Records the calls MLflowTracker makes (the real library runs in the CI mlops job)."""

    def __init__(self):
        super().__init__("mlflow")
        self.calls = []
        fake = self

        class Version:
            def __init__(self, v):
                self.version = v

        class Client:
            def search_model_versions(self, query):
                fake.calls.append(("search", query))
                return [Version("3"), Version("4")]

            def set_registered_model_alias(self, name, alias, version):
                fake.calls.append(("alias", name, alias, version))

        self.sklearn = types.SimpleNamespace(log_model=lambda model, **kw: self.calls.append(("log_model", kw)))
        self.tracking = types.SimpleNamespace(MlflowClient=Client)

    def set_tracking_uri(self, uri):
        self.calls.append(("uri", uri))

    def set_experiment(self, name):
        self.calls.append(("experiment", name))

    @contextmanager
    def start_run(self, run_name):
        self.calls.append(("run", run_name))
        yield types.SimpleNamespace(info=types.SimpleNamespace(run_id="r123"))

    def log_params(self, p):
        self.calls.append(("params", p))

    def log_metrics(self, m):
        self.calls.append(("metrics", m))


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = FakeMlflow()
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", fake.tracking)
    return fake


def test_mlflow_tracker_registers_and_sets_alias(fake_mlflow, tmp_path):
    t = MLflowTracker("sqlite:///x.db", "churn")
    with t.start_run("champion"):
        t.log_params({"C": 1.0})
        version = t.log_model(tmp_path / "m", "churn-classifier", "champion", model=object(), input_example=None)
    assert version == "4"
    assert ("search", "name='churn-classifier' and run_id='r123'") in fake_mlflow.calls
    assert ("alias", "churn-classifier", "champion", "4") in fake_mlflow.calls
    kw = next(c[1] for c in fake_mlflow.calls if c[0] == "log_model")
    assert kw["registered_model_name"] == "churn-classifier" and kw["artifact_path"] == "model"


def test_auto_mode_uses_mlflow_when_importable(fake_mlflow, tmp_path, monkeypatch):
    monkeypatch.delenv("CHURN_TRACKER", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    t = make_tracker(tmp_path)
    assert t.kind == "mlflow"
    assert ("uri", f"sqlite:///{tmp_path.parent / 'mlflow.db'}") in fake_mlflow.calls


def test_auto_mode_falls_back_with_a_warning(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("CHURN_TRACKER", raising=False)
    monkeypatch.setitem(sys.modules, "mlflow", None)          # import fails
    t = make_tracker(tmp_path)
    assert t.kind == "local" and "falling back" in caplog.text


def test_forcing_mlflow_without_it_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CHURN_TRACKER", "mlflow")
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(ImportError):
        make_tracker(tmp_path)
    assert tracking.LocalTracker
