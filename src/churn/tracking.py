"""Experiment tracking behind one small interface.

* ``MLflowTracker``: params, metrics, artifacts, the sklearn model, the Model Registry and an alias
  (``champion``). Used whenever MLflow is installed (CI, Docker, your laptop after ``pip install mlflow``).
* ``LocalTracker``: the same calls written to plain JSON files under ``mlruns_local/``. Used only
  when MLflow is not importable, and it says so in the log. It exists so the pipeline still runs in
  minimal environments; it is not a replacement for an MLflow server.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger("churn.tracking")


class LocalTracker:
    kind = "local"

    def __init__(self, root: str | Path, experiment: str):
        self.root = Path(root)
        self.experiment = experiment
        self.run_dir: Path | None = None
        self.run_id: str | None = None

    @contextmanager
    def start_run(self, run_name: str):
        self.run_id = uuid.uuid4().hex[:12]
        self.run_dir = self.root / self.experiment / f"{run_name}-{self.run_id}"
        (self.run_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "meta.json").write_text(json.dumps({"run_id": self.run_id, "run_name": run_name,
                                                            "start_time": time.time()}))
        try:
            yield self
        finally:
            self.run_dir = None

    def _merge(self, name: str, values: dict) -> None:
        p = self.run_dir / name
        cur = json.loads(p.read_text()) if p.exists() else {}
        cur.update(values)
        p.write_text(json.dumps(cur, indent=1, default=str))

    def log_params(self, params: dict) -> None:
        self._merge("params.json", {k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        self._merge("metrics.json", {k: float(v) for k, v in metrics.items()})

    def log_artifact(self, path: str | Path) -> None:
        shutil.copy(path, self.run_dir / "artifacts" / Path(path).name)

    def log_model(self, model_path: str | Path, registered_name: str, alias: str, **_) -> str:
        """Copy the model into a numbered registry version and point the alias at it."""
        self.log_artifact(model_path)
        reg = self.root / "registry" / registered_name
        versions = sorted(int(p.name) for p in reg.glob("[0-9]*")) if reg.exists() else []
        version = str((versions[-1] if versions else 0) + 1)
        (reg / version).mkdir(parents=True)
        shutil.copy(model_path, reg / version / Path(model_path).name)
        (reg / version / "meta.json").write_text(json.dumps({"run_id": self.run_id, "version": version}))
        aliases_p = reg / "aliases.json"
        aliases = json.loads(aliases_p.read_text()) if aliases_p.exists() else {}
        aliases[alias] = version
        aliases_p.write_text(json.dumps(aliases))
        return version


class MLflowTracker:
    kind = "mlflow"

    def __init__(self, tracking_uri: str, experiment: str):
        import mlflow

        self.mlflow = mlflow
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        self.run_id: str | None = None

    @contextmanager
    def start_run(self, run_name: str):
        with self.mlflow.start_run(run_name=run_name) as run:
            self.run_id = run.info.run_id
            yield self

    def log_params(self, params: dict) -> None:
        self.mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict) -> None:
        self.mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_artifact(self, path: str | Path) -> None:
        self.mlflow.log_artifact(str(path))

    def log_model(self, model_path: str | Path, registered_name: str, alias: str, model=None, input_example=None
                  ) -> str:
        from mlflow.tracking import MlflowClient

        self.mlflow.sklearn.log_model(model, artifact_path="model", registered_model_name=registered_name,
                                      input_example=input_example)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{registered_name}' and run_id='{self.run_id}'")
        version = max(versions, key=lambda v: int(v.version)).version
        client.set_registered_model_alias(registered_name, alias, version)
        return str(version)


def make_tracker(root: str | Path, experiment: str = "churn") -> LocalTracker | MLflowTracker:
    """``CHURN_TRACKER=mlflow|local|auto`` (default auto: MLflow if importable)."""
    mode = os.environ.get("CHURN_TRACKER", "auto")
    if mode in ("auto", "mlflow"):
        try:
            import mlflow  # noqa: F401
        except ImportError:
            if mode == "mlflow":
                raise
            log.warning("MLflow is not installed: falling back to the local JSON tracker in %s/", root)
        else:
            uri = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{Path(root).parent / 'mlflow.db'}")
            log.info("tracking with MLflow at %s", uri)
            return MLflowTracker(uri, experiment)
    return LocalTracker(Path(root) / "mlruns_local", experiment)
