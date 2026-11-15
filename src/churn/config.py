"""Paths and params shared by every stage."""
from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"
PLOTS = ROOT / "plots"
REPORTS = ROOT / "reports"
IMAGES = ROOT / "docs" / "images"


def load_params(path: str | Path | None = None) -> dict:
    return yaml.safe_load(Path(path or ROOT / "params.yaml").read_text())
