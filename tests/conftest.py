import json
import shutil

import pytest

from churn import config, stages


def point_config_at(mp, root):
    for name, sub in (("ROOT", ""), ("DATA", "data"), ("MODELS", "models"), ("METRICS", "metrics"),
                      ("PLOTS", "plots"), ("REPORTS", "reports"), ("IMAGES", "docs/images")):
        mp.setattr(config, name, root / sub if sub else root)


@pytest.fixture(scope="session")
def params():
    p = config.load_params()
    p = json.loads(json.dumps(p))
    p["generate"]["n_customers"] = 4000          # smaller for speed; same code path
    p["monitor"]["n_customers"] = 2000
    p["train"]["candidates"]["random_forest"]["n_estimators"] = 60
    p["train"]["candidates"]["hist_gradient_boosting"]["max_iter"] = 80
    return p


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory, params):
    """Run every DVC stage once into a temporary project root (local tracker)."""
    root = tmp_path_factory.mktemp("project")
    shutil.copy(config.ROOT / "params.yaml", root / "params.yaml")
    with pytest.MonkeyPatch.context() as mp:
        point_config_at(mp, root)
        mp.setenv("CHURN_TRACKER", "local")
        for s in ("generate", "prepare", "train", "evaluate", "monitor"):
            getattr(stages, f"stage_{s}")(params)
        yield root
