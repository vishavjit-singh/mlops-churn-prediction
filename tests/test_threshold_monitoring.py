import numpy as np
import pandas as pd

from churn.monitoring import psi_categorical, psi_numeric
from churn.stages import best_threshold


def test_best_threshold_minimises_cost():
    y = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.6, 0.3, 0.35, 0.2, 0.1, 0.05, 0.02])
    t, curve = best_threshold(y, p, cost_fn=100, cost_fp=1)
    assert t <= 0.3                                  # catching the 0.3 churner is worth one false positive
    t2, _ = best_threshold(y, p, cost_fn=1, cost_fp=100)
    assert t2 > 0.35                                 # false positives are expensive: be strict
    assert curve["cost_per_customer"].min() == curve.set_index("threshold").loc[t, "cost_per_customer"]


def test_precision_is_undefined_when_nothing_is_flagged():
    _, curve = best_threshold(np.array([1, 0]), np.array([0.1, 0.2]), 1, 1)
    assert np.isnan(curve.set_index("threshold").loc[0.98, "precision"])


def test_psi_numeric():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 5000))
    assert psi_numeric(ref, pd.Series(rng.normal(0, 1, 5000))) < 0.02
    assert psi_numeric(ref, pd.Series(rng.normal(1, 1, 5000))) > 0.25
    assert psi_numeric(pd.Series([0] * 100), pd.Series([0] * 50 + [1] * 50)) > 0.25   # constant reference


def test_psi_categorical_handles_new_categories():
    ref = pd.Series(["a"] * 50 + ["b"] * 50)
    assert psi_categorical(ref, ref.copy()) < 1e-6
    assert psi_categorical(ref, pd.Series(["a"] * 50 + ["c"] * 50)) > 1
