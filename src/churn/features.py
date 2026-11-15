"""The sklearn preprocessing + model pipeline."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL, NUMERIC


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", numeric, list(NUMERIC)), ("cat", categorical, list(CATEGORICAL))])


def estimator(name: str, params: dict, seed: int):
    if name == "logistic_regression":
        return LogisticRegression(max_iter=2000, random_state=seed, **params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(random_state=seed, **params)
    raise ValueError(f"unknown candidate {name}")


def build(name: str, params: dict, seed: int) -> Pipeline:
    return Pipeline([("prep", preprocessor()), ("model", estimator(name, params, seed))])
