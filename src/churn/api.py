"""Prediction API (Starlette + pydantic).

FastAPI could not be installed in the environment this was built in, so the service uses Starlette,
the ASGI toolkit FastAPI is built on, with pydantic models for request validation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from . import config
from .schema import FEATURES

MAX_BATCH = 1000


class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: str = Field(min_length=1, max_length=40)
    tenure_months: int = Field(ge=0, le=72)
    contract: Literal["month-to-month", "one-year", "two-year"]
    internet_service: Literal["fiber", "dsl", "none"]
    payment_method: Literal["card_autopay", "bank_autopay", "upi", "cash"]
    monthly_charges: float = Field(ge=0, le=400)
    total_charges: float | None = Field(default=None, ge=0, le=30000)
    support_calls_90d: int = Field(ge=0, le=30)
    data_usage_gb: float | None = Field(default=None, ge=0, le=500)
    num_addons: int = Field(ge=0, le=6)
    late_payments_12m: int = Field(ge=0, le=12)
    senior: Literal[0, 1]
    has_partner: Literal[0, 1]
    region: Literal["north", "south", "east", "west"]


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[Customer] = Field(min_length=1, max_length=MAX_BATCH)


def risk_band(p: float, threshold: float) -> str:
    return "high" if p >= max(threshold, 0.5) else "medium" if p >= threshold else "low"


class ModelBundle:
    def __init__(self, model_dir: Path):
        self.model = joblib.load(model_dir / "model.joblib")
        self.meta = json.loads((model_dir / "metadata.json").read_text())
        ev = model_dir / "evaluation.json"
        if ev.exists():                        # written by the evaluate stage
            self.meta.update(json.loads(ev.read_text()))
        self.threshold = float(self.meta.get("threshold", 0.5))

    def predict(self, customers: list[Customer]) -> list[dict]:
        df = pd.DataFrame([c.model_dump() for c in customers])
        probs = self.model.predict_proba(df[FEATURES])[:, 1]
        return [{"customer_id": c.customer_id, "churn_probability": round(float(p), 4),
                 "will_churn": bool(p >= self.threshold), "risk": risk_band(float(p), self.threshold)}
                for c, p in zip(customers, probs)]


def _error(status: int, code: str, message: str, **extra) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, **extra}}, status_code=status)


async def health(request: Request):
    return JSONResponse({"status": "ok", "model_version": request.app.state.bundle.meta["version"]})


async def model_info(request: Request):
    m = request.app.state.bundle.meta
    keys = ("model_name", "version", "alias", "algorithm", "threshold", "trained_at", "train_rows", "train_data_sha",
            "tracker", "test_metrics", "features")
    return JSONResponse({k: m.get(k) for k in keys})


async def predict(request: Request):
    try:
        body = await request.json()
    except ValueError:
        return _error(400, "malformed_json", "request body is not valid JSON")
    try:
        customers = Batch.model_validate(body).customers if isinstance(body, dict) and "customers" in body \
            else [Customer.model_validate(body)]
    except ValidationError as exc:
        errors = [{"loc": ".".join(str(x) for x in e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _error(422, "validation_error", "request validation failed", errors=errors)
    preds = request.app.state.bundle.predict(customers)
    bundle = request.app.state.bundle
    out = {"model_version": bundle.meta["version"], "threshold": bundle.threshold}
    return JSONResponse({**out, "predictions": preds} if "customers" in body else {**out, **preds[0]})


def create_app(model_dir: str | Path | None = None) -> Starlette:
    app = Starlette(routes=[Route("/health", health), Route("/model", model_info),
                            Route("/predict", predict, methods=["POST"])])
    app.state.bundle = ModelBundle(Path(model_dir or os.environ.get("MODEL_DIR", config.MODELS)))
    return app


def __getattr__(name):            # `uvicorn churn.api:app` builds the app lazily (model must exist)
    if name == "app":
        return create_app()
    raise AttributeError(name)
