import pandas as pd
import pytest
from starlette.testclient import TestClient

from churn.api import MAX_BATCH, create_app

CUSTOMER = {"customer_id": "X1", "tenure_months": 3, "contract": "month-to-month", "internet_service": "fiber",
            "payment_method": "cash", "monthly_charges": 120.0, "total_charges": 360.0, "support_calls_90d": 4,
            "data_usage_gb": 80.0, "num_addons": 0, "late_payments_12m": 2, "senior": 0, "has_partner": 0,
            "region": "north"}
LOYAL = {**CUSTOMER, "customer_id": "X2", "tenure_months": 60, "contract": "two-year", "payment_method": "card_autopay",
         "support_calls_90d": 0, "late_payments_12m": 0, "num_addons": 4, "total_charges": 6000.0}


@pytest.fixture(scope="module")
def client(pipeline):
    return TestClient(create_app(pipeline / "models"))


def test_health_and_model(client):
    assert client.get("/health").json() == {"status": "ok", "model_version": "1"}
    m = client.get("/model").json()
    assert m["alias"] == "champion" and 0 < m["threshold"] < 0.5 and m["test_metrics"]["roc_auc"] > 0.75


def test_single_prediction(client):
    r = client.post("/predict", json=CUSTOMER).json()
    assert r["customer_id"] == "X1" and r["will_churn"] is True and r["risk"] in ("medium", "high")


def test_batch_prediction_orders_risk(client):
    r = client.post("/predict", json={"customers": [CUSTOMER, LOYAL]}).json()
    p = {x["customer_id"]: x["churn_probability"] for x in r["predictions"]}
    assert p["X1"] > 0.5 > 0.1 > p["X2"]
    assert r["model_version"] == "1"


def test_missing_optional_values_are_imputed(client):
    body = {**CUSTOMER, "total_charges": None, "data_usage_gb": None}
    assert client.post("/predict", json=body).status_code == 200


def test_matches_the_model_artifact(client, pipeline):
    import joblib

    from churn.schema import FEATURES

    model = joblib.load(pipeline / "models" / "model.joblib")
    expected = model.predict_proba(pd.DataFrame([CUSTOMER])[FEATURES])[:, 1][0]
    assert client.post("/predict", json=CUSTOMER).json()["churn_probability"] == round(float(expected), 4)


@pytest.mark.parametrize("body, field", [
    ({**CUSTOMER, "contract": "weekly"}, "contract"),
    ({**CUSTOMER, "tenure_months": -1}, "tenure_months"),
    ({**CUSTOMER, "favourite_colour": "blue"}, "favourite_colour"),
    ({k: v for k, v in CUSTOMER.items() if k != "region"}, "region"),
])
def test_validation_errors(client, body, field):
    r = client.post("/predict", json=body)
    assert r.status_code == 422
    assert any(e["loc"].endswith(field) for e in r.json()["error"]["errors"])


def test_batch_limits_and_bad_json(client):
    assert client.post("/predict", json={"customers": []}).status_code == 422
    big = {"customers": [{**CUSTOMER, "customer_id": f"C{i}"} for i in range(MAX_BATCH + 1)]}
    assert client.post("/predict", json=big).status_code == 422
    r = client.post("/predict", content=b"{nope", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "malformed_json"
