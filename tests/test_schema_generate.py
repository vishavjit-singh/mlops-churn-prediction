import pandas as pd
import pytest

from churn.generate import generate
from churn.schema import FEATURES, SchemaError, validate


def test_generator_is_deterministic():
    pd.testing.assert_frame_equal(generate(500, 3), generate(500, 3))
    assert not generate(500, 3).equals(generate(500, 4))


def test_generated_data_fits_the_contract():
    df = generate(3000, 1)
    report = validate(df)
    assert 0.18 < report["churn_rate"] < 0.33
    assert report["out_of_range"] == {} and report["unseen_categories"] == {}
    assert set(report["nulls"]) <= {"data_usage_gb", "total_charges"}


def test_drifted_batch_shifts_in_the_planned_direction():
    base, later = generate(4000, 1), generate(4000, 2, drifted=True)
    assert later["support_calls_90d"].mean() > 1.8 * base["support_calls_90d"].mean()
    assert later["monthly_charges"].mean() > base["monthly_charges"].mean() * 1.08
    assert later["churned"].mean() > base["churned"].mean()


def test_contract_breaks_raise():
    df = generate(200, 1)
    with pytest.raises(SchemaError, match="missing columns"):
        validate(df.drop(columns=["contract"]))
    with pytest.raises(SchemaError, match="duplicate"):
        validate(pd.concat([df, df.head(2)]))
    bad = df.copy()
    bad.loc[0, "tenure_months"] = 500
    with pytest.raises(SchemaError, match="out of range"):
        validate(bad)
    bad = df.copy()
    bad.loc[0, "churned"] = 2
    with pytest.raises(SchemaError, match="0/1"):
        validate(bad)


def test_unseen_categories_are_reported_not_fatal():
    df = generate(200, 1)
    df.loc[0, "payment_method"] = "wallet"
    assert validate(df)["unseen_categories"] == {"payment_method": ["wallet"]}
    assert len(FEATURES) == 13
