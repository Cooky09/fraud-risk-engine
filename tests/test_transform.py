import pandas as pd

from src.etl.extract import extract_transactions
from src.etl.transform import transform_transactions


def test_transform_creates_fraud_features():
    raw_df = extract_transactions()
    transformed_df = transform_transactions(raw_df)

    expected_features = {
        "transaction_hour",
        "transaction_day_of_week",
        "is_weekend",
        "is_night",
        "country_mismatch",
        "high_velocity_24h",
        "high_velocity_7d",
        "account_age_months",
        "new_account",
        "amount_log",
        "amount_change_ratio",
    }

    assert expected_features.issubset(
        transformed_df.columns
    )


def test_transform_preserves_transaction_count():
    raw_df = extract_transactions()
    transformed_df = transform_transactions(raw_df)

    assert len(transformed_df) == len(raw_df)


def test_fraud_target_is_binary():
    raw_df = extract_transactions()
    transformed_df = transform_transactions(raw_df)

    assert set(
        transformed_df["is_fraud"].unique()
    ).issubset({0, 1})


def test_transaction_amounts_are_valid():
    raw_df = extract_transactions()
    transformed_df = transform_transactions(raw_df)

    assert (
        transformed_df["amount"] >= 0
    ).all()


def test_timestamp_is_datetime():
    raw_df = extract_transactions()
    transformed_df = transform_transactions(raw_df)

    assert pd.api.types.is_datetime64_any_dtype(
        transformed_df["transaction_timestamp"]
    )