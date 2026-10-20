from typing import List

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


TARGET_COLUMN = "is_fraud"

CATEGORICAL_FEATURES: List[str] = [
    "merchant_category",
    "merchant_country",
    "customer_country",
]

NUMERICAL_FEATURES: List[str] = [
    "amount",
    "account_age_days",
    "transaction_count_24h",
    "transaction_count_7d",
    "previous_transaction_amount",
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
]


def get_feature_columns() -> List[str]:
    """Return all model feature columns."""
    return CATEGORICAL_FEATURES + NUMERICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for ML models."""

    return ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_FEATURES,
            ),
            (
                "numerical",
                "passthrough",
                NUMERICAL_FEATURES,
            ),
        ]
    )


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from target."""

    feature_columns = get_feature_columns()

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].astype(int).copy()

    return X, y