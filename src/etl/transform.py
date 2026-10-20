import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
    "amount",
    "merchant_category",
    "merchant_country",
    "customer_country",
    "account_age_days",
    "transaction_count_24h",
    "transaction_count_7d",
    "previous_transaction_amount",
    "is_fraud",
}


def transform_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transactions and create fraud detection features."""

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    transformed = df.copy()

    # Convert timestamp to datetime.
    transformed["transaction_timestamp"] = pd.to_datetime(
        transformed["transaction_timestamp"],
        errors="coerce",
    )

    if transformed["transaction_timestamp"].isna().any():
        raise ValueError(
            "Invalid transaction timestamps found."
        )

    # Basic validation.
    if (transformed["amount"] < 0).any():
        raise ValueError(
            "Transaction amounts cannot be negative."
        )

    # Time-based features.
    transformed["transaction_hour"] = (
        transformed["transaction_timestamp"].dt.hour
    )

    transformed["transaction_day_of_week"] = (
        transformed["transaction_timestamp"].dt.dayofweek
    )

    transformed["is_weekend"] = (
        transformed["transaction_day_of_week"] >= 5
    ).astype(int)

    transformed["is_night"] = (
        (transformed["transaction_hour"] < 6)
        | (transformed["transaction_hour"] >= 22)
    ).astype(int)

    # Geographic feature.
    transformed["country_mismatch"] = (
        transformed["merchant_country"]
        != transformed["customer_country"]
    ).astype(int)

    # Transaction velocity features.
    transformed["high_velocity_24h"] = (
        transformed["transaction_count_24h"] >= 8
    ).astype(int)

    transformed["high_velocity_7d"] = (
        transformed["transaction_count_7d"] >= 40
    ).astype(int)

    # Account age.
    transformed["account_age_months"] = (
        transformed["account_age_days"] / 30.0
    ).round(2)

    transformed["new_account"] = (
        transformed["account_age_days"] < 90
    ).astype(int)

    # Amount-based features.
    transformed["amount_log"] = np.log1p(
        transformed["amount"]
    )

    transformed["amount_change_ratio"] = (
        transformed["amount"]
        / transformed["previous_transaction_amount"].replace(
            0,
            np.nan,
        )
    )

    transformed["amount_change_ratio"] = (
        transformed["amount_change_ratio"]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(1.0)
    )

    # Ensure fraud target is integer.
    transformed["is_fraud"] = (
        transformed["is_fraud"].astype(int)
    )

    return transformed