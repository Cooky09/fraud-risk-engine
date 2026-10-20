from __future__ import annotations

import pandas as pd
from sqlalchemy.engine import Engine
from sklearn.model_selection import train_test_split


TARGET = "is_fraud"

DROP_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_timestamp",
]


def load_dataset(engine: Engine) -> pd.DataFrame:
    """Load the ML dataset from MySQL."""

    query = """
        SELECT *
        FROM transactions
    """

    return pd.read_sql(query, engine)


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features from the target."""

    data = df.copy()

    X = data.drop(columns=[TARGET, *DROP_COLUMNS])
    y = data[TARGET]

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Create stratified train, validation, and test datasets."""

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42,
    )

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    )