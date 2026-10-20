from src.database.connection import engine
from src.ml.dataset import (
    load_dataset,
    prepare_features,
    split_dataset,
)


def test_dataset_loads():
    df = load_dataset(engine)

    assert len(df) == 50_000
    assert "is_fraud" in df.columns


def test_feature_preparation():
    df = load_dataset(engine)

    X, y = prepare_features(df)

    assert "is_fraud" not in X.columns
    assert "transaction_id" not in X.columns
    assert "customer_id" not in X.columns

    assert len(X) == len(y)


def test_stratified_split():
    df = load_dataset(engine)

    X, y = prepare_features(df)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = split_dataset(X, y)

    assert len(X_train) == 35_000
    assert len(X_val) == 7_500
    assert len(X_test) == 7_500

    assert y_train.sum() > 0
    assert y_val.sum() > 0
    assert y_test.sum() > 0