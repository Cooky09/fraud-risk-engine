import pandas as pd
import joblib

from src.etl.transform import transform_transactions
from src.ml.preprocessing import prepare_features


MODEL_PATH = "models/lightgbm_fraud_model.joblib"
PREPROCESSOR_PATH = "models/lightgbm_preprocessor.joblib"

THRESHOLD = 0.75


model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)


def predict_transaction(transaction: dict) -> dict:
    """Predict fraud probability for a single transaction."""

    # Convert single transaction to DataFrame.
    df = pd.DataFrame([transaction])

    # transform_transactions currently requires the target column.
    # For prediction, use a dummy value because the real target is unknown.
    df["is_fraud"] = 0

    # Apply the exact same feature engineering used during training.
    transformed = transform_transactions(df)

    # Select model features.
    X, _ = prepare_features(transformed)

    # Apply saved preprocessing.
    X_processed = preprocessor.transform(X)

    # Get fraud probability.
    fraud_probability = model.predict_proba(X_processed)[:, 1][0]

    # Apply production threshold.
    is_fraud = fraud_probability >= THRESHOLD

    return {
        "fraud_probability": round(float(fraud_probability), 4),
        "threshold": THRESHOLD,
        "is_fraud": bool(is_fraud),
        "decision": "FRAUD" if is_fraud else "LEGITIMATE",
    }