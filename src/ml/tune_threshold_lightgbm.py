import pandas as pd
from joblib import load
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)

from src.database.connection import engine
from src.ml.preprocessing import prepare_features
from src.ml.split import split_dataset


MODEL_PATH = "models/lightgbm_fraud_model.joblib"
PREPROCESSOR_PATH = "models/lightgbm_preprocessor.joblib"


def main() -> None:
    print("Loading data...")

    df = pd.read_sql(
        "SELECT * FROM transactions",
        engine,
    )

    X, y = prepare_features(df)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(X, y)

    model = load(MODEL_PATH)
    preprocessor = load(PREPROCESSOR_PATH)

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    thresholds = [
        0.05,
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    print()
    print("LightGBM threshold analysis")
    print("=" * 75)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'Flagged':<12}"
    )

    best_threshold = None
    best_f1 = -1

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_validation,
            predictions,
            zero_division=0,
        )

        flagged = predictions.sum()

        print(
            f"{threshold:<12.2f}"
            f"{precision:<12.3f}"
            f"{recall:<12.3f}"
            f"{f1:<12.3f}"
            f"{flagged:<12}"
        )

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print()
    print("Best threshold by F1:")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"F1:        {best_f1:.3f}")


if __name__ == "__main__":
    main()