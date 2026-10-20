import pandas as pd
from joblib import load
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.database.connection import engine
from src.ml.preprocessing import prepare_features
from src.ml.split import split_dataset


MODEL_PATH = "models/lightgbm_fraud_model.joblib"
PREPROCESSOR_PATH = "models/lightgbm_preprocessor.joblib"

THRESHOLD = 0.75


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

    X_test_processed = preprocessor.transform(
        X_test
    )

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print()
    print("===== LightGBM Final Test Results =====")
    print(f"Threshold: {THRESHOLD:.2f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print()
    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("Classification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    print()
    print(
        f"Transactions flagged as fraud: "
        f"{predictions.sum()}"
    )


if __name__ == "__main__":
    main()