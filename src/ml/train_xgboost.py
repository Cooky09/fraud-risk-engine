from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.database.connection import engine
from src.ml.preprocessing import (
    prepare_features,
    build_preprocessor,
)
from src.ml.split import split_dataset


MODEL_PATH = Path("models/xgboost_fraud_model.joblib")
PREPROCESSOR_PATH = Path("models/xgboost_preprocessor.joblib")


def main() -> None:
    print("Loading dataset from MySQL...")

    df = pd.read_sql(
        "SELECT * FROM transactions",
        engine,
    )

    print(f"Loaded {len(df):,} transactions")

    X, y = prepare_features(df)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_dataset(X, y)

    print(f"Training rows:   {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"Test rows:       {len(X_test):,}")

    print(f"Training fraud:      {y_train.sum():,}")
    print(f"Training legitimate: {(y_train == 0).sum():,}")

    scale_pos_weight = (
        (y_train == 0).sum() / y_train.sum()
    )

    print(
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    print("\nBuilding preprocessor...")

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_validation_processed = preprocessor.transform(
        X_validation
    )

    X_test_processed = preprocessor.transform(
        X_test
    )

    print(
        "Processed feature count:",
        X_train_processed.shape[1],
    )

    print("\nBuilding XGBoost model...")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print("Training XGBoost...")

    model.fit(
        X_train_processed,
        y_train,
    )

    print("Training complete.")

    print("\nEvaluating validation set...")

    validation_probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    print(
        f"Validation PR-AUC: {validation_pr_auc:.4f}"
    )

    print("\nEvaluating test set at threshold 0.50...")

    test_probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    threshold = 0.50

    test_predictions = (
        test_probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        test_predictions,
        zero_division=0,
    )

    pr_auc = average_precision_score(
        y_test,
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    print("\n===== XGBoost Results =====")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            test_predictions,
        )
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            test_predictions,
            zero_division=0,
        )
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    print("\nSaving model...")
    print(f"Model saved to: {MODEL_PATH}")
    print(
        f"Preprocessor saved to: "
        f"{PREPROCESSOR_PATH}"
    )


if __name__ == "__main__":
    main()