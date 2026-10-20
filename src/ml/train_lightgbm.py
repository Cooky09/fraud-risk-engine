import pandas as pd
import lightgbm as lgb
from joblib import dump
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from src.database.connection import engine
from src.ml.preprocessing import prepare_features
from src.ml.split import split_dataset


MODEL_PATH = "models/lightgbm_fraud_model.joblib"
PREPROCESSOR_PATH = "models/lightgbm_preprocessor.joblib"

THRESHOLD = 0.75


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
    print(f"Training fraud:  {y_train.sum():,}")
    print(f"Training legitimate: {(y_train == 0).sum():,}")

    scale_pos_weight = (
        (y_train == 0).sum() / y_train.sum()
    )

    print(
        f"scale_pos_weight: {scale_pos_weight:.2f}"
    )

    print()
    print("Building preprocessor...")

    from src.ml.preprocessing import build_preprocessor

    preprocessor = build_preprocessor()

    X_train_processed = preprocessor.fit_transform(
        X_train
    )

    X_validation_processed = (
        preprocessor.transform(X_validation)
    )

    X_test_processed = (
        preprocessor.transform(X_test)
    )

    print(
        "Processed feature count:",
        X_train_processed.shape[1],
    )

    print()
    print("Building LightGBM model...")

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1,
    )

    print("Training LightGBM...")

    model.fit(
        X_train_processed,
        y_train,
    )

    print("Training complete.")

    print()
    print("Evaluating validation set...")

    validation_probabilities = model.predict_proba(
        X_validation_processed
    )[:, 1]

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities,
    )

    print(
        f"Validation PR-AUC: "
        f"{validation_pr_auc:.4f}"
    )

    print()
    print(
        f"Evaluating test set at threshold "
        f"{THRESHOLD:.2f}..."
    )

    test_probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    predictions = (
        test_probabilities >= THRESHOLD
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
        test_probabilities,
    )

    roc_auc = roc_auc_score(
        y_test,
        test_probabilities,
    )

    print()
    print("===== LightGBM Results =====")
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
    print("Saving model...")

    dump(model, MODEL_PATH)
    dump(preprocessor, PREPROCESSOR_PATH)

    print(
        f"Model saved to: {MODEL_PATH}"
    )

    print(
        f"Preprocessor saved to: "
        f"{PREPROCESSOR_PATH}"
    )


if __name__ == "__main__":
    main()