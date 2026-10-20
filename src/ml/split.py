from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Split data into:

    70% training
    15% validation
    15% test

    Stratification preserves the fraud ratio.
    """

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )