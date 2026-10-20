import pandas as pd

from src.ml.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


def test_preprocessor():
    df = pd.DataFrame(
        {
            "merchant_category": ["travel", "food"],
            "merchant_country": ["US", "CA"],
            "customer_country": ["US", "GB"],
            **{
                feature: [1, 2]
                for feature in NUMERICAL_FEATURES
            },
        }
    )

    preprocessor = build_preprocessor()

    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 2
    assert transformed.shape[1] > len(NUMERICAL_FEATURES)


def test_feature_lists():
    assert len(CATEGORICAL_FEATURES) == 3
    assert len(NUMERICAL_FEATURES) == 16