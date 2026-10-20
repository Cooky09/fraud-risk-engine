from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
TRANSACTION_COUNT = 50_000

rng = np.random.default_rng(SEED)

output_path = Path("data/raw/transactions.csv")


def generate_transactions() -> pd.DataFrame:
    transaction_ids = np.arange(1, TRANSACTION_COUNT + 1)

    customer_ids = rng.integers(
        low=10_000,
        high=20_000,
        size=TRANSACTION_COUNT,
    )

    timestamps = pd.date_range(
        start="2025-01-01",
        periods=TRANSACTION_COUNT,
        freq="15min",
    )

    merchant_categories = rng.choice(
        [
            "grocery",
            "electronics",
            "travel",
            "restaurant",
            "online",
            "subscription",
        ],
        size=TRANSACTION_COUNT,
    )

    countries = rng.choice(
        ["US", "GB", "CA", "DE", "FR", "AU"],
        size=TRANSACTION_COUNT,
    )

    customer_countries = countries.copy()

    account_age_days = rng.integers(
        low=30,
        high=3650,
        size=TRANSACTION_COUNT,
    )

    transaction_count_24h = rng.poisson(
        lam=3,
        size=TRANSACTION_COUNT,
    )

    transaction_count_7d = (
        transaction_count_24h
        + rng.poisson(
            lam=12,
            size=TRANSACTION_COUNT,
        )
    )

    amounts = np.round(
        rng.lognormal(
            mean=3.2,
            sigma=1.0,
            size=TRANSACTION_COUNT,
        ),
        2,
    )

    previous_amount = np.round(
        amounts
        * rng.uniform(
            0.5,
            1.5,
            size=TRANSACTION_COUNT,
        ),
        2,
    )

    # ---------------------------------------------------------
    # Generate latent suspicious behavior
    # ---------------------------------------------------------

    suspicious = rng.random(TRANSACTION_COUNT) < 0.006

    suspicious_count = suspicious.sum()

    print(
        f"Generating {suspicious_count:,} suspicious transactions"
    )

    # Unusually large transactions
    large_amount = (
        rng.random(TRANSACTION_COUNT) < 0.65
    ) & suspicious

    amounts[large_amount] *= rng.uniform(
        2.5,
        6.0,
        size=large_amount.sum(),
    )

    amounts = np.round(amounts, 2)

    # High transaction velocity
    high_velocity = (
        rng.random(TRANSACTION_COUNT) < 0.60
    ) & suspicious

    transaction_count_24h[high_velocity] += rng.integers(
        8,
        20,
        size=high_velocity.sum(),
    )

    transaction_count_7d[high_velocity] += rng.integers(
        20,
        60,
        size=high_velocity.sum(),
    )

    # New accounts
    new_account_behavior = (
        rng.random(TRANSACTION_COUNT) < 0.35
    ) & suspicious

    account_age_days[new_account_behavior] = rng.integers(
        1,
        90,
        size=new_account_behavior.sum(),
    )

    # International mismatch
    country_mismatch_behavior = (
        rng.random(TRANSACTION_COUNT) < 0.55
    ) & suspicious

    mismatch_indices = np.where(
        country_mismatch_behavior
    )[0]

    for index in mismatch_indices:
        available = [
            country
            for country in ["US", "GB", "CA", "DE", "FR", "AU"]
            if country != countries[index]
        ]

        customer_countries[index] = rng.choice(
            available
        )

    # Unusual hours
    unusual_hour_behavior = (
        rng.random(TRANSACTION_COUNT) < 0.55
    ) & suspicious

    unusual_indices = np.where(
        unusual_hour_behavior
    )[0]

    timestamp_values = timestamps.to_numpy().copy()

    for index in unusual_indices:
        timestamp = pd.Timestamp(timestamp_values[index])

        timestamp_values[index] = timestamp.replace(
            hour=int(
                rng.choice(
                    [0, 1, 2, 3, 4, 5]
                )
            ),
            minute=0,
            second=0,
        )

    timestamps = pd.DatetimeIndex(timestamp_values)
    # Large change from previous transaction
    amount_change_behavior = (
        rng.random(TRANSACTION_COUNT) < 0.50
    ) & suspicious

    previous_amount[amount_change_behavior] = np.round(
        amounts[amount_change_behavior]
        * rng.uniform(
            0.05,
            0.25,
            size=amount_change_behavior.sum(),
        ),
        2,
    )

    # ---------------------------------------------------------
    # Build fraud probability from multiple weak signals
    # ---------------------------------------------------------

    hour = timestamps.hour

    country_mismatch = (
        customer_countries != countries
    )

    amount_change_ratio = (
        amounts
        / np.maximum(previous_amount, 0.01)
    )

    suspicious_signals = (
        (amounts > 150).astype(int)
        + (transaction_count_24h > 8).astype(int)
        + (transaction_count_7d > 35).astype(int)
        + (account_age_days < 90).astype(int)
        + country_mismatch.astype(int)
        + ((hour < 6) | (hour >= 23)).astype(int)
        + (amount_change_ratio > 2.5).astype(int)
    )

    # Fraud probability depends on combinations of signals.
    fraud_probability = np.where(
        suspicious_signals >= 4,
        0.75,
        np.where(
            suspicious_signals == 3,
            0.35,
            np.where(
                suspicious_signals == 2,
                0.08,
                0.001,
            ),
        ),
    )

    is_fraud = (
        rng.random(TRANSACTION_COUNT)
        < fraud_probability
    )

    return pd.DataFrame(
        {
            "transaction_id": transaction_ids,
            "customer_id": customer_ids,
            "transaction_timestamp": timestamps,
            "amount": amounts,
            "merchant_category": merchant_categories,
            "merchant_country": countries,
            "customer_country": customer_countries,
            "account_age_days": account_age_days,
            "transaction_count_24h": transaction_count_24h,
            "transaction_count_7d": transaction_count_7d,
            "previous_transaction_amount": previous_amount,
            "is_fraud": is_fraud,
        }
    )


if __name__ == "__main__":
    df = generate_transactions()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print(f"Created {len(df):,} transactions")
    print(f"Saved to: {output_path}")
    print()

    print("Fraud distribution:")
    print(df["is_fraud"].value_counts())

    print()

    print("Fraud rate:")
    print(f"{df['is_fraud'].mean():.2%}")

    print()

    print("Fraud feature comparison:")

    print(
        df.groupby("is_fraud")[
            [
                "amount",
                "transaction_count_24h",
                "transaction_count_7d",
                "account_age_days",
            ]
        ].mean()
    )