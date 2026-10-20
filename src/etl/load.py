import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine


TABLE_NAME = "transactions"


def create_transactions_table(engine: Engine) -> None:
    """Create the transactions table if it does not already exist."""

    query = text(
        """
        CREATE TABLE IF NOT EXISTS transactions (
            transaction_id BIGINT PRIMARY KEY,
            customer_id BIGINT NOT NULL,
            transaction_timestamp DATETIME NOT NULL,
            amount DECIMAL(12, 2) NOT NULL,
            merchant_category VARCHAR(50) NOT NULL,
            merchant_country VARCHAR(10) NOT NULL,
            customer_country VARCHAR(10) NOT NULL,
            account_age_days INT NOT NULL,
            transaction_count_24h INT NOT NULL,
            transaction_count_7d INT NOT NULL,
            previous_transaction_amount DECIMAL(12, 2) NOT NULL,
            is_fraud TINYINT NOT NULL,

            transaction_hour TINYINT NOT NULL,
            transaction_day_of_week TINYINT NOT NULL,
            is_weekend TINYINT NOT NULL,
            is_night TINYINT NOT NULL,
            country_mismatch TINYINT NOT NULL,
            high_velocity_24h TINYINT NOT NULL,
            high_velocity_7d TINYINT NOT NULL,
            account_age_months DECIMAL(10, 2) NOT NULL,
            new_account TINYINT NOT NULL,
            amount_log DOUBLE NOT NULL,
            amount_change_ratio DOUBLE NOT NULL,

            INDEX idx_customer_id (customer_id),
            INDEX idx_transaction_timestamp (transaction_timestamp),
            INDEX idx_is_fraud (is_fraud)
        )
        """
    )

    with engine.begin() as connection:
        connection.execute(query)


def load_transactions(
    df: pd.DataFrame,
    engine: Engine,
) -> int:
    """Load transformed transactions into MySQL in small batches."""

    create_transactions_table(engine)

    total_loaded = 0
    chunk_size = 100

    for start in range(0, len(df), chunk_size):
        chunk = df.iloc[start:start + chunk_size]

        chunk.to_sql(
            TABLE_NAME,
            con=engine,
            if_exists="append",
            index=False,
            method=None,
        )

        total_loaded += len(chunk)

        print(
            f"Loaded {total_loaded:,}/{len(df):,} transactions",
            flush=True,
        )

    return total_loaded