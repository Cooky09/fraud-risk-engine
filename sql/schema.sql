CREATE TABLE IF NOT EXISTS transactions (
    transaction_id BIGINT PRIMARY KEY,
    customer_id BIGINT NOT NULL,
    transaction_timestamp DATETIME NOT NULL,
    amount DECIMAL(12, 2) NOT NULL,
    merchant_category VARCHAR(100),
    merchant_country CHAR(2),
    customer_country CHAR(2),
    account_age_days INT,
    transaction_count_24h INT,
    transaction_count_7d INT,
    previous_transaction_amount DECIMAL(12, 2),
    is_fraud BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transactions_customer
    ON transactions(customer_id);

CREATE INDEX idx_transactions_timestamp
    ON transactions(transaction_timestamp);

CREATE INDEX idx_transactions_fraud
    ON transactions(is_fraud);