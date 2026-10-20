import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.predict import predict_transaction


transaction = {
    "transaction_id": 999999,
    "customer_id": 99999,
    "transaction_timestamp": "2025-01-15 03:00:00",
    "amount": 500.00,
    "merchant_category": "online",
    "merchant_country": "US",
    "customer_country": "US",
    "account_age_days": 30,
    "transaction_count_24h": 12,
    "transaction_count_7d": 50,
    "previous_transaction_amount": 20.00,
}

result = predict_transaction(transaction)

print()
print("===== Suspicious Transaction Test =====")

for key, value in result.items():
    print(f"{key}: {value}")