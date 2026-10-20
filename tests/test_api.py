from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def valid_transaction():
    return {
        "transaction_id": 999999,
        "customer_id": 12345,
        "transaction_timestamp": "2025-01-15T14:30:00",
        "amount": 250.00,
        "merchant_category": "electronics",
        "merchant_country": "US",
        "customer_country": "US",
        "account_age_days": 120,
        "transaction_count_24h": 3,
        "transaction_count_7d": 15,
        "previous_transaction_amount": 50.00,
    }


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prediction_returns_result():
    response = client.post(
        "/predict",
        json=valid_transaction(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "fraud_probability" in data
    assert "threshold" in data
    assert "is_fraud" in data
    assert "decision" in data

    assert 0 <= data["fraud_probability"] <= 1
    assert data["threshold"] == 0.75
    assert data["decision"] in {"FRAUD", "LEGITIMATE"}


def test_high_risk_transaction_is_fraud():
    transaction = valid_transaction()

    transaction.update(
        {
            "transaction_id": 999998,
            "amount": 1500.00,
            "merchant_country": "NG",
            "customer_country": "US",
            "account_age_days": 15,
            "transaction_count_24h": 25,
            "transaction_count_7d": 80,
            "previous_transaction_amount": 20.00,
            "transaction_timestamp": "2025-01-15T02:30:00",
        }
    )

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["fraud_probability"] >= data["threshold"]
    assert data["is_fraud"] is True
    assert data["decision"] == "FRAUD"


def test_missing_required_field_returns_422():
    transaction = valid_transaction()
    del transaction["amount"]

    response = client.post(
        "/predict",
        json=transaction,
    )

    assert response.status_code == 422
