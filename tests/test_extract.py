from src.etl.extract import extract_transactions


def test_extract_transactions():
    df = extract_transactions()

    assert not df.empty
    assert len(df) == 50_000
    assert "transaction_id" in df.columns
    assert "customer_id" in df.columns
    assert "amount" in df.columns
    assert "is_fraud" in df.columns