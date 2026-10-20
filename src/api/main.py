from fastapi import FastAPI
from pydantic import BaseModel

from src.ml.predict import predict_transaction


app = FastAPI(
    title="Fraud Risk Engine",
    version="1.0.0",
)


class Transaction(BaseModel):
    transaction_id: int
    customer_id: int
    transaction_timestamp: str
    amount: float
    merchant_category: str
    merchant_country: str
    customer_country: str
    account_age_days: int
    transaction_count_24h: int
    transaction_count_7d: int
    previous_transaction_amount: float


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(transaction: Transaction):
    return predict_transaction(transaction.model_dump())