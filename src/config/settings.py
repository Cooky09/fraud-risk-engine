import os

from dotenv import load_dotenv


load_dotenv()


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://fraud_app:fraud_password@127.0.0.1:3307/fraud_risk",
)