# Fraud Risk Engine

An end-to-end machine learning fraud detection system built with Python, MySQL, LightGBM, XGBoost, FastAPI, Docker, and Pytest.

The system processes transaction data through an ETL pipeline, engineers fraud-related features, trains machine learning models, evaluates their performance, tunes the fraud decision threshold, and exposes real-time fraud predictions through a REST API.

---

## Features

- End-to-end ETL pipeline
- MySQL transaction storage
- Fraud-focused feature engineering
- LightGBM and XGBoost models
- Stratified train/validation/test splitting
- Imbalanced classification evaluation
- Precision, recall, F1, PR-AUC, and ROC-AUC evaluation
- Decision threshold tuning
- Real-time fraud prediction API
- Dockerized MySQL environment
- Automated tests with Pytest

---

## Architecture

```text
                    transactions.csv
                           |
                           v
                    +-------------+
                    |    Extract  |
                    +-------------+
                           |
                           v
                    +-------------+
                    |  Transform  |
                    |  Features   |
                    +-------------+
                           |
                           v
                    +-------------+
                    |    MySQL    |
                    +-------------+
                           |
                           v
                    +-------------+
                    |   Feature   |
                    | Preparation |
                    +-------------+
                           |
                  +--------+--------+
                  |                 |
                  v                 v
             XGBoost            LightGBM
                  |                 |
                  +--------+--------+
                           |
                           v
                    Model Evaluation
                           |
                           v
                    Threshold Tuning
                           |
                           v
                       LightGBM
                           |
                           v
                       FastAPI
                           |
                  +--------+--------+
                  |                 |
                  v                 v
              LEGITIMATE          FRAUD
