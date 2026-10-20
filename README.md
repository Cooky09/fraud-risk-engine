# Fraud Risk Engine

[![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![MySQL](https://img.shields.io/badge/MySQL-Database-4479A1?logo=mysql&logoColor=white)](https://www.mysql.com/)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML-9ACD32)](https://lightgbm.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-ML-189AB4)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Pytest-Testing-0A9EDC?logo=pytest&logoColor=white)](https://pytest.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Processing-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

An end-to-end machine learning fraud detection system built with Python, MySQL, LightGBM, XGBoost, FastAPI, Docker, and Pytest.

The system processes raw transaction data through an ETL pipeline, stores engineered transaction data in MySQL, trains and evaluates machine learning models, tunes the fraud decision threshold, and exposes real-time fraud predictions through a REST API.

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
