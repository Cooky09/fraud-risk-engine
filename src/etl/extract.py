from pathlib import Path

import pandas as pd


RAW_DATA_PATH = Path("data/raw/transactions.csv")


def extract_transactions(
    file_path: Path = RAW_DATA_PATH,
) -> pd.DataFrame:
    """Extract raw transaction data from CSV."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Transaction dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("Transaction dataset is empty.")

    return df