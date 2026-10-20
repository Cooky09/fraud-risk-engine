from src.database.connection import engine
from src.etl.extract import extract_transactions
from src.etl.transform import transform_transactions
from src.etl.load import load_transactions


def main() -> None:
    print("Starting ETL pipeline...")

    print("Extracting transactions...")
    raw_df = extract_transactions()

    print(f"Extracted {len(raw_df):,} transactions")

    print("Transforming transactions...")
    transformed_df = transform_transactions(raw_df)

    print(
        f"Transformed {len(transformed_df):,} transactions"
    )

    print("Loading transactions into MySQL...")
    loaded_count = load_transactions(
        transformed_df,
        engine,
    )

    print(
        f"Loaded {loaded_count:,} transactions into MySQL."
    )

    print("ETL pipeline completed successfully.")


if __name__ == "__main__":
    main()