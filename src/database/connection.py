from sqlalchemy import create_engine, text

from src.config.settings import DATABASE_URL


engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)


def test_connection() -> bool:
    with engine.connect() as connection:
        connection.execute(text("SELECT 1"))

    return True