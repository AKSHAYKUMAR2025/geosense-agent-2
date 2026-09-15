import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")


DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)


def get_engine():
    """
    Create and return the PostgreSQL database engine.
    """
    return create_engine(DATABASE_URL)


# Test database connection when this file is run directly
if __name__ == "__main__":

    engine = get_engine()

    with engine.connect() as conn:

        result = conn.execute(
            text("SELECT PostGIS_Version();")
        )

        print("PostGIS version:", result.scalar())
        print("Database connection successful!")