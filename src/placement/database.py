"""Load the CSV into a SQLite database and query training data with SQL."""
import sqlite3
from pathlib import Path

import pandas as pd

from .config import CSV_PATH, DB_PATH

TABLE = "students"


def load_csv_to_db(csv_path: Path = CSV_PATH, db_path: Path = DB_PATH) -> int:
    df = pd.read_csv(csv_path)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(TABLE, conn, if_exists="replace", index=False)
    return len(df)


def fetch_training_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Fetch rows with valid core fields; the WHERE clause drops corrupt records."""
    query = f"""
        SELECT *
        FROM {TABLE}
        WHERE cgpa BETWEEN 0 AND 10
          AND aptitude_score BETWEEN 0 AND 100
          AND placed IN (0, 1)
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def placement_rate_by_branch(db_path: Path = DB_PATH) -> pd.DataFrame:
    query = f"""
        SELECT branch,
               COUNT(*)                     AS students,
               ROUND(AVG(placed) * 100, 1)  AS placement_rate_pct,
               ROUND(AVG(cgpa), 2)          AS avg_cgpa
        FROM {TABLE}
        GROUP BY branch
        ORDER BY placement_rate_pct DESC
    """
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
