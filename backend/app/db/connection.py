"""
SQLite connection + migration helper.

Module: M9 (Execution Trace & Logging) - persistence layer.
"""
import sqlite3
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "app.db"
SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def init_db() -> None:
    """Create tables if they do not already exist. Safe to call on every startup."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        with open(SCHEMA_PATH, "r") as f:
            conn.executescript(f.read())
        conn.commit()


@contextmanager
def get_connection():
    """Context-managed SQLite connection with row factory set for dict-like access."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


if __name__ == "__main__":
    init_db()
    print(f"Database initialized at {DB_PATH}")
