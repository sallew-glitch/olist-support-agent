"""Read-only database access for the agent's tools."""

import sqlite3
from pathlib import Path

# Actions a read-only query needs. Everything else (writes, PRAGMA, ATTACH, ...) is denied.
_ALLOWED_ACTIONS = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}


def _read_only_authorizer(action, *_):
    return sqlite3.SQLITE_OK if action in _ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    """Open the database so that no statement can modify it.

    Three independent guards:
    - mode=ro in the URI: SQLite opens the file read-only.
    - PRAGMA query_only: SQLite rejects write statements on this connection.
    - an authorizer that allows only reads. This also blocks ATTACH, which the
      first two do not: ATTACH could create a new, writable database file.
    """
    path = Path(db_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Database not found: {path}. Run scripts/prepare_data.py first.")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_read_only_authorizer)
    conn.row_factory = sqlite3.Row
    return conn
