import hashlib
import sqlite3

import pytest

from agent.db import connect_readonly
from agent.tools import TOOLS, execute_tool
from conftest import ORDER_LATE, SELLER_1


@pytest.mark.parametrize(
    "statement",
    [
        "INSERT INTO sellers VALUES ('x', '1', 'c', 'S')",
        "UPDATE orders SET order_status = 'canceled'",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "CREATE TABLE evil (x)",
        "ATTACH DATABASE 'other.db' AS other",
        "PRAGMA query_only = OFF",
    ],
)
def test_connection_cannot_write(conn, statement, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # so a failed ATTACH guard could not leave a file in the repo
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute(statement)
    assert not (tmp_path / "other.db").exists()


def test_file_is_read_only_even_without_other_guards(db_path):
    # Remove the authorizer and switch query_only off: the read-only file mode still refuses writes.
    c = connect_readonly(db_path)
    c.set_authorizer(None)
    c.execute("PRAGMA query_only = OFF")
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        c.execute("DELETE FROM orders")
    c.close()


def test_running_every_tool_leaves_file_unchanged(db_path):
    before = hashlib.sha256(db_path.read_bytes()).hexdigest()
    c = connect_readonly(db_path)
    for name in TOOLS:
        arg = {"seller_id": SELLER_1} if name == "get_seller_stats" else {"order_id": ORDER_LATE}
        assert "error" not in execute_tool(c, name, arg)
    c.close()
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before


def test_missing_database_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="prepare_data"):
        connect_readonly(tmp_path / "nope.db")
