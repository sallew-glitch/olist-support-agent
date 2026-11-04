"""Shared fixtures: a tiny hand-made database with the real schema, and a network block."""

import socket
import sqlite3

import pytest

from agent.db import connect_readonly
from agent.schema import create_schema

ORDER_LATE = "a" * 32       # delivered 3 days after estimate, 2 sellers, 2 payments
ORDER_EARLY = "b" * 32      # delivered 5 days before estimate, two reviews
ORDER_SHIPPED = "c" * 32    # not delivered yet
ORDER_CANCELED = "d" * 32   # no items, no payments
ORDER_BIG = "e" * 32        # 25 items, 12 payments (exceeds caps), delivered on the estimate date
ORDER_UNKNOWN = "9" * 32    # valid format, not in the DB

SELLER_1 = "1" * 32
SELLER_2 = "2" * 32
SELLER_3 = "3" * 32         # exists, no orders
SELLER_UNKNOWN = "8" * 32

PROD_TRANSLATED = "0" * 31 + "1"
PROD_UNTRANSLATED = "0" * 31 + "2"
PROD_NO_CATEGORY = "0" * 31 + "3"

CUSTOMER = "c0" * 16


def build_fixture_db(path) -> None:
    conn = sqlite3.connect(path)
    create_schema(conn)
    conn.execute("INSERT INTO customers VALUES (?, ?, '01037', 'sao paulo', 'SP')", (CUSTOMER, CUSTOMER))
    conn.executemany(
        "INSERT INTO sellers VALUES (?, ?, ?, ?)",
        [(SELLER_1, "13023", "campinas", "SP"), (SELLER_2, "01001", "sao paulo", "SP"), (SELLER_3, "20000", "rio de janeiro", "RJ")],
    )
    conn.execute("INSERT INTO product_category_name_translation VALUES ('beleza_saude', 'health_beauty')")
    conn.executemany(
        "INSERT INTO products (product_id, product_category_name) VALUES (?, ?)",
        [(PROD_TRANSLATED, "beleza_saude"), (PROD_UNTRANSLATED, "pc_gamer"), (PROD_NO_CATEGORY, None)],
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [
            (ORDER_LATE, CUSTOMER, "delivered", "2018-01-01 10:00:00", "2018-01-01 11:00:00",
             "2018-01-03 09:00:00", "2018-01-15 12:00:00", "2018-01-12 00:00:00"),
            (ORDER_EARLY, CUSTOMER, "delivered", "2018-02-01 10:00:00", "2018-02-01 10:30:00",
             "2018-02-02 08:00:00", "2018-02-05 09:00:00", "2018-02-10 00:00:00"),
            (ORDER_SHIPPED, CUSTOMER, "shipped", "2018-03-01 10:00:00", "2018-03-01 12:00:00",
             "2018-03-03 08:00:00", None, "2018-03-20 00:00:00"),
            (ORDER_CANCELED, CUSTOMER, "canceled", "2018-03-05 10:00:00", None, None, None, "2018-03-25 00:00:00"),
            (ORDER_BIG, CUSTOMER, "delivered", "2018-04-01 10:00:00", "2018-04-01 10:10:00",
             "2018-04-02 08:00:00", "2018-04-10 18:00:00", "2018-04-10 00:00:00"),
        ],
    )
    items = [
        (ORDER_LATE, 1, PROD_TRANSLATED, SELLER_1, "2018-01-05 00:00:00", 100.0, 10.0),
        (ORDER_LATE, 2, PROD_UNTRANSLATED, SELLER_2, "2018-01-05 00:00:00", 50.5, 5.25),
        (ORDER_EARLY, 1, PROD_NO_CATEGORY, SELLER_1, "2018-02-03 00:00:00", 20.0, 2.0),
        (ORDER_SHIPPED, 1, PROD_TRANSLATED, SELLER_1, "2018-03-04 00:00:00", 30.0, 3.0),
    ]
    items += [(ORDER_BIG, n, PROD_TRANSLATED, SELLER_2, "2018-04-03 00:00:00", 1.0, 0.5) for n in range(1, 26)]
    conn.executemany("INSERT INTO order_items VALUES (?, ?, ?, ?, ?, ?, ?)", items)
    payments = [
        (ORDER_LATE, 1, "credit_card", 3, 120.0),
        (ORDER_LATE, 2, "voucher", 1, 45.75),
        (ORDER_EARLY, 1, "boleto", 1, 22.0),
        (ORDER_SHIPPED, 1, "credit_card", 1, 33.0),
    ]
    payments += [(ORDER_BIG, n, "voucher", 1, 3.125) for n in range(1, 13)]
    conn.executemany("INSERT INTO order_payments VALUES (?, ?, ?, ?, ?)", payments)
    conn.executemany(
        "INSERT INTO order_reviews VALUES (?, ?, ?, '2018-01-20 00:00:00', '2018-01-21 00:00:00')",
        [
            ("r" * 32, ORDER_LATE, 2),
            ("r" * 32, ORDER_LATE, 2),  # exact duplicate row: must be counted once
            ("s" * 32, ORDER_EARLY, 4),
            ("t" * 32, ORDER_EARLY, 5),  # second review on the same order
        ],
    )
    conn.commit()
    conn.close()


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Tests must never reach the network (and so can never spend API credit)."""

    def blocked(*args, **kwargs):
        raise RuntimeError("Network access is disabled in tests")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "fixture.db"
    build_fixture_db(path)
    return path


@pytest.fixture
def conn(db_path):
    c = connect_readonly(db_path)
    yield c
    c.close()
