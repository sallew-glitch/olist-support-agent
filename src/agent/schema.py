"""Database schema, shared by scripts/prepare_data.py and the test fixtures.

Column names match the Olist CSV headers exactly (including the dataset's own
typos such as "product_name_lenght"). Timestamps are stored as TEXT in the
dataset's "YYYY-MM-DD HH:MM:SS" form, which sorts correctly and works with
SQLite's date functions. Zip-code prefixes stay TEXT to keep leading zeros.
"""

import sqlite3

# table name -> source CSV, kept columns (name, SQLite type), primary key, and
# CSV columns that are deliberately NOT loaded.
TABLES = {
    "customers": {
        "csv": "olist_customers_dataset.csv",
        "columns": [
            ("customer_id", "TEXT"),
            ("customer_unique_id", "TEXT"),
            ("customer_zip_code_prefix", "TEXT"),
            ("customer_city", "TEXT"),
            ("customer_state", "TEXT"),
        ],
        "primary_key": ["customer_id"],
    },
    "sellers": {
        "csv": "olist_sellers_dataset.csv",
        "columns": [
            ("seller_id", "TEXT"),
            ("seller_zip_code_prefix", "TEXT"),
            ("seller_city", "TEXT"),
            ("seller_state", "TEXT"),
        ],
        "primary_key": ["seller_id"],
    },
    "product_category_name_translation": {
        "csv": "product_category_name_translation.csv",
        "columns": [
            ("product_category_name", "TEXT"),
            ("product_category_name_english", "TEXT"),
        ],
        "primary_key": ["product_category_name"],
    },
    "products": {
        "csv": "olist_products_dataset.csv",
        "columns": [
            ("product_id", "TEXT"),
            ("product_category_name", "TEXT"),
            ("product_name_lenght", "INTEGER"),
            ("product_description_lenght", "INTEGER"),
            ("product_photos_qty", "INTEGER"),
            ("product_weight_g", "INTEGER"),
            ("product_length_cm", "INTEGER"),
            ("product_height_cm", "INTEGER"),
            ("product_width_cm", "INTEGER"),
        ],
        "primary_key": ["product_id"],
    },
    "orders": {
        "csv": "olist_orders_dataset.csv",
        "columns": [
            ("order_id", "TEXT"),
            ("customer_id", "TEXT"),
            ("order_status", "TEXT"),
            ("order_purchase_timestamp", "TEXT"),
            ("order_approved_at", "TEXT"),
            ("order_delivered_carrier_date", "TEXT"),
            ("order_delivered_customer_date", "TEXT"),
            ("order_estimated_delivery_date", "TEXT"),
        ],
        "primary_key": ["order_id"],
    },
    "order_items": {
        "csv": "olist_order_items_dataset.csv",
        "columns": [
            ("order_id", "TEXT"),
            ("order_item_id", "INTEGER"),
            ("product_id", "TEXT"),
            ("seller_id", "TEXT"),
            ("shipping_limit_date", "TEXT"),
            ("price", "REAL"),
            ("freight_value", "REAL"),
        ],
        "primary_key": ["order_id", "order_item_id"],
    },
    "order_payments": {
        "csv": "olist_order_payments_dataset.csv",
        "columns": [
            ("order_id", "TEXT"),
            ("payment_sequential", "INTEGER"),
            ("payment_type", "TEXT"),
            ("payment_installments", "INTEGER"),
            ("payment_value", "REAL"),
        ],
        "primary_key": ["order_id", "payment_sequential"],
    },
    "order_reviews": {
        "csv": "olist_order_reviews_dataset.csv",
        "columns": [
            ("review_id", "TEXT"),
            ("order_id", "TEXT"),
            ("review_score", "INTEGER"),
            ("review_creation_date", "TEXT"),
            ("review_answer_timestamp", "TEXT"),
        ],
        # review_id is not unique in this dataset (one review can cover several
        # orders), so there is no primary key.
        "primary_key": None,
        # Review TEXT never enters the database, so no tool can expose it.
        "dropped": ["review_comment_title", "review_comment_message"],
    },
}

# Indexes for the lookups the tools make that primary keys do not already cover.
INDEXES = [
    "CREATE INDEX idx_order_items_seller ON order_items (seller_id)",
    "CREATE INDEX idx_order_reviews_order ON order_reviews (order_id)",
]


def create_schema(conn: sqlite3.Connection) -> None:
    """Create all tables and indexes in an empty database."""
    for table, spec in TABLES.items():
        cols = [f"{name} {sql_type}" for name, sql_type in spec["columns"]]
        if spec["primary_key"]:
            cols.append(f"PRIMARY KEY ({', '.join(spec['primary_key'])})")
        conn.execute(f"CREATE TABLE {table} ({', '.join(cols)})")
    for statement in INDEXES:
        conn.execute(statement)
    conn.commit()
