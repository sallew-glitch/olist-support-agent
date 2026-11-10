"""The five read-only tools the agent can call.

Every tool:
- takes a validated ID argument (32 lowercase hex characters, like all Olist IDs),
- runs fixed, parameterized SQL (model-provided strings are only ever bound as parameters),
- returns a small JSON-serializable dict, or an {"error": {...}} object.
"""

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import date
from typing import Any, Callable

ID_PATTERN = "^[0-9a-f]{32}$"
MAX_ITEMS = 20
MAX_PAYMENTS = 10
MAX_TOP_CATEGORIES = 5
MAX_RESULT_CHARS = 4000  # final guard; per-tool caps keep results well below this
DELIVERY_NOTE = "No customer delivery date is recorded for this order."  # hint field (see apply_hints)


def error(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def _id_schema(name: str, what: str) -> dict:
    return {
        "type": "object",
        "properties": {
            name: {
                "type": "string",
                "pattern": ID_PATTERN,
                "description": f"The {what}: 32 lowercase hexadecimal characters.",
            }
        },
        "required": [name],
        "additionalProperties": False,
    }


def _days_vs_estimate(delivered: str | None, estimated: str | None) -> int | None:
    """Calendar days delivered after (+) or before (-) the estimate. None if either date is missing.

    Olist estimates are dates (time is always 00:00:00), so we compare dates, not timestamps.
    """
    if not delivered or not estimated:
        return None
    return (date.fromisoformat(delivered[:10]) - date.fromisoformat(estimated[:10])).days


def _order_exists(conn: sqlite3.Connection, order_id: str) -> bool:
    return conn.execute("SELECT 1 FROM orders WHERE order_id = ?", (order_id,)).fetchone() is not None


# --- tools ------------------------------------------------------------------


def get_order_status(conn: sqlite3.Connection, args: dict) -> dict:
    row = conn.execute(
        """
        SELECT o.order_id, o.order_status, o.order_purchase_timestamp,
               (SELECT COUNT(*) FROM order_items i WHERE i.order_id = o.order_id) AS item_count
        FROM orders o WHERE o.order_id = ?
        """,
        (args["order_id"],),
    ).fetchone()
    if row is None:
        return error("not_found", f"No order with order_id {args['order_id']}.")
    return {
        "order_id": row["order_id"],
        "status": row["order_status"],
        "purchased_at": row["order_purchase_timestamp"],
        "item_count": row["item_count"],
    }


def get_delivery_performance(conn: sqlite3.Connection, args: dict) -> dict:
    row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (args["order_id"],)).fetchone()
    if row is None:
        return error("not_found", f"No order with order_id {args['order_id']}.")
    delivered = row["order_delivered_customer_date"]
    estimated = row["order_estimated_delivery_date"]
    days = _days_vs_estimate(delivered, estimated)
    result = {
        "order_id": row["order_id"],
        "status": row["order_status"],
        "purchased_at": row["order_purchase_timestamp"],
        "approved_at": row["order_approved_at"],
        "handed_to_carrier_at": row["order_delivered_carrier_date"],
        "delivered_to_customer_at": delivered,
        "estimated_delivery_date": estimated[:10] if estimated else None,
        "days_vs_estimate": days,
        "is_late": None if days is None else days > 0,
    }
    if delivered is None:
        result["note"] = DELIVERY_NOTE
    return result


def get_order_items(conn: sqlite3.Connection, args: dict) -> dict:
    order_id = args["order_id"]
    if not _order_exists(conn, order_id):
        return error("not_found", f"No order with order_id {order_id}.")
    rows = conn.execute(
        """
        SELECT i.order_item_id, i.product_id, i.seller_id, i.price, i.freight_value,
               COALESCE(t.product_category_name_english, p.product_category_name) AS category
        FROM order_items i
        LEFT JOIN products p ON p.product_id = i.product_id
        LEFT JOIN product_category_name_translation t ON t.product_category_name = p.product_category_name
        WHERE i.order_id = ?
        ORDER BY i.order_item_id
        """,
        (order_id,),
    ).fetchall()
    result = {
        "order_id": order_id,
        "item_count": len(rows),
        "total_price": round(sum(r["price"] for r in rows), 2),
        "total_freight": round(sum(r["freight_value"] for r in rows), 2),
        "items": [
            {
                "order_item_id": r["order_item_id"],
                "product_id": r["product_id"],
                "category": r["category"],
                "price": r["price"],
                "freight_value": r["freight_value"],
                "seller_id": r["seller_id"],
            }
            for r in rows[:MAX_ITEMS]
        ],
    }
    if len(rows) > MAX_ITEMS:
        result["truncated"] = f"Showing first {MAX_ITEMS} of {len(rows)} items; totals cover all items."
    if not rows:
        result["note"] = "This order has no items recorded."
    return result


def get_order_payments(conn: sqlite3.Connection, args: dict) -> dict:
    order_id = args["order_id"]
    if not _order_exists(conn, order_id):
        return error("not_found", f"No order with order_id {order_id}.")
    rows = conn.execute(
        """
        SELECT payment_sequential, payment_type, payment_installments, payment_value
        FROM order_payments WHERE order_id = ? ORDER BY payment_sequential
        """,
        (order_id,),
    ).fetchall()
    result = {
        "order_id": order_id,
        "payment_count": len(rows),
        "total_value": round(sum(r["payment_value"] for r in rows), 2),
        "payments": [
            {
                "sequential": r["payment_sequential"],
                "type": r["payment_type"],
                "installments": r["payment_installments"],
                "value": r["payment_value"],
            }
            for r in rows[:MAX_PAYMENTS]
        ],
    }
    if len(rows) > MAX_PAYMENTS:
        result["truncated"] = f"Showing first {MAX_PAYMENTS} of {len(rows)} payments; total covers all."
    if not rows:
        result["note"] = "This order has no payments recorded."
    return result


def get_seller_stats(conn: sqlite3.Connection, args: dict) -> dict:
    seller_id = args["seller_id"]
    seller = conn.execute("SELECT * FROM sellers WHERE seller_id = ?", (seller_id,)).fetchone()
    if seller is None:
        return error("not_found", f"No seller with seller_id {seller_id}.")

    counts = conn.execute(
        "SELECT COUNT(DISTINCT order_id) AS orders, COUNT(*) AS items FROM order_items WHERE seller_id = ?",
        (seller_id,),
    ).fetchone()

    # Reviews belong to orders. Count each (review, order) pair once.
    reviews = conn.execute(
        """
        SELECT AVG(review_score) AS avg_score, COUNT(*) AS n FROM (
            SELECT DISTINCT r.review_id, r.order_id, r.review_score
            FROM order_reviews r
            WHERE r.order_id IN (SELECT order_id FROM order_items WHERE seller_id = ?)
        )
        """,
        (seller_id,),
    ).fetchone()

    # Late = delivered on a calendar date after the estimated date (same rule as get_delivery_performance).
    delivery = conn.execute(
        """
        SELECT COUNT(*) AS delivered,
               SUM(date(order_delivered_customer_date) > date(order_estimated_delivery_date)) AS late
        FROM orders
        WHERE order_id IN (SELECT order_id FROM order_items WHERE seller_id = ?)
          AND order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
          AND order_estimated_delivery_date IS NOT NULL
        """,
        (seller_id,),
    ).fetchone()

    categories = conn.execute(
        """
        SELECT COALESCE(t.product_category_name_english, p.product_category_name, 'unknown') AS category,
               COUNT(*) AS items
        FROM order_items i
        LEFT JOIN products p ON p.product_id = i.product_id
        LEFT JOIN product_category_name_translation t ON t.product_category_name = p.product_category_name
        WHERE i.seller_id = ?
        GROUP BY 1 ORDER BY items DESC, category LIMIT ?
        """,
        (seller_id, MAX_TOP_CATEGORIES),
    ).fetchall()

    delivered = delivery["delivered"]
    return {
        "seller_id": seller_id,
        "city": seller["seller_city"],
        "state": seller["seller_state"],
        "order_count": counts["orders"],
        "items_sold": counts["items"],
        "review_count": reviews["n"],
        "average_review_score": round(reviews["avg_score"], 2) if reviews["avg_score"] is not None else None,
        "delivered_orders": delivered,
        "late_delivered_orders": delivery["late"] or 0,
        "late_share": round((delivery["late"] or 0) / delivered, 3) if delivered else None,
        "top_categories": [{"category": c["category"], "items": c["items"]} for c in categories],
        "note": "Review scores are per order; on orders with several sellers the score counts for each seller.",
    }


# --- registry ---------------------------------------------------------------


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict
    func: Callable[[sqlite3.Connection, dict], dict]


TOOLS: dict[str, Tool] = {
    t.name: t
    for t in [
        Tool(
            "get_order_status",
            "Get an order's current status (e.g. delivered, shipped, canceled), purchase timestamp and number of items.",
            _id_schema("order_id", "order ID"),
            get_order_status,
        ),
        Tool(
            "get_delivery_performance",
            "Get an order's timeline (purchase, approval, handed to carrier, delivered to customer, estimated "
            "delivery date) and how many calendar days it was delivered after (+) or before (-) the estimate. "
            "Delivery fields are null when the order has not been delivered.",
            _id_schema("order_id", "order ID"),
            get_delivery_performance,
        ),
        Tool(
            "get_order_items",
            "List the items in an order: product ID, product category (English), price, freight value and seller ID, "
            f"plus totals. At most {MAX_ITEMS} items are listed.",
            _id_schema("order_id", "order ID"),
            get_order_items,
        ),
        Tool(
            "get_order_payments",
            "List how an order was paid: payment type (credit_card, boleto, voucher, debit_card), number of "
            f"installments and value, plus the total. At most {MAX_PAYMENTS} payments are listed.",
            _id_schema("order_id", "order ID"),
            get_order_payments,
        ),
        Tool(
            "get_seller_stats",
            "Get aggregate stats for a seller: number of orders and items, average review score (1-5, numbers only), "
            "share of delivered orders that arrived after the estimated date, and top product categories.",
            _id_schema("seller_id", "seller ID"),
            get_seller_stats,
        ),
    ]
}


def tool_specs() -> list[dict]:
    """Provider-neutral tool definitions: name, description, JSON schema."""
    return [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in TOOLS.values()]


def validate_args(schema: dict, args: Any) -> str | None:
    """Validate arguments against our (deliberately simple) tool schemas. Returns a problem or None."""
    if not isinstance(args, dict):
        return "Arguments must be a JSON object."
    unknown = set(args) - set(schema["properties"])
    if unknown:
        return f"Unknown argument(s): {sorted(unknown)}."
    for name in schema["required"]:
        if name not in args:
            return f"Missing required argument '{name}'."
    for name, value in args.items():
        prop = schema["properties"][name]
        if not isinstance(value, str):
            return f"Argument '{name}' must be a string."
        if not re.fullmatch(prop["pattern"], value):
            return f"Argument '{name}' must be 32 lowercase hexadecimal characters."
    return None


def execute_tool(conn: sqlite3.Connection, name: str, args: Any) -> dict:
    """Run one tool call. Never raises: every failure becomes an error object for the model."""
    tool = TOOLS.get(name)
    if tool is None:
        return error("unknown_tool", f"No tool named '{name}'. Available: {sorted(TOOLS)}.")
    problem = validate_args(tool.input_schema, args)
    if problem:
        return error("invalid_arguments", problem)
    try:
        result = tool.func(conn, args)
    except sqlite3.Error as e:
        return error("database_error", f"{type(e).__name__}: {e}")
    except Exception as e:  # a bug in a tool should not crash the run
        return error("tool_failed", f"{type(e).__name__}: {e}")
    if len(json.dumps(result)) > MAX_RESULT_CHARS:
        return error("result_too_large", f"Result exceeded {MAX_RESULT_CHARS} characters.")
    return result


# --- hints experiment ---------------------------------------------------------

# Explanatory hint fields that --hints off hides from the model. Data fields (counts, totals,
# nulls), the empty-list notes, the get_seller_stats note and error objects are never removed.
HINT_FIELDS = {
    "get_delivery_performance": ("note",),
    "get_order_items": ("truncated",),
    "get_order_payments": ("truncated",),
}


def apply_hints(tool_name: str, result: dict, hints: bool) -> dict:
    """The tool result as the model will see it. With hints on, `result` is returned unchanged."""
    if hints:
        return result
    hidden = HINT_FIELDS.get(tool_name, ())
    return {key: value for key, value in result.items() if key not in hidden}
