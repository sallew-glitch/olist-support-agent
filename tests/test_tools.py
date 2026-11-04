import json

import pytest

from agent.tools import MAX_ITEMS, MAX_PAYMENTS, TOOLS, execute_tool, tool_specs
from conftest import (
    ORDER_BIG, ORDER_CANCELED, ORDER_EARLY, ORDER_LATE, ORDER_SHIPPED, ORDER_UNKNOWN,
    PROD_TRANSLATED, SELLER_1, SELLER_2, SELLER_3, SELLER_UNKNOWN,
)


def test_exactly_five_tools_with_schemas():
    assert sorted(TOOLS) == [
        "get_delivery_performance", "get_order_items", "get_order_payments", "get_order_status", "get_seller_stats",
    ]
    for spec in tool_specs():
        assert spec["description"]
        assert spec["input_schema"]["additionalProperties"] is False


# --- get_order_status ---

def test_order_status(conn):
    result = execute_tool(conn, "get_order_status", {"order_id": ORDER_LATE})
    assert result == {"order_id": ORDER_LATE, "status": "delivered", "purchased_at": "2018-01-01 10:00:00", "item_count": 2}


def test_order_status_canceled_without_items(conn):
    result = execute_tool(conn, "get_order_status", {"order_id": ORDER_CANCELED})
    assert result["status"] == "canceled"
    assert result["item_count"] == 0


@pytest.mark.parametrize("tool", ["get_order_status", "get_delivery_performance", "get_order_items", "get_order_payments"])
def test_unknown_order_returns_not_found(conn, tool):
    result = execute_tool(conn, tool, {"order_id": ORDER_UNKNOWN})
    assert result["error"]["code"] == "not_found"


# --- get_delivery_performance ---

def test_delivery_late(conn):
    result = execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_LATE})
    assert result["days_vs_estimate"] == 3
    assert result["is_late"] is True
    assert result["estimated_delivery_date"] == "2018-01-12"
    assert result["handed_to_carrier_at"] == "2018-01-03 09:00:00"


def test_delivery_early(conn):
    result = execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_EARLY})
    assert result["days_vs_estimate"] == -5
    assert result["is_late"] is False


def test_delivery_on_estimate_date_is_not_late(conn):
    # Delivered at 18:00 on the estimated date: same calendar day, so not late.
    result = execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_BIG})
    assert result["days_vs_estimate"] == 0
    assert result["is_late"] is False


def test_delivery_undelivered_is_null_safe(conn):
    result = execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_SHIPPED})
    assert result["delivered_to_customer_at"] is None
    assert result["days_vs_estimate"] is None
    assert result["is_late"] is None
    assert "note" in result


def test_delivery_canceled_has_nulls(conn):
    result = execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_CANCELED})
    assert result["approved_at"] is None
    assert result["handed_to_carrier_at"] is None
    assert result["is_late"] is None


# --- get_order_items ---

def test_order_items_with_translation_and_fallback(conn):
    result = execute_tool(conn, "get_order_items", {"order_id": ORDER_LATE})
    assert result["item_count"] == 2
    assert result["total_price"] == 150.5
    assert result["total_freight"] == 15.25
    first, second = result["items"]
    assert first == {
        "order_item_id": 1, "product_id": PROD_TRANSLATED, "category": "health_beauty",
        "price": 100.0, "freight_value": 10.0, "seller_id": SELLER_1,
    }
    assert second["category"] == "pc_gamer"  # no English translation: Portuguese name kept
    assert second["seller_id"] == SELLER_2


def test_order_items_without_category(conn):
    result = execute_tool(conn, "get_order_items", {"order_id": ORDER_EARLY})
    assert result["items"][0]["category"] is None


def test_order_items_capped(conn):
    result = execute_tool(conn, "get_order_items", {"order_id": ORDER_BIG})
    assert result["item_count"] == 25
    assert len(result["items"]) == MAX_ITEMS
    assert result["total_price"] == 25.0  # totals cover all items
    assert "truncated" in result


def test_order_items_empty(conn):
    result = execute_tool(conn, "get_order_items", {"order_id": ORDER_CANCELED})
    assert result["items"] == []
    assert "note" in result


# --- get_order_payments ---

def test_order_payments(conn):
    result = execute_tool(conn, "get_order_payments", {"order_id": ORDER_LATE})
    assert result["payment_count"] == 2
    assert result["total_value"] == 165.75
    assert result["payments"][0] == {"sequential": 1, "type": "credit_card", "installments": 3, "value": 120.0}


def test_order_payments_capped(conn):
    result = execute_tool(conn, "get_order_payments", {"order_id": ORDER_BIG})
    assert result["payment_count"] == 12
    assert len(result["payments"]) == MAX_PAYMENTS
    assert result["total_value"] == 37.5
    assert "truncated" in result


def test_order_payments_none_recorded(conn):
    result = execute_tool(conn, "get_order_payments", {"order_id": ORDER_CANCELED})
    assert result["payments"] == []
    assert "note" in result


# --- get_seller_stats ---

def test_seller_stats(conn):
    result = execute_tool(conn, "get_seller_stats", {"seller_id": SELLER_1})
    assert result["order_count"] == 3
    assert result["items_sold"] == 3
    # Reviews: LATE=2 (duplicate row counted once), EARLY=4 and 5 -> (2+4+5)/3
    assert result["review_count"] == 3
    assert result["average_review_score"] == 3.67
    # Delivered: LATE (late) and EARLY (early); SHIPPED is excluded
    assert result["delivered_orders"] == 2
    assert result["late_delivered_orders"] == 1
    assert result["late_share"] == 0.5
    assert result["top_categories"] == [{"category": "health_beauty", "items": 2}, {"category": "unknown", "items": 1}]


def test_seller_stats_on_time_delivery_not_late(conn):
    result = execute_tool(conn, "get_seller_stats", {"seller_id": SELLER_2})
    assert result["order_count"] == 2
    assert result["delivered_orders"] == 2
    assert result["late_delivered_orders"] == 1  # LATE yes, BIG (same day) no
    assert result["average_review_score"] == 2.0
    assert result["top_categories"][0] == {"category": "health_beauty", "items": 25}


def test_seller_without_orders(conn):
    result = execute_tool(conn, "get_seller_stats", {"seller_id": SELLER_3})
    assert result["order_count"] == 0
    assert result["average_review_score"] is None
    assert result["late_share"] is None
    assert result["top_categories"] == []


def test_unknown_seller(conn):
    assert execute_tool(conn, "get_seller_stats", {"seller_id": SELLER_UNKNOWN})["error"]["code"] == "not_found"


# --- validation and error handling ---

@pytest.mark.parametrize(
    "args",
    [
        {},
        {"order_id": "abc"},
        {"order_id": ORDER_LATE.upper()},
        {"order_id": ORDER_LATE + "'; DROP TABLE orders; --"},
        {"order_id": 12345},
        {"order_id": ORDER_LATE, "extra": "x"},
        "not an object",
        None,
    ],
)
def test_invalid_arguments(conn, args):
    assert execute_tool(conn, "get_order_status", args)["error"]["code"] == "invalid_arguments"


def test_unknown_tool(conn):
    assert execute_tool(conn, "drop_tables", {})["error"]["code"] == "unknown_tool"


def test_database_error_is_returned_not_raised(conn):
    conn.close()
    result = execute_tool(conn, "get_order_status", {"order_id": ORDER_LATE})
    assert result["error"]["code"] == "database_error"


def test_results_are_small_json(conn):
    for name, arg in [("get_order_items", {"order_id": ORDER_BIG}), ("get_seller_stats", {"seller_id": SELLER_2})]:
        assert len(json.dumps(execute_tool(conn, name, arg))) < 4000
