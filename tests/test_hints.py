import json

import pytest

from agent.config import DEFAULT_CONFIG, load_settings
from agent.loop import Limits, run_agent
from agent.tools import HINT_FIELDS, TOOLS, apply_hints, execute_tool
from agent.trace import Tracer
from conftest import (
    ORDER_BIG, ORDER_CANCELED, ORDER_EARLY, ORDER_LATE, ORDER_SHIPPED, ORDER_UNKNOWN, SELLER_1,
)
from fakes import FakeLLM, answer, tool_calls

# Every tool on inputs that produce every kind of result: normal, hint fields, empty lists, errors.
CASES = [
    (tool, {"order_id": order})
    for tool in ("get_order_status", "get_delivery_performance", "get_order_items", "get_order_payments")
    for order in (ORDER_LATE, ORDER_EARLY, ORDER_SHIPPED, ORDER_CANCELED, ORDER_BIG, ORDER_UNKNOWN)
] + [("get_seller_stats", {"seller_id": SELLER_1})]


def test_hint_fields_are_exactly_the_agreed_list():
    assert HINT_FIELDS == {
        "get_delivery_performance": ("note",),
        "get_order_items": ("truncated",),
        "get_order_payments": ("truncated",),
    }
    assert set(HINT_FIELDS) <= set(TOOLS)


@pytest.mark.parametrize("tool, args", CASES)
def test_hints_on_returns_result_unchanged(conn, tool, args):
    result = execute_tool(conn, tool, args)
    assert apply_hints(tool, result, hints=True) is result


@pytest.mark.parametrize("tool, args", CASES)
def test_hints_off_removes_only_hint_fields(conn, tool, args):
    result = execute_tool(conn, tool, args)
    visible = apply_hints(tool, result, hints=False)
    removed = set(result) - set(visible)
    assert removed == set(result) & set(HINT_FIELDS.get(tool, ()))
    assert all(visible[k] == result[k] for k in visible)  # nothing else changed
    assert list(visible) == [k for k in result if k in visible]  # order kept


def test_hints_off_on_specific_results(conn):
    delivery = apply_hints("get_delivery_performance", execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_SHIPPED}), False)
    assert "note" not in delivery
    assert delivery["delivered_to_customer_at"] is None and delivery["is_late"] is None  # nulls stay

    payments = apply_hints("get_order_payments", execute_tool(conn, "get_order_payments", {"order_id": ORDER_BIG}), False)
    assert "truncated" not in payments
    assert payments["payment_count"] == 12 and payments["total_value"] == 37.5 and len(payments["payments"]) == 10

    items = apply_hints("get_order_items", execute_tool(conn, "get_order_items", {"order_id": ORDER_BIG}), False)
    assert "truncated" not in items and items["item_count"] == 25

    # Kept on purpose: empty-list notes, the seller note, error objects.
    assert "note" in apply_hints("get_order_payments", execute_tool(conn, "get_order_payments", {"order_id": ORDER_CANCELED}), False)
    assert "note" in apply_hints("get_order_items", execute_tool(conn, "get_order_items", {"order_id": ORDER_CANCELED}), False)
    assert "note" in apply_hints("get_seller_stats", execute_tool(conn, "get_seller_stats", {"seller_id": SELLER_1}), False)
    err = execute_tool(conn, "get_delivery_performance", {"order_id": ORDER_UNKNOWN})
    assert apply_hints("get_delivery_performance", err, False) == err


def _run(conn, tmp_path, hints=None):
    llm = FakeLLM([
        tool_calls(("get_delivery_performance", {"order_id": ORDER_SHIPPED}), ("get_order_payments", {"order_id": ORDER_BIG})),
        answer("done"),
    ])
    kwargs = {} if hints is None else {"hints": hints}
    with Tracer(tmp_path) as tracer:
        result = run_agent("q", llm, conn, Limits(), tracer, **kwargs)
    events = [json.loads(line) for line in result.trace_path.read_text(encoding="utf-8").splitlines()]
    sent = [json.loads(r.content) for r in llm.calls[1]["messages"][-1].tool_results]
    return events, sent, llm


def test_loop_default_is_hints_on_and_unchanged(conn, tmp_path):
    events, sent, llm = _run(conn, tmp_path)
    assert events[0]["hints"] == "on"
    tool_events = [e for e in events if e["event"] == "tool_call"]
    for event, result in zip(tool_events, llm.calls[1]["messages"][-1].tool_results):
        assert "model_visible_result" not in event
        # What the model received is exactly the serialized raw result.
        assert result.content == json.dumps(event["result"], ensure_ascii=False)
        assert event["result_chars"] == len(result.content)
    assert "note" in sent[0] and "truncated" in sent[1]


def test_loop_hints_off(conn, tmp_path):
    events, sent, _ = _run(conn, tmp_path, hints=False)
    assert events[0]["hints"] == "off"
    assert "note" not in sent[0] and "truncated" not in sent[1]
    delivery, payments = [e for e in events if e["event"] == "tool_call"]
    # The trace keeps the raw result and records what the model actually saw.
    assert "note" in delivery["result"] and "note" not in delivery["model_visible_result"]
    assert "truncated" in payments["result"] and payments["model_visible_result"] == sent[1]
    assert payments["result_chars"] == len(json.dumps(sent[1], ensure_ascii=False))


def test_run_start_logs_request_params(conn, tmp_path):
    events, _, _ = _run(conn, tmp_path)
    assert events[0]["request_params"] == {"model": "fake-model", "sampling_params_sent": []}


def test_config_hints():
    assert load_settings(DEFAULT_CONFIG).hints is True
    assert load_settings(DEFAULT_CONFIG, hints="off").hints is False
    with pytest.raises(ValueError):
        load_settings(DEFAULT_CONFIG, hints="maybe")
