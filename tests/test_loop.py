import json

import pytest

from agent.llm.base import PermanentLLMError, TransientLLMError
from agent.loop import Limits, run_agent
from agent.tools import TOOLS, Tool
from agent.trace import Tracer
from conftest import ORDER_LATE, ORDER_UNKNOWN
from fakes import FakeClock, FakeLLM, answer, tool_calls


@pytest.fixture
def run(conn, tmp_path):
    """Run the agent with a FakeLLM script; returns (result, fake_llm)."""

    def _run(script, limits=None, clock=None, prices=None, model="fake-model"):
        llm = FakeLLM(script, model=model)
        clock = clock or FakeClock()
        with Tracer(tmp_path / "traces") as tracer:
            result = run_agent("question?", llm, conn, limits or Limits(), tracer, prices=prices, clock=clock, sleep=clock.sleep)
        return result, llm

    return _run


def tool_results_sent(llm, call_index):
    """The tool results the loop sent to the model on a given call."""
    last = llm.calls[call_index]["messages"][-1]
    assert last.role == "tool"
    return [json.loads(r.content) for r in last.tool_results]


def test_multi_step_tool_use(run):
    result, llm = run([
        tool_calls(("get_order_status", {"order_id": ORDER_LATE})),
        tool_calls(("get_delivery_performance", {"order_id": ORDER_LATE}), ("get_order_items", {"order_id": ORDER_LATE})),
        answer("It was delivered 3 days late."),
    ])
    assert result.stop_reason == "final_answer"
    assert result.answer == "It was delivered 3 days late."
    assert result.iterations == 3
    assert (result.input_tokens, result.output_tokens) == (300, 60)

    assert tool_results_sent(llm, 1)[0]["status"] == "delivered"
    delivery, items = tool_results_sent(llm, 2)  # two parallel calls -> two results in one message
    assert delivery["days_vs_estimate"] == 3
    assert items["item_count"] == 2
    # The conversation grows: user, assistant, tool, assistant, tool
    assert [m.role for m in llm.calls[2]["messages"]] == ["user", "assistant", "tool", "assistant", "tool"]


def test_answer_without_tools(run):
    result, llm = run([answer("Please give me your order ID.")])
    assert result.stop_reason == "final_answer"
    assert result.iterations == 1


def test_unknown_tool_is_returned_to_model(run):
    result, llm = run([tool_calls(("refund_order", {"order_id": ORDER_LATE})), answer("I can't issue refunds.")])
    assert result.stop_reason == "final_answer"
    sent = llm.calls[1]["messages"][-1].tool_results[0]
    assert sent.is_error
    assert json.loads(sent.content)["error"]["code"] == "unknown_tool"


@pytest.mark.parametrize("bad_args", [{"order_id": "123"}, {"order_id": ORDER_LATE, "x": 1}, '{"order_id": "abc'])
def test_bad_arguments_are_returned_to_model(run, bad_args):
    result, llm = run([tool_calls(("get_order_status", bad_args)), answer("That ID looks invalid.")])
    assert result.stop_reason == "final_answer"
    assert tool_results_sent(llm, 1)[0]["error"]["code"] == "invalid_arguments"


def test_not_found_is_a_normal_tool_result(run):
    result, llm = run([tool_calls(("get_order_status", {"order_id": ORDER_UNKNOWN})), answer("No such order.")])
    assert tool_results_sent(llm, 1)[0]["error"]["code"] == "not_found"


def test_tool_exception_is_returned_not_raised(run, monkeypatch):
    def boom(conn, args):
        raise RuntimeError("tool bug")

    original = TOOLS["get_order_status"]
    monkeypatch.setitem(TOOLS, "get_order_status", Tool(original.name, original.description, original.input_schema, boom))
    result, llm = run([tool_calls(("get_order_status", {"order_id": ORDER_LATE})), answer("Something went wrong.")])
    assert result.stop_reason == "final_answer"
    err = tool_results_sent(llm, 1)[0]["error"]
    assert err["code"] == "tool_failed"
    assert "tool bug" in err["message"]


def test_max_iterations_stop(run):
    script = [tool_calls(("get_order_status", {"order_id": ORDER_LATE})) for _ in range(10)]
    result, llm = run(script, limits=Limits(max_iterations=3))
    assert result.stop_reason == "max_iterations"
    assert result.answer is None
    assert len(llm.calls) == 3


def test_token_budget_stop(run):
    script = [tool_calls(("get_order_status", {"order_id": ORDER_LATE}), usage=(9000, 1000)) for _ in range(5)]
    result, llm = run(script, limits=Limits(token_budget=15000, max_output_tokens=2048))
    assert result.stop_reason == "token_budget"
    assert len(llm.calls) == 2  # 10k after call 1 (continue), 20k after call 2 (stop)
    assert result.input_tokens + result.output_tokens == 20000
    assert result.answer is None


def test_max_tokens_clamped_to_remaining_budget(run):
    script = [tool_calls(("get_order_status", {"order_id": ORDER_LATE}), usage=(9500, 500)), answer("done", usage=(10, 10))]
    result, llm = run(script, limits=Limits(token_budget=10300, max_output_tokens=2048))
    assert llm.calls[0]["max_tokens"] == 2048
    assert llm.calls[1]["max_tokens"] == 300  # 10300 - 10000 left


def test_timeout_stop_after_slow_call(run):
    clock = FakeClock()

    def slow():
        clock.advance(61)
        return tool_calls(("get_order_status", {"order_id": ORDER_LATE}))

    result, llm = run([slow, answer("never reached")], limits=Limits(timeout_seconds=60), clock=clock)
    assert result.stop_reason == "timeout"
    assert len(llm.calls) == 1


def test_timeout_passed_to_llm_shrinks(run):
    clock = FakeClock()

    def step():
        clock.advance(20)
        return tool_calls(("get_order_status", {"order_id": ORDER_LATE}))

    result, llm = run([step, answer("ok")], limits=Limits(timeout_seconds=60), clock=clock)
    assert llm.calls[0]["timeout"] == 60
    assert llm.calls[1]["timeout"] == 40


def test_llm_timeout_error_reports_timeout(run):
    clock = FakeClock()

    def times_out():
        clock.advance(60)
        return TransientLLMError("request timed out")

    result, llm = run([times_out], limits=Limits(timeout_seconds=60), clock=clock)
    assert result.stop_reason == "timeout"
    assert len(llm.calls) == 1  # no retry past the deadline


def test_transient_error_is_retried(run):
    clock = FakeClock()
    result, llm = run([TransientLLMError("overloaded", 529), answer("ok")], clock=clock)
    assert result.stop_reason == "final_answer"
    assert len(llm.calls) == 2
    assert clock.slept == [1.0]


def test_permanent_error_is_not_retried(run):
    result, llm = run([PermanentLLMError("API error 404: model not found", 404), answer("unused")])
    assert result.stop_reason == "llm_error"
    assert "model not found" in result.error
    assert len(llm.calls) == 1


def test_truncated_output(run):
    result, _ = run([answer("partial ans", stop_reason="max_tokens")])
    assert result.stop_reason == "output_truncated"
    assert result.answer == "partial ans"


def test_truncated_tool_call_is_not_executed(run):
    result, llm = run([tool_calls(("get_order_status", {"order_id": ORDER_LATE}), stop_reason="max_tokens")])
    assert result.stop_reason == "output_truncated"
    assert len(llm.calls) == 1


def test_refusal(run):
    result, _ = run([answer("", stop_reason="refusal")])
    assert result.stop_reason == "refusal"


def test_cost_only_when_priced(run):
    prices = {"priced-model": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}}
    result, _ = run([answer("ok", usage=(1000, 100))], prices=prices, model="priced-model")
    assert result.estimated_cost_usd == pytest.approx(0.0015)
    result, _ = run([answer("ok", usage=(1000, 100))], prices=prices, model="other-model")
    assert result.estimated_cost_usd is None


def test_missing_usage_makes_cost_unknown(run):
    prices = {"m": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}}
    result, _ = run([tool_calls(("get_order_status", {"order_id": ORDER_LATE})), answer("ok", usage=None)], prices=prices, model="m")
    assert result.estimated_cost_usd is None


def test_limits_must_be_positive():
    with pytest.raises(ValueError):
        Limits(token_budget=0)
