import json

from agent.loop import Limits, run_agent
from agent.trace import Tracer
from conftest import ORDER_LATE
from fakes import FakeClock, FakeLLM, answer, tool_calls

FAKE_KEY = "sk-ant-test-THIS-MUST-NOT-APPEAR"


def read_events(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_trace_format(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    llm = FakeLLM([
        tool_calls(("get_order_status", {"order_id": ORDER_LATE}), ("nope", {})),
        answer("Delivered."),
    ])
    clock = FakeClock()
    prices = {"fake-model": {"input_per_mtok": 1.0, "output_per_mtok": 5.0}}
    with Tracer(tmp_path) as tracer:
        result = run_agent("Where is my order?", llm, conn, Limits(), tracer, prices=prices, clock=clock, sleep=clock.sleep)

    events = read_events(result.trace_path)
    assert [e["event"] for e in events] == ["run_start", "llm_call", "tool_call", "tool_call", "llm_call", "run_end"]
    assert len({e["run_id"] for e in events}) == 1
    assert all("timestamp" in e for e in events)

    start = events[0]
    assert start["provider"] == "fake" and start["model"] == "fake-model"
    assert start["message"] == {"role": "user", "text": "Where is my order?"}
    assert start["limits"]["max_iterations"] == 8
    assert len(start["tools"]) == 5

    call = events[1]
    assert call["iteration"] == 1
    assert call["usage"] == {"input_tokens": 100, "output_tokens": 20}
    assert call["estimated_cost_usd"] > 0
    for key in ("latency_ms", "attempts", "max_tokens", "stop_reason"):
        assert key in call
    assert call["message"]["tool_calls"][0]["arguments"] == {"order_id": ORDER_LATE}

    ok, bad = events[2], events[3]
    assert ok["tool"] == "get_order_status" and ok["arguments"] == {"order_id": ORDER_LATE}
    assert ok["is_error"] is False and ok["result_chars"] > 0 and "latency_ms" in ok
    assert bad["is_error"] is True and bad["error_code"] == "unknown_tool"

    end = events[-1]
    assert end["stop_reason"] == "final_answer"
    assert end["final_answer"] == "Delivered."
    assert end["iterations"] == 2
    assert end["usage"] == {"input_tokens": 200, "output_tokens": 40}
    assert end["estimated_cost_usd"] == call["estimated_cost_usd"] * 2
    assert "duration_ms" in end


def test_trace_never_contains_api_key(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", FAKE_KEY)
    llm = FakeLLM([answer("hi")])
    with Tracer(tmp_path) as tracer:
        result = run_agent("hello", llm, conn, Limits(), tracer)
    assert FAKE_KEY not in result.trace_path.read_text(encoding="utf-8")


def test_trace_without_prices_has_null_cost(conn, tmp_path):
    with Tracer(tmp_path) as tracer:
        result = run_agent("hello", FakeLLM([answer("hi")]), conn, Limits(), tracer)
    end = read_events(result.trace_path)[-1]
    assert end["estimated_cost_usd"] is None


def test_each_run_gets_its_own_file(conn, tmp_path):
    paths = set()
    for _ in range(2):
        with Tracer(tmp_path) as tracer:
            paths.add(run_agent("hello", FakeLLM([answer("hi")]), conn, Limits(), tracer).trace_path)
    assert len(paths) == 2
