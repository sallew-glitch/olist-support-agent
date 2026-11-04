import importlib.util
import json

from agent.config import ROOT
from agent.loop import Limits, run_agent
from agent.trace import Tracer
from conftest import ORDER_LATE
from fakes import FakeLLM, answer, tool_calls

spec = importlib.util.spec_from_file_location("summarize_traces", ROOT / "scripts" / "summarize_traces.py")
summarize_traces = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summarize_traces)


def test_summary_groups_by_model_and_reports_only_trace_data(conn, tmp_path):
    prices = {"model-a": {"input_per_mtok": 1.0, "output_per_mtok": 1.0}}
    scripts = [
        ("model-a", [tool_calls(("get_order_status", {"order_id": ORDER_LATE})), answer("ok")]),
        ("model-a", [tool_calls(("bogus_tool", {})), answer("ok")]),
        ("model-b", [answer("ok", usage=(50, 5))]),
    ]
    for model, script in scripts:
        with Tracer(tmp_path) as tracer:
            run_agent("q", FakeLLM(script, model=model), conn, Limits(), tracer, prices=prices)
    # An incomplete trace (no run_end), as if the process was killed mid-run.
    (tmp_path / "zz_incomplete.jsonl").write_text(
        json.dumps({"event": "run_start", "provider": "fake", "model": "model-b"}) + "\n", encoding="utf-8"
    )

    runs, bad = summarize_traces.load_runs(tmp_path)
    groups = summarize_traces.summarize(runs)

    a, b = groups["fake/model-a"], groups["fake/model-b"]
    assert a["runs"] == 2 and a["incomplete"] == 0
    assert a["iterations"] == [2, 2]
    assert a["tool_calls"] == {"get_order_status": 1, "bogus_tool": 1}
    assert a["tool_errors"] == {"unknown_tool": 1}
    assert (a["input_tokens"], a["output_tokens"]) == (400, 80)
    assert a["runs_with_cost"] == 2

    assert b["runs"] == 2 and b["incomplete"] == 1
    assert b["runs_with_cost"] == 0  # model-b has no price: no estimate
    assert (b["input_tokens"], b["output_tokens"]) == (50, 5)

    summarize_traces.print_report(groups)  # smoke test: must not crash
