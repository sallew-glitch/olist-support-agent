"""Offline tests for the chart's data extraction (fixture traces only; the real traces are gitignored)."""

import importlib.util
import json
import random

import pytest

from agent import experiment as ex
from agent.config import ROOT
from agent.loop import Limits, run_agent
from agent.tools import DELIVERY_NOTE
from agent.trace import Tracer
from conftest import ORDER_CANCELED, ORDER_LATE, ORDER_SHIPPED
from fakes import FakeLLM, answer, tool_calls

spec = importlib.util.spec_from_file_location("make_visual", ROOT / "scripts" / "make_visual.py")
mv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mv)  # must not need matplotlib: it is imported only when rendering

# Per (model, hints): for repeats 1..5, (calls the delivery tool?, answer says "never"?)
DESIGN = {
    ("claude-sonnet-5-5", "on"): [(True, False)] * 5,
    ("claude-sonnet-5-5", "off"): [(True, True)] * 4 + [(True, False)],
    ("claude-haiku-5-5", "on"): [(True, True), (False, False), (False, False), (True, True), (False, False)],
    ("claude-haiku-5-5", "off"): [(False, False), (False, False), (False, True), (True, True), (True, False)],
}


def steps(delivery: bool, never: bool):
    calls = [("get_order_status", {"order_id": ORDER_CANCELED}), ("get_order_payments", {"order_id": ORDER_CANCELED})]
    if delivery:  # the canceled order has no delivery date, so the raw result carries the note
        calls.append(("get_delivery_performance", {"order_id": ORDER_CANCELED}))
    text = "It was canceled and NEVER shipped." if never else "It was canceled. Nevertheless, payment was 1 installment."
    return [tool_calls(*calls), answer(text)]


@pytest.fixture
def experiment_folder(conn, tmp_path):
    folder = tmp_path / "experiment"
    folder.mkdir()
    plan = [ex.RunSpec(m, h, "q3", r) for (m, h) in DESIGN for r in range(1, 6)]
    plan += [ex.RunSpec(m, h, "q1", 1) for (m, h) in DESIGN]  # other questions must be ignored
    random.Random(0).shuffle(plan)

    def ask(spec):
        if spec.question_id == "q3":
            script = steps(*DESIGN[(spec.model, spec.hints)][spec.repeat - 1])
        else:
            script = [tool_calls(("get_delivery_performance", {"order_id": ORDER_SHIPPED})), answer("never delivered")]
        with Tracer(folder) as tracer:
            return run_agent(ex.QUESTIONS[spec.question_id], FakeLLM(script, model=spec.model), conn, Limits(), tracer,
                             hints=spec.hints == "on")

    prices = {m: {"input_per_mtok": 1.0, "output_per_mtok": 1.0} for m, _ in DESIGN}
    ex.run_plan(plan, folder, ask, prices, 1.0, 20000, log=lambda m: None)
    assert len(ex.completed_entries(ex.read_manifest(folder))) == len(plan)
    return folder


def test_extract_counts_and_per_run_exposure(experiment_folder):
    cells = mv.extract(experiment_folder)
    assert set(cells) == set(mv.ROW_ORDER) == set(DESIGN)
    for cell, design in DESIGN.items():
        runs = cells[cell]
        assert [r.repeat for r in runs] == [1, 2, 3, 4, 5]  # circles in repeat order
        assert [r.says_never for r in runs] == [never for _, never in design]
        assert [r.called_delivery for r in runs] == [delivery for delivery, _ in design]
        assert [r.exposed for r in runs] == [delivery for delivery, _ in design]  # note only via the delivery tool
    assert {cell: sum(r.says_never for r in runs) for cell, runs in cells.items()} == {
        ("claude-sonnet-5-5", "on"): 0, ("claude-sonnet-5-5", "off"): 4,
        ("claude-haiku-5-5", "on"): 2, ("claude-haiku-5-5", "off"): 2,
    }


def test_chart_counts_match_the_analysis_counts(experiment_folder):
    sheet, key = ex.build_sheet(experiment_folder, random.Random(1))
    cells = mv.extract(experiment_folder)
    expected = mv.analysis_never_counts(sheet, key)
    assert expected == {cell: (sum(r.says_never for r in runs), len(runs)) for cell, runs in cells.items()}
    # FakeLLM sends no thinking effort, so check() must refuse to draw; nothing else should be wrong.
    assert mv.check(cells, expected) == ["expected one recorded thinking effort across runs, found {None}"]


def test_check_reports_a_mismatch(experiment_folder):
    cells = mv.extract(experiment_folder)
    wrong = {cell: (sum(r.says_never for r in runs), len(runs)) for cell, runs in cells.items()}
    wrong[("claude-haiku-5-5", "on")] = (3, 5)
    assert any("chart counts (2, 5) != analysis counts (3, 5)" in p for p in mv.check(cells, wrong))


def run(repeat, delivery, never=False):
    return mv.Run(repeat=repeat, says_never=never, exposed=delivery, called_delivery=delivery, effort="low")


def test_texts_are_computed_and_note_comes_from_the_tool_constant(tmp_path):
    (tmp_path / ex.PLAN_FILE).write_text(json.dumps({"git": {"commit": "0123456789abcdef"}}), encoding="utf-8")
    cells = {cell: [run(i, delivery=i <= 2) for i in range(1, 6)] for cell in mv.ROW_ORDER}
    t = mv.texts(cells, tmp_path)
    assert t["caption"] == f"The note: “{DELIVERY_NOTE}”"
    assert "5 runs per cell." in t["subtitle"]
    assert "Claude Haiku 5.5 and Sonnet 5.5, low thinking effort." in t["fine_print"]
    assert "Haiku called the delivery tool in 2 of 5 runs per condition." in t["fine_print"]
    assert "commit 0123456" in t["provenance"]
    assert "“never”" in t["title"] and '"' not in "".join(t.values())  # curly quotes only

    cells[("claude-haiku-5-5", "off")] = [run(i, delivery=i <= 3) for i in range(1, 6)]
    assert "in 2 of 5 runs with the note on and 3 of 5 with it off." in mv.texts(cells, tmp_path)["fine_print"]


def test_model_label():
    assert mv.model_label("claude-sonnet-5-5") == "Sonnet 5.5"
    assert mv.model_label("claude-haiku-5-5") == "Haiku 5.5"
