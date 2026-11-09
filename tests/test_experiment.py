import csv
import json
import random

import pytest

from agent import experiment as ex
from agent.llm.base import PermanentLLMError
from agent.loop import Limits, run_agent
from agent.trace import Tracer
from conftest import ORDER_BIG, ORDER_CANCELED, ORDER_SHIPPED
from fakes import FakeLLM, answer, tool_calls

PRICES = {
    "claude-haiku-5-5": {"input_per_mtok": 0.10, "output_per_mtok": 0.50},
    "claude-sonnet-5-5": {"input_per_mtok": 2.00, "output_per_mtok": 10.00},
}
BUDGET = 20000

# What each fake run does, per question: the baseline tools, then an answer.
BASELINE_SCRIPT = {
    "q1": lambda: [tool_calls(("get_delivery_performance", {"order_id": ORDER_SHIPPED})), answer("No delivery date is recorded.")],
    "q2": lambda: [tool_calls(("get_order_payments", {"order_id": ORDER_BIG})), answer("12 payments; first 10 shown.")],
    "q3": lambda: [tool_calls(("get_order_status", {"order_id": ORDER_CANCELED}), ("get_order_payments", {"order_id": ORDER_CANCELED})),
                   answer("It was canceled.")],
}


def make_ask(conn, folder, script=None, calls=None):
    """An ask(spec) that runs the real loop against the fixture DB with a FakeLLM."""

    def ask(spec):
        if calls is not None:
            calls.append(spec.key)
        steps = (script or BASELINE_SCRIPT)[spec.question_id]()
        with Tracer(folder) as tracer:
            return run_agent(ex.QUESTIONS[spec.question_id], FakeLLM(steps, model=spec.model), conn, Limits(), tracer,
                             prices=PRICES, hints=spec.hints == "on")

    return ask


# --- plan ---

def test_plan_is_full_grid_shuffled_with_seed():
    plan = ex.build_plan(seed=7, repeats=5)
    assert len(plan) == 60
    assert len({s.key for s in plan}) == 60
    assert {(s.model, s.hints, s.question_id) for s in plan} == {
        (m, h, q) for m in ex.MODELS for h in ex.HINTS for q in ex.QUESTIONS
    }
    assert [s.key for s in ex.build_plan(7, 5)] == [s.key for s in plan]  # same seed, same order
    assert [s.key for s in ex.build_plan(8, 5)] != [s.key for s in plan]
    grid_order = sorted(plan, key=lambda s: (ex.MODELS.index(s.model), ex.HINTS.index(s.hints), s.question_id, s.repeat))
    assert plan != grid_order


def test_plan_file_must_match_on_resume(tmp_path):
    ex.load_or_create_plan(tmp_path, seed=1, repeats=2)
    saved = json.loads((tmp_path / ex.PLAN_FILE).read_text(encoding="utf-8"))
    assert saved["seed"] == 1 and len(saved["order"]) == 24 and "git" in saved
    ex.load_or_create_plan(tmp_path, seed=1, repeats=2)  # same settings: fine
    with pytest.raises(ValueError):
        ex.load_or_create_plan(tmp_path, seed=2, repeats=2)


# --- runner ---

def test_runner_completes_and_resume_skips_done(conn, tmp_path):
    plan = ex.build_plan(seed=1, repeats=1)
    calls = []
    ask = make_ask(conn, tmp_path, calls=calls)
    assert ex.run_plan(plan[:5], tmp_path, ask, PRICES, 1.0, BUDGET, log=lambda m: None) == "done"
    assert ex.run_plan(plan, tmp_path, ask, PRICES, 1.0, BUDGET, log=lambda m: None) == "done"
    assert len(calls) == 12 and len(set(calls)) == 12  # nothing ran twice
    entries = ex.read_manifest(tmp_path)
    assert len(ex.completed_entries(entries)) == 12
    assert all((tmp_path / e["trace_file"]).exists() for e in entries)


def test_hints_condition_reaches_the_trace(conn, tmp_path):
    plan = ex.build_plan(seed=1, repeats=1)
    ex.run_plan(plan, tmp_path, make_ask(conn, tmp_path), PRICES, 1.0, BUDGET, log=lambda m: None)
    for e in ex.read_manifest(tmp_path):
        start = ex.load_events(tmp_path / e["trace_file"])[0]
        assert start["hints"] == e["hints"] and start["model"] == e["model"]


def test_llm_error_runs_are_retried_on_resume(conn, tmp_path):
    plan = ex.build_plan(seed=1, repeats=1)[:3]
    failing = {k: (lambda: [PermanentLLMError("boom", 400)]) for k in ex.QUESTIONS}
    ex.run_plan(plan[:1], tmp_path, make_ask(conn, tmp_path, script=failing), PRICES, 1.0, BUDGET, log=lambda m: None)
    assert ex.completed_entries(ex.read_manifest(tmp_path)) == {}
    calls = []
    ex.run_plan(plan, tmp_path, make_ask(conn, tmp_path, calls=calls), PRICES, 1.0, BUDGET, log=lambda m: None)
    assert calls == [s.key for s in plan]  # the failed run was retried


def test_three_consecutive_api_errors_stop(conn, tmp_path):
    failing = {k: (lambda: [PermanentLLMError("boom", 400)]) for k in ex.QUESTIONS}
    calls = []
    outcome = ex.run_plan(ex.build_plan(1, 1), tmp_path, make_ask(conn, tmp_path, script=failing, calls=calls),
                          PRICES, 1.0, BUDGET, log=lambda m: None)
    assert outcome == "api_errors"
    assert len(calls) == 3


def test_spend_guard_never_lets_cost_pass_limit(conn, tmp_path):
    plan = ex.build_plan(seed=1, repeats=5)
    # Realistic-size runs: 9000 in / 1000 out per call, two calls (Sonnet run ~= $0.056).
    heavy = {q: (lambda: [tool_calls(("get_order_payments", {"order_id": ORDER_BIG}), usage=(9000, 1000)),
                          answer("ok", usage=(9000, 1000))]) for q in ex.QUESTIONS}
    calls = []
    outcome = ex.run_plan(plan, tmp_path, make_ask(conn, tmp_path, script=heavy, calls=calls), PRICES, 0.25, BUDGET, log=lambda m: None)
    assert outcome == "cost_limit"
    assert len(calls) < 60
    assert ex.spent(ex.read_manifest(tmp_path), PRICES, BUDGET) <= 0.25


def test_spend_counts_runs_without_cost_at_worst_case():
    entries = [{"model": "claude-sonnet-5-5", "estimated_cost_usd": None}, {"model": "claude-haiku-5-5", "estimated_cost_usd": 0.001}]
    assert ex.spent(entries, PRICES, BUDGET) == pytest.approx(0.20 + 0.001)


def test_baseline_costs_from_earlier_traces(conn, tmp_path):
    spec = ex.RunSpec("claude-haiku-5-5", "on", "q2", 1)
    result = make_ask(conn, tmp_path)(spec)
    costs = ex.baseline_costs([result.trace_path], PRICES)
    assert costs[("claude-haiku-5-5", "q2")] == [pytest.approx(result.estimated_cost_usd)]


# --- scoring sheet ---

@pytest.fixture
def finished(conn, tmp_path):
    """A finished 12-run experiment where one q1 run calls an extra tool."""
    folder = tmp_path / "experiment"
    plan = ex.build_plan(seed=3, repeats=1)
    extra_key = next(s.key for s in plan if s.question_id == "q1")
    base_ask = make_ask(conn, folder)

    def ask(spec):
        if spec.key == extra_key:
            steps = [tool_calls(("get_delivery_performance", {"order_id": ORDER_SHIPPED}), ("get_order_items", {"order_id": ORDER_SHIPPED})),
                     answer("Not delivered; 1 item.")]
            with Tracer(folder) as tracer:
                return run_agent(ex.QUESTIONS["q1"], FakeLLM(steps, model=spec.model), conn, Limits(), tracer,
                                 prices=PRICES, hints=spec.hints == "on")
        return base_ask(spec)

    folder.mkdir()
    ex.load_or_create_plan(folder, 3, 1)
    ex.run_plan(plan, folder, ask, PRICES, 1.0, BUDGET, log=lambda m: None)
    return folder, extra_key


def test_sheet_is_blind_and_key_maps_back(finished):
    folder, extra_key = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    assert len(sheet) == len(key) == 12
    assert all(list(row) == ex.SHEET_COLUMNS for row in sheet)
    text = json.dumps(sheet)
    for leak in [*ex.MODELS, "haiku", "sonnet", "hints", ".jsonl", "|r1"] + [e["trace_file"] for e in ex.read_manifest(folder)]:
        assert leak not in text.lower(), leak
    assert all(r["unsupported_claim"] == "" and r["which_claim"] == "" for r in sheet)

    manifest = {e["run_key"]: e for e in ex.read_manifest(folder)}
    sheet_by_id = {r["row_id"]: r for r in sheet}
    for k in key:
        run = manifest[k["run_key"]]
        assert (k["model"], k["hints"], k["question_id"], k["repeat"]) == (run["model"], run["hints"], run["question_id"], run["repeat"])
        end = ex.run_end(ex.load_events(folder / run["trace_file"]))
        assert sheet_by_id[k["row_id"]]["answer"] == end["final_answer"]
        assert sheet_by_id[k["row_id"]]["question_id"] == run["question_id"]
        flagged = "EXTRA TOOL: check trace" in sheet_by_id[k["row_id"]]["flag"]
        assert flagged == (k["run_key"] == extra_key)


def test_sheet_rows_are_shuffled(finished):
    folder, _ = finished
    orders = set()
    for seed in range(5):
        sheet, key = ex.build_sheet(folder, random.Random(seed))
        by_id = {k["row_id"]: k for k in key}
        orders.add(tuple(by_id[r["row_id"]]["question_id"] + by_id[r["row_id"]]["model"] + by_id[r["row_id"]]["hints"] for r in sheet))
    assert len(orders) > 1  # different shuffles give different row orders


def test_write_sheet_refuses_to_overwrite(finished, tmp_path):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    out = tmp_path / "scoring"
    ex.write_sheet(out, sheet, key)
    assert [r["row_id"] for r in ex.read_csv(out / "sheet.csv")] == [r["row_id"] for r in sheet]
    with pytest.raises(FileExistsError):
        ex.write_sheet(out, sheet, key)
    ex.write_sheet(out, sheet, key, force=True)


# --- analysis ---

def test_analysis_handles_unclear_blank_and_invalid(finished, tmp_path):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    values = ["yes", "no", "unclear", "", "  YES ", "maybe"]
    for i, row in enumerate(sheet):
        row["unsupported_claim"] = values[i % len(values)]
    ex.write_sheet(tmp_path / "s", sheet, key)
    lines = ex.analyze(ex.read_csv(tmp_path / "s" / "sheet.csv"), ex.read_csv(tmp_path / "s" / "key.csv"))
    report = "\n".join(lines)
    assert "WARNING" in report and "invalid" in report  # "maybe" is reported, not silently merged
    cell_lines = [l for l in lines if l.startswith("claude-") and len(l.split()) == 11]  # main per-cell table
    assert len(cell_lines) == 12  # one per model x hints x question
    assert "%" not in report


def test_analysis_counts_scores(finished, tmp_path):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    for row in sheet:
        row["unsupported_claim"] = "unclear"
    lines = ex.analyze(sheet, key)
    start = lines.index("Totals per model x hints (counts out of N)") + 2  # skip the column header
    totals = [l.split() for l in lines[start:start + 4]]
    # Columns: model hints N yes no unclear blank invalid. N=3 each (repeats=1, 3 questions), all unclear.
    assert sorted((t[0], t[1], t[2], t[5]) for t in totals) == sorted((m, h, "3", "3") for m in ex.MODELS for h in ex.HINTS)


def test_analysis_splits_scores_by_hint_exposure(finished):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    qid_by_row = {k["row_id"]: k["question_id"] for k in key}
    for row in sheet:  # score exposed questions (q1, q2) "yes" and q3 "no"
        row["unsupported_claim"] = "no" if qid_by_row[row["row_id"]] == "q3" else "yes"
    lines = ex.analyze(sheet, key)
    start = next(i for i, l in enumerate(lines) if l.startswith("Split by hint exposure")) + 2
    split = [l.split() for l in lines[start:start + 24]]  # 12 cells x exposed yes/no
    # Columns: model hints q exposed N yes no unclear blank invalid
    for model, hints, qid, exposed, n, yes, no, *_ in split:
        if qid in ("q1", "q2"):  # fake q1/q2 runs always get a hint field
            assert (exposed, n, yes, no) == ("yes", "1", "1", "0") or (exposed, n) == ("no", "0")
        else:  # fake q3 runs never do
            assert (exposed, n, yes, no) == ("no", "1", "0", "1") or (exposed, n) == ("yes", "0")


def test_rescore_sheet(finished):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    verdicts = ["yes", "unclear", "no", "no", "no", "no", "no", "no", "no", "YES ", "no", "no"]
    for row, v in zip(sheet, verdicts):
        row["unsupported_claim"], row["which_claim"] = v, "some claim"
    flagged = {r["row_id"] for r in sheet if r["unsupported_claim"].strip().lower() in ("yes", "unclear")}

    rescore, rescore_key = ex.build_rescore(sheet, key, n_no=4, rng=random.Random(1))
    assert len(rescore) == len(rescore_key) == 3 + 4
    assert all(list(r) == ex.SHEET_COLUMNS for r in rescore)
    assert all(r["unsupported_claim"] == "" and r["which_claim"] == "" for r in rescore)  # verdicts cleared
    new_ids = [r["row_id"] for r in rescore]
    assert len(set(new_ids)) == 7 and not set(new_ids) & {r["row_id"] for r in sheet}
    originals = {k["original_row_id"] for k in rescore_key}
    assert flagged <= originals  # every yes/unclear row is included
    original_by_id = {r["row_id"]: r for r in sheet}
    assert sum(original_by_id[o]["unsupported_claim"] == "no" for o in originals) == 4
    # The key maps back to the same run, and the answer is the same text.
    key_by_id = {k["row_id"]: k for k in key}
    rescore_by_id = {r["row_id"]: r for r in rescore}
    for k in rescore_key:
        orig = key_by_id[k["original_row_id"]]
        assert (k["run_key"], k["model"], k["hints"], k["trace_file"]) == (orig["run_key"], orig["model"], orig["hints"], orig["trace_file"])
        assert rescore_by_id[k["row_id"]]["answer"] == original_by_id[k["original_row_id"]]["answer"]
    text = json.dumps(rescore).lower()
    assert not any(leak in text for leak in ("haiku", "sonnet", "hints", ".jsonl"))


def test_reads_sheet_resaved_by_excel(finished, tmp_path):
    """Excel on Windows can re-save the sheet as Windows-1252, tab-delimited text."""
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    sheet[0]["answer"] = "Delivered 2018–01–15 in São Paulo"  # en dash and ã exist in cp1252
    for row in sheet:
        row["unsupported_claim"] = "no"
    path = tmp_path / "sheet.csv"
    with open(path, "w", encoding="cp1252", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ex.SHEET_COLUMNS, delimiter="\t")
        writer.writeheader()
        writer.writerows(sheet)
    assert ex.csv_format(path) == ("cp1252", "\t")
    rows = ex.read_csv(path)
    assert [r["row_id"] for r in rows] == [r["row_id"] for r in sheet]
    assert rows[0]["answer"] == "Delivered 2018–01–15 in São Paulo"
    assert "WARNING" not in "\n".join(ex.analyze(rows, key))


def test_reads_sheet_as_written(finished, tmp_path):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    ex.write_sheet(tmp_path, sheet, key)
    assert ex.csv_format(tmp_path / "sheet.csv") == ("utf-8", ",")


def test_duplicate_count():
    assert ex.duplicate_count(["a", "a", "b", "c", "c", "c"]) == 5
    assert ex.duplicate_count(["a", "b"]) == 0
    assert ex.duplicate_count([None, None, "a"]) == 0


def test_analysis_reports_duplicates(finished):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    lines = ex.analyze(sheet, key)
    # repeats=1: one answer per cell, so no duplicates anywhere. Columns end with dup_answers, hint_exposed.
    cell_lines = [l.split() for l in lines if l.startswith("claude-") and len(l.split()) == 11]
    assert len(cell_lines) == 12 and all(c[-2] == "0" for c in cell_lines)


def test_hint_exposed_uses_raw_result_for_both_conditions(conn, tmp_path):
    for hints in ("on", "off"):
        spec = ex.RunSpec("claude-haiku-5-5", hints, "q2", 1)  # payments on ORDER_BIG: raw result has `truncated`
        events = ex.load_events(make_ask(conn, tmp_path)(spec).trace_path)
        assert ex.hint_exposed(events) is True
    spec = ex.RunSpec("claude-haiku-5-5", "off", "q3", 1)  # status + payments on a canceled order: no hint fields
    assert ex.hint_exposed(ex.load_events(make_ask(conn, tmp_path)(spec).trace_path)) is False


def test_analysis_reports_hint_exposure(finished):
    folder, _ = finished
    sheet, key = ex.build_sheet(folder, random.Random(0))
    lines = ex.analyze(sheet, key)
    cells = {(c[0], c[1], c[2]): c for c in (l.split() for l in lines) if c and c[0].startswith("claude-") and len(c) == 11}
    # Fake runs: q1 gets the delivery `note`, q2 the payments `truncated`, q3 neither -- in both hint conditions.
    for (model, hints, qid), c in cells.items():
        assert c[-1] == {"q1": "1", "q2": "1", "q3": "0"}[qid], (model, hints, qid)
