"""The hints experiment: run grid, resumable runner, blind scoring sheet, and raw-count analysis.

The logic lives here as plain functions so it can be tested offline. The scripts in scripts/
(run_experiment.py, make_scoring_sheet.py, analyze_experiment.py) are thin command-line wrappers.

Files in an experiment folder (traces/experiment_<date>/):
- plan.json       seed, grid and shuffled run order (written once; a resume must match it)
- manifest.jsonl  one line per finished run: run key -> trace file, stop reason, tokens, cost
- *.jsonl         one trace per run, as written by agent.trace.Tracer
"""

import csv
import json
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from agent.config import ROOT
from agent.tools import apply_hints

MODELS = ("claude-haiku-5-5", "claude-sonnet-5-5")
HINTS = ("on", "off")
QUESTIONS = {
    "q1": "Order 2d858f451373b04fb5c984a1cc2defaf is marked delivered. On what date did it arrive?",
    "q2": "How was order fa65dad1b0e818e3ccc5cb0e39231352 paid? List all the payments.",
    "q3": "What happened to order e5fa5a7210941f7d56d0208e4e071d35, and how was it paid?",
}
# Tools each question used across the earlier (baseline) runs. Any other tool is flagged EXTRA TOOL.
BASELINE_TOOLS = {
    "q1": {"get_delivery_performance"},
    "q2": {"get_order_payments"},
    "q3": {"get_order_status", "get_order_payments", "get_delivery_performance"},
}
DEFAULT_SEED = 20261109
DEFAULT_REPEATS = 5
MAX_CONSECUTIVE_ERRORS = 3

PLAN_FILE = "plan.json"
MANIFEST_FILE = "manifest.jsonl"
SHEET_COLUMNS = ["row_id", "question_id", "answer", "tools_called", "flag", "unsupported_claim", "which_claim"]
KEY_COLUMNS = ["row_id", "run_key", "model", "hints", "question_id", "repeat", "trace_file"]
SCORES = ("yes", "no", "unclear", "blank", "invalid")


@dataclass(frozen=True)
class RunSpec:
    model: str
    hints: str
    question_id: str
    repeat: int

    @property
    def key(self) -> str:
        return f"{self.model}|hints={self.hints}|{self.question_id}|r{self.repeat}"


# --- plan --------------------------------------------------------------------


def build_plan(seed: int, repeats: int) -> list[RunSpec]:
    """Every model x hints x question x repeat, shuffled with a fixed seed."""
    runs = [RunSpec(m, h, q, r) for m in MODELS for h in HINTS for q in QUESTIONS for r in range(1, repeats + 1)]
    random.Random(seed).shuffle(runs)
    return runs


def _git_state() -> dict:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit": commit, "uncommitted_changes": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "uncommitted_changes": None}


def load_or_create_plan(folder: Path, seed: int, repeats: int) -> list[RunSpec]:
    """Create plan.json, or check that an existing one matches (so a resume runs the same experiment)."""
    plan = build_plan(seed, repeats)
    wanted = {"seed": seed, "repeats": repeats, "models": list(MODELS), "hints": list(HINTS), "questions": QUESTIONS}
    path = folder / PLAN_FILE
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if {k: saved[k] for k in wanted} != wanted or saved["order"] != [s.key for s in plan]:
            raise ValueError(f"{path} was created with different settings; use a new --out folder.")
        return plan
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({**wanted, "order": [s.key for s in plan], "git": _git_state()}, indent=2), encoding="utf-8")
    return plan


# --- manifest ----------------------------------------------------------------


def read_manifest(folder: Path) -> list[dict]:
    path = folder / MANIFEST_FILE
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_manifest(folder: Path, entry: dict) -> None:
    with open(folder / MANIFEST_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def completed_entries(entries: list[dict]) -> dict[str, dict]:
    """Runs that produced a result. A run that ended in llm_error is not complete and is retried."""
    return {e["run_key"]: e for e in entries if e["stop_reason"] != "llm_error"}


# --- cost --------------------------------------------------------------------


def worst_case_run_cost(model: str, prices: dict, token_budget: int) -> float:
    """The whole token budget at the model's dearer rate (input or output)."""
    p = prices[model]
    return token_budget / 1_000_000 * max(p["input_per_mtok"], p["output_per_mtok"])


def spent(entries: list[dict], prices: dict, token_budget: int) -> float:
    """Estimated cost of all recorded runs. A run without a cost estimate counts at its worst case."""
    return sum(
        e["estimated_cost_usd"] if e["estimated_cost_usd"] is not None else worst_case_run_cost(e["model"], prices, token_budget)
        for e in entries
    )


def baseline_costs(trace_paths: list[Path], prices: dict) -> dict[tuple[str, str], list[float]]:
    """Estimated cost of earlier runs of the experiment questions, keyed by (model, question_id)."""
    by_text = {text: qid for qid, text in QUESTIONS.items()}
    costs = defaultdict(list)
    for path in trace_paths:
        events = load_events(path)
        if not events or events[0].get("event") != "run_start" or events[-1].get("event") != "run_end":
            continue
        qid, model = by_text.get(events[0]["message"]["text"]), events[0]["model"]
        if qid and model in prices:
            usage, p = events[-1]["usage"], prices[model]
            costs[(model, qid)].append(
                usage["input_tokens"] / 1e6 * p["input_per_mtok"] + usage["output_tokens"] / 1e6 * p["output_per_mtok"]
            )
    return costs


# --- runner ------------------------------------------------------------------


def run_plan(
    plan: list[RunSpec],
    folder: Path,
    ask: Callable,  # ask(spec) -> agent.loop.RunResult
    prices: dict,
    max_cost: float,
    token_budget: int,
    log: Callable[[str], None] = print,
) -> str:
    """Run every not-yet-completed run in plan order. Returns why it stopped: done, cost_limit or api_errors."""
    entries = read_manifest(folder)
    done = completed_entries(entries)
    total = spent(entries, prices, token_budget)
    consecutive_errors = 0
    todo = [s for s in plan if s.key not in done]
    for n, spec in enumerate(todo, 1):
        worst = worst_case_run_cost(spec.model, prices, token_budget)
        if total + worst > max_cost:
            log(f"Spend guard: ${total:.4f} spent + ${worst:.4f} worst case for the next run would pass ${max_cost:.2f}. Stopping.")
            return "cost_limit"
        result = ask(spec)
        entry = {
            "run_key": spec.key,
            "model": spec.model,
            "hints": spec.hints,
            "question_id": spec.question_id,
            "repeat": spec.repeat,
            "trace_file": result.trace_path.name,
            "stop_reason": result.stop_reason,
            "iterations": result.iterations,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "estimated_cost_usd": result.estimated_cost_usd,
            "error": result.error,
        }
        append_manifest(folder, entry)
        total += spent([entry], prices, token_budget)
        log(f"[{n}/{len(todo)}] {spec.key} stop={result.stop_reason} tokens={result.input_tokens}+{result.output_tokens} spent=${total:.4f}")
        if result.stop_reason == "llm_error":
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                log(f"{MAX_CONSECUTIVE_ERRORS} API errors in a row (last: {result.error}). Stopping; rerun to resume.")
                return "api_errors"
        else:
            consecutive_errors = 0
    return "done"


# --- traces ------------------------------------------------------------------


def load_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def tools_called(events: list[dict]) -> list[str]:
    return [e["tool"] for e in events if e["event"] == "tool_call"]


def extra_tools(question_id: str, tools: list[str]) -> list[str]:
    return sorted(set(tools) - BASELINE_TOOLS[question_id])


def hint_exposed(events: list[dict]) -> bool:
    """Did any tool's raw result contain a field that --hints off removes? Checked on the raw result,
    so it means the same thing for hints=on and hints=off runs."""
    return any(
        apply_hints(e["tool"], e["result"], hints=False) != e["result"] for e in events if e["event"] == "tool_call"
    )


def run_end(events: list[dict]) -> dict:
    return next(e for e in events if e["event"] == "run_end")


# --- blind scoring sheet -----------------------------------------------------


def build_sheet(folder: Path, rng: random.Random) -> tuple[list[dict], list[dict]]:
    """Sheet rows (shuffled, no condition information) and key rows (row_id -> condition)."""
    runs = list(completed_entries(read_manifest(folder)).values())
    row_ids = rng.sample(range(1000, 10000), len(runs))
    sheet, key = [], []
    for row_id, entry in zip(row_ids, runs):
        trace = folder / entry["trace_file"]
        events = load_events(trace)
        end = run_end(events)
        tools = tools_called(events)
        flags = []
        if extra_tools(entry["question_id"], tools):
            flags.append("EXTRA TOOL: check trace")
        if end["stop_reason"] != "final_answer":
            flags.append(f"NO FINAL ANSWER ({end['stop_reason']})")
        sheet.append({
            "row_id": f"R{row_id}",
            "question_id": entry["question_id"],
            "answer": end["final_answer"] or "",
            "tools_called": "; ".join(tools),
            "flag": "; ".join(flags),
            "unsupported_claim": "",
            "which_claim": "",
        })
        key.append({
            "row_id": f"R{row_id}",
            "run_key": entry["run_key"],
            "model": entry["model"],
            "hints": entry["hints"],
            "question_id": entry["question_id"],
            "repeat": entry["repeat"],
            "trace_file": trace.resolve().relative_to(ROOT).as_posix() if trace.resolve().is_relative_to(ROOT) else str(trace.resolve()),
        })
    rng.shuffle(sheet)
    key.sort(key=lambda r: r["row_id"])
    return sheet, key


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    # utf-8-sig so Excel on Windows shows accented characters correctly.
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_sheet(out_dir: Path, sheet: list[dict], key: list[dict], force: bool = False) -> None:
    """Write sheet.csv and key.csv. Refuses to overwrite, so a rerun cannot scramble a sheet being scored."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("sheet.csv", "key.csv"):
        if (out_dir / name).exists() and not force:
            raise FileExistsError(f"{out_dir / name} already exists; pass --force to overwrite.")
    write_csv(out_dir / "sheet.csv", SHEET_COLUMNS, sheet)
    write_csv(out_dir / "key.csv", KEY_COLUMNS, key)


def read_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


# --- analysis ----------------------------------------------------------------


def normalize_score(value: str | None) -> str:
    value = (value or "").strip().lower()
    if value in ("yes", "no", "unclear"):
        return value
    return "blank" if value == "" else "invalid"


def duplicate_count(answers: list[str | None]) -> int:
    """How many answers are exact string duplicates of another answer in the same list (missing answers ignored)."""
    counts = Counter(a for a in answers if a is not None)
    return sum(n for n in counts.values() if n > 1)


def analyze(sheet: list[dict], key: list[dict], root: Path = ROOT) -> list[str]:
    """Raw counts only: no percentages, no tests, no conclusions."""
    lines = []
    key_by_id = {r["row_id"]: r for r in key}
    sheet_by_id = {r["row_id"]: r for r in sheet}
    missing = sorted(set(key_by_id) ^ set(sheet_by_id))
    if missing:
        lines.append(f"WARNING: row_ids not in both files (left out): {missing}")

    rows = []
    for row_id in sorted(set(key_by_id) & set(sheet_by_id)):
        k = key_by_id[row_id]
        events = load_events(root / k["trace_file"])
        rows.append({
            "row_id": row_id,
            "model": k["model"],
            "hints": k["hints"],
            "question_id": k["question_id"],
            "score": normalize_score(sheet_by_id[row_id]["unsupported_claim"]),
            "end": run_end(events),
            "extra_tool": bool(extra_tools(k["question_id"], tools_called(events))),
            "hint_exposed": hint_exposed(events),
        })

    invalid = [r["row_id"] for r in rows if r["score"] == "invalid"]
    if invalid:
        lines.append(f"WARNING: unsupported_claim values other than yes/no/unclear/blank in rows: {invalid}")

    cells, totals = defaultdict(list), defaultdict(list)
    for r in rows:
        cells[(r["model"], r["hints"], r["question_id"])].append(r)
        totals[(r["model"], r["hints"])].append(r)

    header = (f"{'model':<20}{'hints':<7}{'q':<4}{'N':>3}" + "".join(f"{s:>9}" for s in SCORES)
              + f"{'dup_answers':>13}{'hint_exposed':>14}")
    lines += ["", "Unsupported claims per model x hints x question (counts out of N)", header]
    for (model, hints, qid), cell in sorted(cells.items()):
        counts = Counter(r["score"] for r in cell)
        dups = duplicate_count([r["end"]["final_answer"] for r in cell])
        exposed = sum(r["hint_exposed"] for r in cell)
        lines.append(f"{model:<20}{hints:<7}{qid:<4}{len(cell):>3}" + "".join(f"{counts[s]:>9}" for s in SCORES)
                     + f"{dups:>13}{exposed:>14}")

    lines += ["", "Totals per model x hints (counts out of N)", f"{'model':<20}{'hints':<7}{'N':>7}" + "".join(f"{s:>9}" for s in SCORES)]
    for (model, hints), group in sorted(totals.items()):
        counts = Counter(r["score"] for r in group)
        lines.append(f"{model:<20}{hints:<7}{len(group):>7}" + "".join(f"{counts[s]:>9}" for s in SCORES))

    lines += ["", "Runs per model x hints (from the traces)",
              f"{'model':<20}{'hints':<7}{'runs':>6}{'avg_iter':>10}{'extra_tool':>12}{'tokens_in':>11}{'tokens_out':>12}{'est_cost_usd':>14}"]
    for (model, hints), group in sorted(totals.items()):
        ends = [r["end"] for r in group]
        extra = sum(r["extra_tool"] for r in group)
        costs = [e["estimated_cost_usd"] for e in ends]
        cost = f"{sum(costs):.4f}" if all(c is not None for c in costs) else "n/a"
        lines.append(
            f"{model:<20}{hints:<7}{len(group):>6}{sum(e['iterations'] for e in ends) / len(ends):>10.2f}{extra:>12}"
            f"{sum(e['usage']['input_tokens'] for e in ends):>11}{sum(e['usage']['output_tokens'] for e in ends):>12}{cost:>14}"
        )
    lines += ["", "dup_answers = answers in the cell that are exact string duplicates of another answer in the same cell.",
              "hint_exposed = runs that called a tool whose raw result contained a field --hints off removes "
              "(for hints=off runs the model did not see that field)."]
    return lines
