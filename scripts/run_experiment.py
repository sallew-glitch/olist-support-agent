"""Run the hints experiment: 2 models x 2 hint conditions x 3 questions x N repeats (default 60 runs).

Usage:
    python scripts/run_experiment.py                  # preview only: run count and cost estimate, no API calls
    python scripts/run_experiment.py --yes            # run; rerunning resumes and skips completed runs
    python scripts/run_experiment.py --out traces/experiment_2026-11-09 --yes   # resume a specific folder

Run order is shuffled with a fixed seed (saved in plan.json). A spend guard stops before any run that
could push the estimated cumulative cost past --max-cost. 3 API errors in a row also stop the run.
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from statistics import mean

from agent.cli import Session
from agent.config import ROOT, load_dotenv, load_settings
from agent.cost import load_prices
from agent.experiment import (
    DEFAULT_REPEATS, DEFAULT_SEED, MODELS, PLAN_FILE, QUESTIONS, baseline_costs, build_plan, completed_entries,
    load_or_create_plan, read_manifest, run_plan, spent, worst_case_run_cost,
)
from agent.llm import LLMError


def other_unfinished_experiments(out: Path) -> list[Path]:
    unfinished = []
    for plan_path in (ROOT / "traces").glob(f"experiment_*/{PLAN_FILE}"):
        folder = plan_path.parent
        if folder.resolve() == out.resolve():
            continue
        order = json.loads(plan_path.read_text(encoding="utf-8"))["order"]
        if len(completed_entries(read_manifest(folder))) < len(order):
            unfinished.append(folder)
    return unfinished


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--yes", action="store_true", help="Actually run (without it: preview only)")
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-cost", type=float, default=1.00, help="Spend guard in USD (default 1.00)")
    parser.add_argument("--out", type=Path, help="Experiment folder (default traces/experiment_<today>)")
    args = parser.parse_args()

    settings = load_settings()
    prices = load_prices(settings.prices_path)
    missing = [m for m in MODELS if m not in prices]
    if missing:
        sys.exit(f"No prices configured for {missing} in {settings.prices_path}. Fill them in, then rerun.")
    token_budget = settings.limits.token_budget

    out = args.out or ROOT / "traces" / f"experiment_{date.today().isoformat()}"
    if not args.out and not out.exists():
        unfinished = other_unfinished_experiments(out)
        if unfinished:
            sys.exit(f"Unfinished experiment(s) found: {[str(p) for p in unfinished]}. Pass --out to resume one, "
                     "or --out with a new folder to start fresh.")

    # Preview never writes files; a real run creates or checks plan.json.
    plan = load_or_create_plan(out, args.seed, args.repeats) if (args.yes or (out / PLAN_FILE).exists()) else build_plan(args.seed, args.repeats)
    entries = read_manifest(out) if out.exists() else []
    done = completed_entries(entries)
    remaining = [s for s in plan if s.key not in done]

    history = [p for p in (ROOT / "traces").rglob("*.jsonl") if not any(part.startswith("experiment_") for part in p.parts)]
    costs = baseline_costs(history, prices)
    expected, no_history = 0.0, set()
    for s in remaining:
        if costs.get((s.model, s.question_id)):
            expected += mean(costs[(s.model, s.question_id)])
        else:
            expected += worst_case_run_cost(s.model, prices, token_budget)
            no_history.add((s.model, s.question_id))
    worst = sum(worst_case_run_cost(s.model, prices, token_budget) for s in remaining)
    already = spent(entries, prices, token_budget)

    print(f"Experiment folder: {out}")
    print(f"Seed: {args.seed} | repeats: {args.repeats} | runs: {len(plan)} total, {len(done)} completed, {len(remaining)} to run")
    for model in MODELS:
        n = sum(s.model == model for s in remaining)
        print(f"  {model}: {n} runs, about {2 * n} API calls (max {n * settings.limits.max_iterations})")
    print(f"Estimated cost of remaining runs: ${expected:.4f} (average of earlier runs of the same questions"
          f"{'; worst case used for ' + str(sorted(no_history)) if no_history else ''})")
    print(f"Worst case if every run used its full {token_budget:,}-token budget at the dearer rate: ${worst:.2f}")
    print(f"Already spent in this folder: ${already:.4f} | spend guard: ${args.max_cost:.2f}")
    for (model, qid), values in sorted(costs.items()):
        print(f"  basis {model} {qid}: {len(values)} earlier run(s), avg ${mean(values):.5f}")

    if not args.yes:
        print("\nPreview only. Pass --yes to start.")
        return 0
    if not remaining:
        print("All runs are complete.")
        return 0

    load_dotenv(ROOT / ".env")
    sessions = {}

    def ask(spec):
        if (spec.model, spec.hints) not in sessions:
            sessions[(spec.model, spec.hints)] = Session(argparse.Namespace(
                config=None, provider=None, model=spec.model, base_url=None,
                max_iterations=None, timeout=None, token_budget=None, hints=spec.hints,
            ))
        return sessions[(spec.model, spec.hints)].ask(QUESTIONS[spec.question_id], traces_dir=out)

    try:
        outcome = run_plan(plan, out, ask, prices, args.max_cost, token_budget)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Rerun with --out {out} --yes to resume.")
        return 1
    except LLMError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    entries = read_manifest(out)
    print(f"\nStopped: {outcome}. Completed {len(completed_entries(entries))}/{len(plan)} runs. "
          f"Estimated spend in this folder: ${spent(entries, prices, token_budget):.4f}")
    return 0 if outcome == "done" else 1


if __name__ == "__main__":
    sys.exit(main())
