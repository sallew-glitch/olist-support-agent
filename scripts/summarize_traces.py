"""Summarize a folder of JSONL traces, grouped by provider/model.

Usage:
    python scripts/summarize_traces.py            # reads traces/
    python scripts/summarize_traces.py path/to/traces

Reports only what is in the traces. Runs without a run_end event (e.g. the
process was killed) are counted as incomplete and left out of the averages.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_runs(folder: Path) -> tuple[list[list[dict]], int]:
    runs, bad_lines = [], 0
    for path in sorted(folder.glob("*.jsonl")):
        events = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                bad_lines += 1
        if events and events[0].get("event") == "run_start":
            runs.append(events)
    return runs, bad_lines


def summarize(runs: list[list[dict]]) -> dict:
    groups = defaultdict(lambda: {
        "runs": 0, "incomplete": 0, "stop_reasons": Counter(), "iterations": [], "durations_ms": [],
        "llm_latencies_ms": [], "retried_calls": 0, "llm_errors": 0, "tool_calls": Counter(),
        "tool_errors": Counter(), "input_tokens": 0, "output_tokens": 0, "cost": 0.0, "runs_with_cost": 0,
    })
    for events in runs:
        start = events[0]
        g = groups[f"{start['provider']}/{start['model']}"]
        g["runs"] += 1
        for e in events:
            if e["event"] == "llm_call":
                g["llm_latencies_ms"].append(e["latency_ms"])
                g["retried_calls"] += e.get("attempts", 1) > 1
            elif e["event"] == "llm_error":
                g["llm_errors"] += 1
            elif e["event"] == "tool_call":
                g["tool_calls"][e["tool"]] += 1
                if e["is_error"]:
                    g["tool_errors"][e["error_code"]] += 1
        end = next((e for e in events if e["event"] == "run_end"), None)
        if end is None:
            g["incomplete"] += 1
            continue
        g["stop_reasons"][end["stop_reason"]] += 1
        g["iterations"].append(end["iterations"])
        g["durations_ms"].append(end["duration_ms"])
        g["input_tokens"] += end["usage"]["input_tokens"]
        g["output_tokens"] += end["usage"]["output_tokens"]
        if end.get("estimated_cost_usd") is not None:
            g["cost"] += end["estimated_cost_usd"]
            g["runs_with_cost"] += 1
    return groups


def print_report(groups: dict) -> None:
    for name, g in sorted(groups.items()):
        completed = g["runs"] - g["incomplete"]
        print(f"\n=== {name} ===")
        print(f"runs:                 {g['runs']} ({completed} complete, {g['incomplete']} incomplete)")
        print(f"stop reasons:         {dict(g['stop_reasons'])}")
        if completed:
            print(f"avg iterations:       {mean(g['iterations']):.2f}")
            print(f"avg run duration:     {mean(g['durations_ms']) / 1000:.2f} s")
        if g["llm_latencies_ms"]:
            print(f"avg LLM call latency: {mean(g['llm_latencies_ms']) / 1000:.2f} s over {len(g['llm_latencies_ms'])} calls")
        print(f"retried LLM calls:    {g['retried_calls']}")
        print(f"LLM errors:           {g['llm_errors']}")
        print(f"tool calls:           {sum(g['tool_calls'].values())} {dict(g['tool_calls'].most_common())}")
        print(f"tool errors:          {sum(g['tool_errors'].values())} {dict(g['tool_errors'])}")
        print(f"tokens:               {g['input_tokens']:,} in / {g['output_tokens']:,} out")
        if g["runs_with_cost"]:
            print(f"estimated cost:       ${g['cost']:.4f} (from {g['runs_with_cost']} of {completed} complete runs with prices)")
        else:
            print("estimated cost:       n/a (no prices configured for this model in these traces)")


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "traces"
    if not folder.is_dir():
        sys.exit(f"No traces folder: {folder}")
    runs, bad_lines = load_runs(folder)
    if not runs:
        print(f"No traces found in {folder}")
        return 0
    print(f"{len(runs)} runs in {folder}")
    if bad_lines:
        print(f"(skipped {bad_lines} unreadable lines)")
    print_report(summarize(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
