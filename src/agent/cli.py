"""Command-line interface: `python -m agent "question"`, or no question for interactive mode."""

import argparse
import json
import sys

from agent.config import ROOT, Settings, load_dotenv, load_settings
from agent.cost import load_prices
from agent.db import connect_readonly
from agent.llm import PROVIDERS, LLMError, make_client
from agent.loop import RunResult, run_agent
from agent.trace import Tracer


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", help="Path to config file (default: config.toml)")
    parser.add_argument("--provider", choices=PROVIDERS, help="Override the provider")
    parser.add_argument("--model", help="Override the model name")
    parser.add_argument("--base-url", help="Override base_url (openai_compat only)")
    parser.add_argument("--max-iterations", type=int, help="Max LLM calls per run")
    parser.add_argument("--timeout", type=float, help="Wall-clock seconds per run")
    parser.add_argument("--token-budget", type=int, help="Max input+output tokens per run")
    parser.add_argument("--hints", choices=("on", "off"), help="Show hint fields to the model (default: on)")


class Session:
    """Everything a run needs, built once and reused for many questions."""

    def __init__(self, args: argparse.Namespace):
        load_dotenv(ROOT / ".env")
        self.settings: Settings = load_settings(
            args.config,
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            max_iterations=args.max_iterations,
            timeout_seconds=args.timeout,
            token_budget=args.token_budget,
            hints=args.hints,
        )
        self.conn = connect_readonly(self.settings.db_path)
        self.prices = load_prices(self.settings.prices_path)
        self.llm = make_client(self.settings.provider, self.settings.model, self.settings.provider_options)

    def ask(self, question: str, traces_dir=None) -> RunResult:
        with Tracer(traces_dir or self.settings.traces_dir) as tracer:
            return run_agent(
                question, self.llm, self.conn, self.settings.limits, tracer, prices=self.prices, hints=self.settings.hints
            )


def format_cost(result: RunResult, model: str) -> str:
    if result.estimated_cost_usd is None:
        return f"n/a (no price configured for {model})"
    return f"${result.estimated_cost_usd:.6f} (estimated)"


def print_result(result: RunResult, session: Session, verbose: bool) -> None:
    if verbose:
        for line in result.trace_path.read_text(encoding="utf-8").splitlines():
            event = json.loads(line)
            if event["event"] == "tool_call":
                outcome = event["error_code"] if event["is_error"] else "ok"
                print(f"  tool {event['tool']}({json.dumps(event['arguments'])}) -> {outcome}, {event['latency_ms']} ms")
    if result.answer:
        print(result.answer)
    else:
        print(f"[No final answer: stopped by {result.stop_reason}]")
    if result.error:
        print(f"[Error: {result.error}]", file=sys.stderr)
    hints = "" if session.settings.hints else " | hints: off"
    print(
        f"\n[{session.settings.provider}/{session.settings.model}{hints} | stop: {result.stop_reason} | "
        f"iterations: {result.iterations} | tokens: {result.input_tokens} in / {result.output_tokens} out | "
        f"cost: {format_cost(result, session.settings.model)} | trace: {result.trace_path.name}]"
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m agent", description="Olist support agent (demo on public data).")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Show each tool call")
    add_common_args(parser)
    args = parser.parse_args(argv)
    # On Windows, redirected output uses a legacy code page; never crash on a character it lacks.
    sys.stdout.reconfigure(errors="replace")

    try:
        session = Session(args)
    except (LLMError, ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2

    if args.question:
        result = session.ask(args.question)
        print_result(result, session, args.verbose)
        return 0 if result.stop_reason == "final_answer" else 1

    print(f"Olist support agent ({session.settings.provider}/{session.settings.model}). "
          "Each question is a separate run. Empty line or 'exit' to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question or question.lower() in ("exit", "quit"):
            return 0
        print_result(session.ask(question), session, args.verbose)
