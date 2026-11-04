"""Run every question in a file through the agent, one run (and one trace) per question.

Usage:
    python scripts/run_questions.py questions/example_questions.txt --model claude-haiku-5-5
    python scripts/run_questions.py questions.json --model claude-sonnet-5-5 --yes

The file is either a JSON list of strings, or plain text with one question per
line (blank lines and lines starting with # are skipped). Answers go to the
traces only; nothing is graded. Compare models with scripts/summarize_traces.py.
"""

import argparse
import json
import sys
from pathlib import Path

from agent.cli import Session, add_common_args
from agent.llm import LLMError


def read_questions(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        questions = json.loads(text)
        if not isinstance(questions, list) or not all(isinstance(q, str) for q in questions):
            sys.exit("JSON questions file must be a list of strings.")
    else:
        questions = [line.strip() for line in text.splitlines()]
        questions = [q for q in questions if q and not q.startswith("#")]
    return questions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("questions_file", type=Path)
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation prompt")
    add_common_args(parser)
    args = parser.parse_args()

    questions = read_questions(args.questions_file)
    if not questions:
        sys.exit("No questions found.")
    try:
        session = Session(args)
    except (LLMError, ValueError, FileNotFoundError) as e:
        sys.exit(f"Error: {e}")

    s = session.settings
    print(f"{len(questions)} questions -> {s.provider}/{s.model}")
    print(f"Worst case: {len(questions) * s.limits.max_iterations} API calls, "
          f"{len(questions) * s.limits.token_budget:,} tokens (limits: {s.limits.max_iterations} calls, "
          f"{s.limits.token_budget:,} tokens per question)")
    if not args.yes and input("Continue? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return 1

    for i, question in enumerate(questions, 1):
        result = session.ask(question)
        print(f"[{i}/{len(questions)}] stop={result.stop_reason} iterations={result.iterations} "
              f"tokens={result.input_tokens}+{result.output_tokens} trace={result.trace_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
