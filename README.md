# Olist Support Agent (Phase 1)

> **This is a demo on public data, not a production system.** It runs on a historical public dataset
> (Brazilian e-commerce orders from 2016–2018). It has no authentication, no real customers, and no
> guarantees about answer quality. It is a portfolio project for showing how a tool-calling agent can
> be built and measured.

A command-line support agent that answers questions about orders and sellers. It calls five read-only
tools against a local SQLite copy of the
[Olist Brazilian E-Commerce dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).
The tool-calling loop is written by hand in plain Python, with no agent framework.

```
question ──► loop ──► LLMClient (Anthropic | OpenAI-compatible)
              ▲  │
              │  └── tool calls ──► tools.py ──► SQLite (read-only)
              └───── tool results ◄─┘
every step ──► traces/<run>.jsonl
```

## Layout

```
config.toml            provider, model, limits (CLI flags override)
prices.toml            USD per million tokens, for estimated cost (you fill in / confirm)
src/agent/
  cli.py, __main__.py  python -m agent
  loop.py              the hand-written agent loop and its limits
  tools.py             the five tools: schemas, argument validation, fixed SQL
  db.py                read-only SQLite connection
  schema.py            table definitions shared by the build script and tests
  llm/base.py          provider-neutral messages + LLMClient interface
  llm/anthropic_client.py, llm/openai_compat.py   adapters
  llm/retry.py         capped retry for transient errors
  cost.py, trace.py, config.py
scripts/
  prepare_data.py      zip/CSVs -> data/olist.db
  run_questions.py     run a file of questions through the agent
  summarize_traces.py  per-model summary of a traces folder
tests/                 offline tests (FakeLLM, tiny fixture DB, network blocked)
questions/             example questions
data/, traces/, .env   gitignored
```

## Setup

Requires Python 3.11+. These commands are for Windows PowerShell. On macOS or Linux, use
`python3 -m venv .venv` and `source .venv/bin/activate` instead.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
copy .env.example .env      # then put your key in .env
```

`.env` holds `ANTHROPIC_API_KEY` (and `OPENAI_COMPAT_API_KEY` if you use that provider). It is gitignored.
Keys are read from the environment only. They are never printed, logged or written to traces.

## Get the data

1. Download the dataset zip from Kaggle:
   <https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce>.
2. Build the database:

```powershell
python scripts/prepare_data.py --zip path\to\archive.zip
# or, if you already extracted the CSVs:
python scripts/prepare_data.py --csv-dir data\raw\archive
```

The script checks each CSV's columns against `src/agent/schema.py` and fails loudly on any mismatch.
It loads the tables into `data/olist.db`, adds indexes, and prints row counts. Geolocation is not
loaded. **Review comment text is never loaded**, so no tool and no model can see it. Only numeric
review scores are kept.

The dataset and the database are not committed to this repo. For the dataset license, see
[Data and license](#data-and-license).

## Run

```powershell
python -m agent "What is the status of order 4fa07e91d7b4f8b191e3a6ab40f14f54?"
python -m agent -v "..."          # also show each tool call
python -m agent                   # interactive mode: each question is a separate run
```

Each run prints the answer and a footer showing the stop reason, iterations, tokens, estimated cost
and trace file name.

### Switch provider or model

```powershell
python -m agent --model claude-sonnet-5-5 "..."
python -m agent --provider openai_compat --base-url http://localhost:11434/v1 --model <model> "..."
python -m agent --max-iterations 4 --timeout 30 --token-budget 10000 "..."
```

Defaults are in `config.toml`. If a model name is rejected, the API's error is shown together with the
model name, and the run stops with `llm_error`.

### Compare models on the same questions

```powershell
python scripts/run_questions.py questions/example_questions.txt --model claude-haiku-5-5
python scripts/run_questions.py questions/example_questions.txt --model claude-sonnet-5-5
python scripts/summarize_traces.py
```

`run_questions.py` prints the worst-case number of API calls and tokens and asks for confirmation
(skip it with `--yes`). Answers go to the traces only. Nothing is graded.

`summarize_traces.py` reports the following for each provider/model, using only what is in the traces:
- runs and stop reasons
- average iterations, run duration and LLM call latency
- retries and LLM errors
- tool calls and tool errors by type
- total tokens
- total estimated cost (when prices were configured)

## Tests

```powershell
python -m pytest
```

All tests are offline. Sockets are blocked by a fixture, and the LLM is replaced by a scripted
`FakeLLM`. The tests cover:
- every tool, against a hand-made fixture DB
- the loop: multi-step tool use, unknown tools, bad arguments, tool exceptions, and each limit
- retries, the trace format, the trace summary and cost calculation
- adapter message conversion and error mapping
- that the DB connection cannot write

## How the cost estimate works

- Each provider response reports input and output tokens. These are summed per run and recorded per
  call in the trace.
- `prices.toml` maps a model name to USD per million input and output tokens. **It ships with the
  entries commented out.** Verify the prices on the provider's pricing page, then uncomment them.
- Cost is computed only for models with a price entry, and is always labelled *estimated*. With no
  price, the cost is `null`/`n/a`. If a call reports no usage, that run's cost is also unknown.
- Claude models think before answering, and thinking is billed as output tokens. `config.toml` sets
  `effort = "low"` for Anthropic to keep simple lookups cheap. Remove it to use the model's default.
- Prompt caching is not used, so all input tokens are priced at the normal input rate.

### What bounds spending

| Limit | Default | Notes |
|---|---|---|
| `max_iterations` | 8 | LLM calls per run |
| `timeout_seconds` | 60 | Wall clock per run. Each HTTP request gets only the time left, and retries never sleep past it |
| `token_budget` | 20,000 | Input + output tokens summed over the run's calls |
| `max_output_tokens` | 2,048 | Per call, and never more than the budget left |
| retries | 3 attempts | Rate limits, 5xx and network errors only, with backoff capped at 8 s. Client errors (400/401/404) are never retried |

The token budget is checked after each call, because a call's usage is only known once it returns.
So a run can exceed the budget by at most the input tokens of its last call; its output is already
clamped to what was left.

## Traces

Each run writes `traces/<utc-time>_<run_id>.jsonl`, with one JSON object per line:

- `run_start`: provider, model, limits, system prompt, tool names, the user message
- `llm_call`: iteration, latency, retry attempts, `max_tokens` sent, token usage, estimated cost, stop
  reason, and the assistant message (text and tool calls with arguments)
- `tool_call`: tool, arguments, latency, result size in characters, error code, and the result itself
- `llm_error`: error message and HTTP status
- `run_end`: stop reason, iterations, final answer, total usage, estimated cost, duration

Stop reasons:

| Stop reason | Meaning |
|---|---|
| `final_answer` | The model answered |
| `max_iterations`, `token_budget`, `timeout` | A limit was hit |
| `output_truncated` | The model hit `max_tokens` |
| `refusal` | The model declined to answer |
| `llm_error` | A request failed and was not retried, or ran out of retries |

## Design decisions

- **A hand-written loop over a small, provider-neutral message format.** The loop in `loop.py` is short
  enough to read in one sitting. Adapters translate to and from each API. The Anthropic adapter
  replays the model's original content blocks unchanged, which the API requires for thinking blocks.
- **The tools are the only access to data.** The model never writes SQL. Every query is fixed and
  parameterized. Arguments are checked against the schema: the exact keys, and IDs that are 32
  lowercase hex characters. Invalid arguments, unknown tools, missing records and tool exceptions all
  go back to the model as error objects. They never crash the run.
- **Three layers of read-only.** The file is opened with `mode=ro`, the connection sets
  `PRAGMA query_only`, and a SQLite authorizer allows only reads. The authorizer was added after a
  test showed that `ATTACH DATABASE` slips past the first two guards.
- **Small results.** Item lists are capped at 20 and payment lists at 10. Totals always cover all rows.
  Results larger than 4,000 characters are refused.
- **Review data.** Only numeric scores are kept, and only in seller aggregates. Each (review, order)
  pair is counted once, because the dataset has duplicate and multi-order reviews.
- **Retries in one place.** SDK retries are turned off, so every retry is the agent's own and shows
  in the trace.
- **Everything is configurable, nothing is hard-coded in the loop.** Provider, model, limits and
  prices come from config files and CLI flags.

## Known limitations

- **Old data.** The data is historical (2016–2018). "Where is my order" means its last recorded state.
- **No memory between questions.** Interactive mode treats each question as a new run.
- **Delivery definitions.** An order counts as late when it was delivered on a later *calendar date*
  than the estimate, because Olist estimates have no time of day. Timestamps have no timezone in the
  source data.
- **Data gaps the tools report as-is.** These are passed through, not corrected:
  - 8 orders are marked delivered but have no delivery date
  - 6 cancelled orders have a delivery date
  - some orders were handed to the carrier before their recorded approval time
- **Missing English names.** Two product categories have no English translation, so the Portuguese
  name is shown. Some products have no category at all.
- **Review attribution.** On orders with several sellers, the order's review score counts toward every
  seller on it.
- **Usage reporting.** If an OpenAI-compatible endpoint does not report usage, the token budget cannot
  be enforced for that run and no cost is estimated.
- **No grading.** Nothing checks whether answers are correct. The traces let you check by hand.

## Data and license

This project uses the Brazilian E-Commerce Public Dataset by Olist, from Kaggle
(https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce), licensed under CC BY-NC-SA 4.0
(https://creativecommons.org/licenses/by-nc-sa/4.0/). The dataset is not included in this repository.

Changes made: the CSV files are loaded into a local SQLite database. Review text is not loaded; only
numeric review scores are kept.

Files in scoring/ contain identifiers and values derived from the dataset and are shared under
CC BY-NC-SA 4.0.

This is a non-commercial portfolio and educational demo on public data, not a production system.

Code in this repository is licensed under the MIT License; see LICENSE.
