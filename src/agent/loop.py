"""The agent loop, written by hand.

    send messages + tool schemas -> model
    if the model asked for tools: run them, append results, repeat
    otherwise: the text is the final answer

Three limits stop a run early: max iterations (LLM calls), wall-clock timeout,
and a token budget (input + output tokens summed over all calls). The reason a
run stopped is always recorded.
"""

import json
import sqlite3
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from agent.cost import estimate_cost
from agent.llm.base import LLMClient, LLMError, Message, ToolResult
from agent.llm.retry import call_with_retry
from agent.tools import execute_tool, tool_specs
from agent.trace import Tracer, message_record

SYSTEM_PROMPT = """\
You are a customer-support assistant for a demo built on the public Olist Brazilian e-commerce dataset (orders from 2016 to 2018).

Rules:
- Answer only from the results of tools you called in this conversation. Never invent or guess order data: no made-up dates, amounts, statuses, categories or IDs.
- If a tool returns an error, or the data you need is missing or null, say so plainly.
- Order and seller IDs are 32-character lowercase hexadecimal strings. If the user has not given the ID you need, ask for it.
- Your tools are read-only. You cannot cancel, refund or change anything; say so if asked.
- Customer review text is not available. Only average review scores per seller are.
- Report dates and amounts as they appear in the tool results (amounts are in Brazilian reais, BRL).
- Keep answers short and factual."""


@dataclass
class Limits:
    max_iterations: int = 8
    timeout_seconds: float = 60.0
    token_budget: int = 20_000
    max_output_tokens: int = 2048  # per LLM call; also clamped to what is left of the budget

    def __post_init__(self):
        for name, value in asdict(self).items():
            if value <= 0:
                raise ValueError(f"limit {name} must be positive, got {value}")


@dataclass
class RunResult:
    answer: str | None
    stop_reason: str  # final_answer, max_iterations, token_budget, timeout, output_truncated, refusal, llm_error
    iterations: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None
    trace_path: Path
    error: str | None = None


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def run_agent(
    question: str,
    llm: LLMClient,
    conn: sqlite3.Connection,
    limits: Limits,
    tracer: Tracer,
    prices: dict | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> RunResult:
    prices = prices or {}
    tools = tool_specs()
    start = clock()
    deadline = start + limits.timeout_seconds
    messages = [Message(role="user", text=question)]

    iterations = 0
    input_tokens = output_tokens = 0
    cost: float | None = 0.0 if llm.model in prices else None
    answer = error = None
    stop_reason = None

    tracer.event(
        "run_start",
        provider=llm.provider,
        model=llm.model,
        limits=asdict(limits),
        system_prompt=SYSTEM_PROMPT,
        tools=[t["name"] for t in tools],
        message=message_record(messages[0]),
    )

    def limit_hit() -> str | None:
        if iterations >= limits.max_iterations:
            return "max_iterations"
        if input_tokens + output_tokens >= limits.token_budget:
            return "token_budget"
        if clock() >= deadline:
            return "timeout"
        return None

    while stop_reason is None:
        stop_reason = limit_hit()
        if stop_reason:
            break

        # --- 1. ask the model ---
        iterations += 1
        max_tokens = min(limits.max_output_tokens, limits.token_budget - input_tokens - output_tokens)
        t0 = clock()
        try:
            response, attempts = call_with_retry(
                lambda: llm.complete(SYSTEM_PROMPT, messages, tools, max_tokens, timeout=max(deadline - clock(), 0.001)),
                deadline=deadline,
                clock=clock,
                sleep=sleep,
            )
        except LLMError as e:
            error = str(e)
            stop_reason = "timeout" if clock() >= deadline else "llm_error"
            tracer.event("llm_error", iteration=iterations, latency_ms=_ms(clock() - t0), error=error, status=e.status)
            break

        call_cost = estimate_cost(response.usage, llm.model, prices)
        if response.usage:
            input_tokens += response.usage.input_tokens
            output_tokens += response.usage.output_tokens
        if cost is not None:
            cost = cost + call_cost if call_cost is not None else None  # a call without usage makes the total unknown
        tracer.event(
            "llm_call",
            iteration=iterations,
            latency_ms=_ms(clock() - t0),
            attempts=attempts,
            max_tokens=max_tokens,
            usage=asdict(response.usage) if response.usage else None,
            estimated_cost_usd=call_cost,
            stop_reason=response.stop_reason,
            raw_stop_reason=response.raw_stop_reason,
            message=message_record(response.message),
        )
        messages.append(response.message)

        # --- 2. no tool calls: this is the final answer ---
        if not response.message.tool_calls:
            answer = response.message.text
            stop_reason = {"max_tokens": "output_truncated", "refusal": "refusal"}.get(response.stop_reason, "final_answer")
            break
        if response.stop_reason == "max_tokens":
            stop_reason = "output_truncated"  # tool call may be cut off; do not run it
            break

        # Do not run tools whose results could never be sent back to the model.
        stop_reason = limit_hit()
        if stop_reason:
            break

        # --- 3. run the requested tools and append their results ---
        results = []
        for call in response.message.tool_calls:
            t0 = clock()
            result = execute_tool(conn, call.name, call.arguments)
            content = json.dumps(result, ensure_ascii=False)
            is_error = "error" in result
            tracer.event(
                "tool_call",
                iteration=iterations,
                tool_call_id=call.id,
                tool=call.name,
                arguments=call.arguments,
                latency_ms=_ms(clock() - t0),
                result_chars=len(content),
                is_error=is_error,
                error_code=result["error"]["code"] if is_error else None,
                result=result,
            )
            results.append(ToolResult(call.id, content, is_error))
        messages.append(Message(role="tool", tool_results=results))

    tracer.event(
        "run_end",
        stop_reason=stop_reason,
        iterations=iterations,
        final_answer=answer,
        error=error,
        usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
        estimated_cost_usd=cost,
        duration_ms=_ms(clock() - start),
    )
    return RunResult(answer, stop_reason, iterations, input_tokens, output_tokens, cost, tracer.path, error)
