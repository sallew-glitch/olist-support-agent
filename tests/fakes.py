"""Test doubles: a scripted LLM and a controllable clock. No network."""

import copy
import itertools

from agent.llm.base import LLMClient, LLMResponse, Message, ToolCall, Usage

_ids = itertools.count(1)


def tool_calls(*calls, usage=(100, 20), stop_reason="tool_use") -> LLMResponse:
    """A response asking for tools. Each call is (name, arguments)."""
    return LLMResponse(
        message=Message(role="assistant", tool_calls=[ToolCall(f"call_{next(_ids)}", n, a) for n, a in calls]),
        usage=Usage(*usage) if usage else None,
        stop_reason=stop_reason,
    )


def answer(text, usage=(100, 20), stop_reason="end_turn") -> LLMResponse:
    return LLMResponse(message=Message(role="assistant", text=text), usage=Usage(*usage) if usage else None, stop_reason=stop_reason)


class FakeLLM(LLMClient):
    """Returns scripted steps in order. A step is an LLMResponse, an exception to raise,
    or a zero-argument callable returning either."""

    provider = "fake"

    def __init__(self, script, model="fake-model"):
        self.model = model
        self.script = list(script)
        self.calls = []  # what the loop sent on each call

    def complete(self, system, messages, tools, max_tokens, timeout):
        self.calls.append({"messages": copy.deepcopy(messages), "max_tokens": max_tokens, "timeout": timeout, "tools": tools})
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        step = self.script.pop(0)
        if callable(step):
            step = step()
        if isinstance(step, Exception):
            raise step
        return step


class FakeClock:
    def __init__(self, start=1000.0):
        self.now = start
        self.slept = []

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds
