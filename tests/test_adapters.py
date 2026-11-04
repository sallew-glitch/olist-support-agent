"""Adapter conversions and error mapping, tested offline with stand-in SDK objects."""

from types import SimpleNamespace as NS

import anthropic
import httpx2
import openai
import pytest

from agent.llm.anthropic_client import AnthropicClient, from_anthropic_response, to_anthropic_messages
from agent.llm.base import Message, PermanentLLMError, ToolCall, ToolResult, TransientLLMError
from agent.llm.openai_compat import OpenAICompatClient, from_openai_response, to_openai_messages

CONVERSATION = [
    Message(role="user", text="Where is order X?"),
    Message(role="assistant", text="Checking.", tool_calls=[ToolCall("t1", "get_order_status", {"order_id": "x"})]),
    Message(role="tool", tool_results=[ToolResult("t1", '{"status": "shipped"}')]),
]


# --- Anthropic ---

def test_anthropic_messages():
    out = to_anthropic_messages(CONVERSATION)
    assert out[0] == {"role": "user", "content": "Where is order X?"}
    assert out[1]["content"] == [
        {"type": "text", "text": "Checking."},
        {"type": "tool_use", "id": "t1", "name": "get_order_status", "input": {"order_id": "x"}},
    ]
    assert out[2] == {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": '{"status": "shipped"}', "is_error": False}],
    }


def test_anthropic_replays_provider_content_unchanged():
    raw = [NS(type="thinking", thinking="", signature="sig"), NS(type="text", text="hi")]
    out = to_anthropic_messages([Message(role="assistant", text="hi", provider_content=raw)])
    assert out[0]["content"] is raw


def test_anthropic_response_parsing_skips_thinking():
    response = NS(
        content=[
            NS(type="thinking", thinking="", signature="sig"),
            NS(type="text", text="Let me check."),
            NS(type="tool_use", id="t9", name="get_order_items", input={"order_id": "y"}),
        ],
        usage=NS(input_tokens=500, output_tokens=80, cache_creation_input_tokens=None, cache_read_input_tokens=0),
        stop_reason="tool_use",
    )
    r = from_anthropic_response(response)
    assert r.message.text == "Let me check."
    assert r.message.tool_calls == [ToolCall("t9", "get_order_items", {"order_id": "y"})]
    assert (r.usage.input_tokens, r.usage.output_tokens) == (500, 80)
    assert r.stop_reason == "tool_use"


def _status_error(cls, status, headers=None):
    req = httpx2.Request("POST", "https://example.invalid")
    return cls("boom", response=httpx2.Response(status, request=req, headers=headers or {}), body=None)


def _anthropic_client_raising(exc):
    client = AnthropicClient("claude-test", api_key="unused")

    def create(**kwargs):
        raise exc

    client._client = NS(messages=NS(create=create))
    return client


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_status_error(anthropic.RateLimitError, 429, {"retry-after": "3"}), TransientLLMError),
        (_status_error(anthropic.InternalServerError, 529), TransientLLMError),
        (anthropic.APITimeoutError(request=httpx2.Request("POST", "https://example.invalid")), TransientLLMError),
        (anthropic.APIConnectionError(request=httpx2.Request("POST", "https://example.invalid")), TransientLLMError),
        (_status_error(anthropic.BadRequestError, 400), PermanentLLMError),
        (_status_error(anthropic.AuthenticationError, 401), PermanentLLMError),
        (_status_error(anthropic.NotFoundError, 404), PermanentLLMError),
    ],
)
def test_anthropic_error_mapping(exc, expected):
    client = _anthropic_client_raising(exc)
    with pytest.raises(expected) as info:
        client.complete("sys", CONVERSATION, [], 100, 10)
    if isinstance(exc, anthropic.RateLimitError):
        assert info.value.retry_after == 3.0
    if isinstance(exc, anthropic.NotFoundError):
        assert "claude-test" in str(info.value)  # unknown model names are surfaced clearly


def test_anthropic_request_shape():
    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        return NS(content=[NS(type="text", text="ok")], usage=NS(input_tokens=1, output_tokens=1,
                  cache_creation_input_tokens=0, cache_read_input_tokens=0), stop_reason="end_turn")

    client = AnthropicClient("claude-test", api_key="unused", effort="low")
    client._client = NS(messages=NS(create=create))
    tool = {"name": "t", "description": "d", "input_schema": {"type": "object"}}
    client.complete("sys", CONVERSATION[:1], [tool], 123, 9.5)
    assert sent["model"] == "claude-test"
    assert sent["max_tokens"] == 123
    assert sent["timeout"] == 9.5
    assert sent["system"] == "sys"
    assert sent["output_config"] == {"effort": "low"}
    assert sent["tools"] == [tool]


# --- OpenAI-compatible ---

def test_openai_messages():
    out = to_openai_messages("sys", CONVERSATION)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1] == {"role": "user", "content": "Where is order X?"}
    assert out[2]["tool_calls"][0] == {
        "id": "t1", "type": "function", "function": {"name": "get_order_status", "arguments": '{"order_id": "x"}'},
    }
    assert out[3] == {"role": "tool", "tool_call_id": "t1", "content": '{"status": "shipped"}'}


def test_openai_invalid_json_arguments_kept_raw():
    tc = NS(id="c1", function=NS(name="get_order_status", arguments='{"order_id": '))
    response = NS(
        choices=[NS(message=NS(content=None, tool_calls=[tc]), finish_reason="tool_calls")],
        usage=NS(prompt_tokens=50, completion_tokens=5),
    )
    r = from_openai_response(response)
    assert r.message.tool_calls[0].arguments == '{"order_id": '
    assert r.stop_reason == "tool_use"
    assert to_openai_messages("s", [r.message])[1]["tool_calls"][0]["function"]["arguments"] == '{"order_id": '


def test_openai_missing_usage_is_none():
    response = NS(choices=[NS(message=NS(content="hi", tool_calls=None), finish_reason="stop")], usage=None)
    r = from_openai_response(response)
    assert r.usage is None
    assert r.stop_reason == "end_turn"


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_status_error(openai.RateLimitError, 429), TransientLLMError),
        (_status_error(openai.InternalServerError, 503), TransientLLMError),
        (_status_error(openai.NotFoundError, 404), PermanentLLMError),
        (_status_error(openai.BadRequestError, 400), PermanentLLMError),
    ],
)
def test_openai_error_mapping(exc, expected):
    client = OpenAICompatClient("local-model", "http://localhost:1/v1", None)

    def create(**kwargs):
        raise exc

    client._client = NS(chat=NS(completions=NS(create=create)))
    with pytest.raises(expected):
        client.complete("sys", CONVERSATION, [], 100, 10)
