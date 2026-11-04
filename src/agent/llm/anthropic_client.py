"""Adapter for the Anthropic Messages API."""

import anthropic

from agent.llm.base import (
    LLMClient, LLMResponse, Message, PermanentLLMError, ToolCall, TransientLLMError, Usage,
)

_STOP_REASONS = {
    "end_turn": "end_turn",
    "stop_sequence": "end_turn",
    "tool_use": "tool_use",
    "max_tokens": "max_tokens",
    "refusal": "refusal",
}


def to_anthropic_messages(messages: list[Message]) -> list[dict]:
    out = []
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            content = m.provider_content
            if content is None:  # built elsewhere (e.g. tests): reconstruct the blocks
                content = [{"type": "text", "text": m.text}] if m.text else []
                content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
            out.append({"role": "assistant", "content": content})
        elif m.role == "tool":
            # All results for one assistant turn go back in a single user message.
            out.append({
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": r.tool_call_id, "content": r.content, "is_error": r.is_error}
                    for r in m.tool_results
                ],
            })
        else:
            raise ValueError(f"Unknown role: {m.role}")
    return out


def from_anthropic_response(response) -> LLMResponse:
    # Read blocks by type: a response may start with thinking blocks.
    text = "\n".join(b.text for b in response.content if b.type == "text")
    calls = [ToolCall(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"]
    u = response.usage
    # Prompt caching is not used, so the cache fields are normally 0; count them anyway.
    input_tokens = u.input_tokens + (u.cache_creation_input_tokens or 0) + (u.cache_read_input_tokens or 0)
    return LLMResponse(
        message=Message(role="assistant", text=text, tool_calls=calls, provider_content=response.content),
        usage=Usage(input_tokens, u.output_tokens),
        stop_reason=_STOP_REASONS.get(response.stop_reason, "other"),
        raw_stop_reason=response.stop_reason,
    )


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, effort: str | None = None, base_url: str | None = None):
        self.model = model
        self.effort = effort
        # SDK retries are off: retries happen in one place (agent.llm.retry) so they are visible in traces.
        self._client = anthropic.Anthropic(api_key=api_key, base_url=base_url, max_retries=0)

    def complete(self, system, messages, tools, max_tokens, timeout) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": to_anthropic_messages(messages),
            "tools": [{"name": t["name"], "description": t["description"], "input_schema": t["input_schema"]} for t in tools],
            "timeout": timeout,
        }
        if self.effort:
            kwargs["output_config"] = {"effort": self.effort}
        try:
            response = self._client.messages.create(**kwargs)
        except anthropic.APITimeoutError as e:
            raise TransientLLMError(f"Anthropic request timed out: {e}") from e
        except anthropic.APIConnectionError as e:
            raise TransientLLMError(f"Could not reach the Anthropic API: {e}") from e
        except anthropic.APIStatusError as e:
            message = f"Anthropic API error {e.status_code}: {e.message}"
            if e.status_code == 404:
                message += f" (is the model name '{self.model}' correct and available to your account?)"
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                raise TransientLLMError(message, e.status_code, _retry_after(e)) from e
            raise PermanentLLMError(message, e.status_code) from e
        return from_anthropic_response(response)


def _retry_after(e: anthropic.APIStatusError) -> float | None:
    try:
        return float(e.response.headers.get("retry-after"))
    except (TypeError, ValueError):
        return None
