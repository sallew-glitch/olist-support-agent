"""Adapter for any OpenAI-compatible Chat Completions endpoint (hosted or local)."""

import json

import openai

from agent.llm.base import (
    LLMClient, LLMResponse, Message, PermanentLLMError, ToolCall, TransientLLMError, Usage,
)

_STOP_REASONS = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "refusal",
}


def to_openai_messages(system: str, messages: list[Message]) -> list[dict]:
    out = [{"role": "system", "content": system}]
    for m in messages:
        if m.role == "user":
            out.append({"role": "user", "content": m.text})
        elif m.role == "assistant":
            msg = {"role": "assistant", "content": m.text or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            # Send back exactly what the model produced, even if it was invalid JSON.
                            "arguments": c.arguments if isinstance(c.arguments, str) else json.dumps(c.arguments),
                        },
                    }
                    for c in m.tool_calls
                ]
            out.append(msg)
        elif m.role == "tool":
            out += [{"role": "tool", "tool_call_id": r.tool_call_id, "content": r.content} for r in m.tool_results]
        else:
            raise ValueError(f"Unknown role: {m.role}")
    return out


def _parse_arguments(raw: str):
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return raw  # the tool validator will reject it and tell the model


def from_openai_response(response) -> LLMResponse:
    if not response.choices:
        raise PermanentLLMError("OpenAI-compatible endpoint returned no choices.")
    choice = response.choices[0]
    msg = choice.message
    calls = [
        ToolCall(tc.id, tc.function.name, _parse_arguments(tc.function.arguments))
        for tc in (msg.tool_calls or [])
        if getattr(tc, "function", None) is not None
    ]
    usage = None
    if response.usage is not None:
        usage = Usage(response.usage.prompt_tokens or 0, response.usage.completion_tokens or 0)
    return LLMResponse(
        message=Message(role="assistant", text=msg.content or "", tool_calls=calls),
        usage=usage,
        stop_reason=_STOP_REASONS.get(choice.finish_reason, "other"),
        raw_stop_reason=choice.finish_reason,
    )


class OpenAICompatClient(LLMClient):
    provider = "openai_compat"

    def __init__(self, model: str, base_url: str, api_key: str | None):
        self.model = model
        # Local servers often need no key, but the SDK requires a non-empty string.
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "not-set", max_retries=0)

    def request_params(self) -> dict:
        return {"model": self.model, "base_url": str(self._client.base_url), "sampling_params_sent": []}

    def complete(self, system, messages, tools, max_tokens, timeout) -> LLMResponse:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=to_openai_messages(system, messages),
                tools=[
                    {"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                    for t in tools
                ],
                max_tokens=max_tokens,
                timeout=timeout,
            )
        except openai.APITimeoutError as e:
            raise TransientLLMError(f"Request timed out: {e}") from e
        except openai.APIConnectionError as e:
            raise TransientLLMError(f"Could not reach the endpoint: {e}") from e
        except openai.APIStatusError as e:
            message = f"API error {e.status_code}: {e.message}"
            if e.status_code == 404:
                message += f" (is the model name '{self.model}' and base_url correct?)"
            if e.status_code in (408, 409, 429) or e.status_code >= 500:
                raise TransientLLMError(message, e.status_code) from e
            raise PermanentLLMError(message, e.status_code) from e
        return from_openai_response(response)
