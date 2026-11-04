"""Provider-neutral message format and the LLMClient interface.

The agent loop only ever sees these types. Each adapter converts them to and
from its provider's wire format.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Any  # parsed JSON object, or the raw string if the model sent invalid JSON


@dataclass
class ToolResult:
    tool_call_id: str
    content: str  # JSON text
    is_error: bool = False


@dataclass
class Message:
    role: str  # "user", "assistant" or "tool"
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    # Assistant messages only: the provider's original content, sent back unchanged on the
    # next request (Anthropic requires thinking blocks to be replayed as-is). Not traced.
    provider_content: Any = None


@dataclass
class Usage:
    input_tokens: int
    output_tokens: int

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class LLMResponse:
    message: Message
    usage: Usage | None  # None when the provider did not report usage
    stop_reason: str  # normalized: "end_turn", "tool_use", "max_tokens", "refusal" or "other"
    raw_stop_reason: str | None = None


class LLMError(Exception):
    """A failed LLM request. `retryable` says whether trying again might help."""

    retryable = False

    def __init__(self, message: str, status: int | None = None, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class TransientLLMError(LLMError):
    """Rate limits, server errors, network problems and timeouts."""

    retryable = True


class PermanentLLMError(LLMError):
    """Client errors such as a bad request, bad API key or unknown model. Never retried."""


class LLMClient(ABC):
    provider: str
    model: str

    @abstractmethod
    def complete(
        self, system: str, messages: list[Message], tools: list[dict], max_tokens: int, timeout: float
    ) -> LLMResponse:
        """Send one request. Raise LLMError (transient or permanent) on failure."""
