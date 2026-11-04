"""Small, capped retry for transient LLM errors."""

import time
from typing import Callable, TypeVar

from agent.llm.base import LLMError

T = TypeVar("T")


def call_with_retry(
    fn: Callable[[], T],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 8.0,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[T, int]:
    """Call fn, retrying only retryable LLMErrors. Returns (result, attempts).

    Backoff doubles from base_delay, honours a server's retry-after hint, and is
    capped at max_delay. We never sleep past the run's deadline.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(), attempt
        except LLMError as e:
            if not e.retryable or attempt == max_attempts:
                raise
            delay = e.retry_after if e.retry_after is not None else base_delay * 2 ** (attempt - 1)
            delay = min(delay, max_delay)
            if deadline is not None and clock() + delay >= deadline:
                raise
            sleep(delay)
    raise AssertionError("unreachable")
