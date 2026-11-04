import pytest

from agent.llm.base import PermanentLLMError, TransientLLMError
from agent.llm.retry import call_with_retry
from fakes import FakeClock


def flaky(errors, result="ok"):
    """A function that raises the given errors in order, then returns result."""
    errors = list(errors)
    calls = []

    def fn():
        calls.append(1)
        if errors:
            raise errors.pop(0)
        return result

    return fn, calls


def test_backoff_doubles_and_is_capped():
    clock = FakeClock()
    fn, calls = flaky([TransientLLMError("x")] * 4)
    assert call_with_retry(fn, max_attempts=5, base_delay=3, max_delay=8, clock=clock, sleep=clock.sleep) == ("ok", 5)
    assert clock.slept == [3, 6, 8, 8]


def test_gives_up_after_max_attempts():
    clock = FakeClock()
    fn, calls = flaky([TransientLLMError("x")] * 5)
    with pytest.raises(TransientLLMError):
        call_with_retry(fn, max_attempts=3, clock=clock, sleep=clock.sleep)
    assert len(calls) == 3


def test_client_errors_are_not_retried():
    fn, calls = flaky([PermanentLLMError("bad request", 400)])
    with pytest.raises(PermanentLLMError):
        call_with_retry(fn, sleep=lambda s: pytest.fail("should not sleep"))
    assert len(calls) == 1


def test_retry_after_is_honoured_but_capped():
    clock = FakeClock()
    fn, _ = flaky([TransientLLMError("429", 429, retry_after=2.5), TransientLLMError("429", 429, retry_after=60)])
    call_with_retry(fn, max_delay=8, clock=clock, sleep=clock.sleep)
    assert clock.slept == [2.5, 8]


def test_never_sleeps_past_deadline():
    clock = FakeClock(start=0)
    fn, calls = flaky([TransientLLMError("x")])
    with pytest.raises(TransientLLMError):
        call_with_retry(fn, base_delay=5, deadline=3, clock=clock, sleep=clock.sleep)
    assert clock.slept == []
