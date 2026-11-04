import os

import pytest

from agent.config import DEFAULT_CONFIG, load_dotenv, load_settings
from agent.llm import LLMError, make_client


def test_defaults_from_shipped_config():
    s = load_settings(DEFAULT_CONFIG)
    assert s.provider == "anthropic"
    assert s.model == "claude-haiku-5-5"
    assert s.limits.max_iterations == 8
    assert s.limits.token_budget == 20000


def test_cli_overrides():
    s = load_settings(DEFAULT_CONFIG, model="claude-sonnet-5-5", max_iterations=3, timeout_seconds=30, token_budget=5000)
    assert s.model == "claude-sonnet-5-5"
    assert (s.limits.max_iterations, s.limits.timeout_seconds, s.limits.token_budget) == (3, 30, 5000)


def test_other_provider_requires_its_own_model():
    with pytest.raises(ValueError, match="--model"):
        load_settings(DEFAULT_CONFIG, provider="openai_compat")
    s = load_settings(DEFAULT_CONFIG, provider="openai_compat", model="some-model", base_url="http://x/v1")
    assert s.provider_options["base_url"] == "http://x/v1"


def test_unknown_provider():
    with pytest.raises(ValueError):
        load_settings(DEFAULT_CONFIG, provider="nope", model="m")


def test_missing_api_key_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        make_client("anthropic", "claude-haiku-5-5", {"api_key_env": "ANTHROPIC_API_KEY"})


def test_dotenv_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("# comment\nTEST_DOTENV_A=from_file\nTEST_DOTENV_B='quoted'\nTEST_DOTENV_EMPTY=\n")
    monkeypatch.setenv("TEST_DOTENV_A", "already_set")
    monkeypatch.delenv("TEST_DOTENV_B", raising=False)
    monkeypatch.delenv("TEST_DOTENV_EMPTY", raising=False)
    load_dotenv(env)
    assert os.environ["TEST_DOTENV_A"] == "already_set"
    assert os.environ["TEST_DOTENV_B"] == "quoted"
    assert "TEST_DOTENV_EMPTY" not in os.environ
    monkeypatch.delenv("TEST_DOTENV_B")
