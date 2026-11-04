"""LLM layer: a provider-neutral interface plus one adapter per provider."""

import os

from agent.llm.base import LLMClient, LLMError

PROVIDERS = ("anthropic", "openai_compat")


def make_client(provider: str, model: str, options: dict) -> LLMClient:
    """Build a client from the [providers.<name>] config table. Keys are read from the environment only."""
    api_key = os.environ.get(options.get("api_key_env", ""), "") or None

    if provider == "anthropic":
        from agent.llm.anthropic_client import AnthropicClient

        if not api_key:
            raise LLMError(f"Set {options.get('api_key_env', 'ANTHROPIC_API_KEY')} in your environment or .env file.")
        return AnthropicClient(model, api_key, effort=options.get("effort") or None, base_url=options.get("base_url") or None)

    if provider == "openai_compat":
        from agent.llm.openai_compat import OpenAICompatClient

        if not options.get("base_url"):
            raise LLMError("Set base_url under [providers.openai_compat] in config.toml or pass --base-url.")
        return OpenAICompatClient(model, options["base_url"], api_key)

    raise LLMError(f"Unknown provider '{provider}'. Choose one of: {', '.join(PROVIDERS)}.")
