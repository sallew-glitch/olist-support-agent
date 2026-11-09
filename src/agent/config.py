"""Settings from config.toml, with command-line overrides on top."""

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

from agent.loop import Limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config.toml"


@dataclass
class Settings:
    provider: str
    model: str
    db_path: Path
    traces_dir: Path
    prices_path: Path
    limits: Limits
    provider_options: dict  # the [providers.<provider>] table from the config
    hints: bool = True  # False hides hint fields from the model (see agent.tools.apply_hints)


def load_dotenv(path: Path) -> None:
    """Minimal .env reader: KEY=VALUE lines. Never overrides variables already set. Values are never printed."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


def load_settings(
    config_path: str | Path | None = None,
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    max_iterations: int | None = None,
    timeout_seconds: float | None = None,
    token_budget: int | None = None,
    hints: str | None = None,
) -> Settings:
    config_path = Path(config_path) if config_path else DEFAULT_CONFIG
    with open(config_path, "rb") as f:
        cfg = tomllib.load(f)
    base = config_path.resolve().parent  # relative paths in the config are relative to the config file

    agent = cfg.get("agent", {})
    provider = provider or agent["provider"]
    providers = cfg.get("providers", {})
    if provider not in providers:
        raise ValueError(f"Provider '{provider}' has no [providers.{provider}] section in {config_path}.")
    options = dict(providers[provider])
    if base_url:
        options["base_url"] = base_url
    # --model wins, then the provider's own default, then [agent].model (only for the default provider:
    # a Claude model name means nothing to another provider).
    model = model or options.get("model") or (agent["model"] if provider == agent["provider"] else None)
    if not model:
        raise ValueError(f"No model set for provider '{provider}'. Pass --model or set model under [providers.{provider}].")

    limits = Limits(**cfg.get("limits", {}))
    overrides = {"max_iterations": max_iterations, "timeout_seconds": timeout_seconds, "token_budget": token_budget}
    limits = replace(limits, **{k: v for k, v in overrides.items() if v is not None})

    hints = hints or agent.get("hints", "on")
    if hints not in ("on", "off"):
        raise ValueError(f"hints must be 'on' or 'off', got {hints!r}")

    return Settings(
        provider=provider,
        model=model,
        db_path=base / agent.get("db_path", "data/olist.db"),
        traces_dir=base / agent.get("traces_dir", "traces"),
        prices_path=base / agent.get("prices_path", "prices.toml"),
        limits=limits,
        provider_options=options,
        hints=hints == "on",
    )
