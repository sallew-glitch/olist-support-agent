"""Estimated cost from token usage and a user-maintained price table."""

import tomllib
from pathlib import Path

from agent.llm.base import Usage


def load_prices(path: str | Path) -> dict[str, dict[str, float]]:
    """Read prices.toml: {model: {"input_per_mtok": float, "output_per_mtok": float}}.

    A missing file or a model without an entry simply means "no estimate".
    """
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, "rb") as f:
        models = tomllib.load(f).get("models", {})
    for model, p in models.items():
        for key in ("input_per_mtok", "output_per_mtok"):
            if not isinstance(p.get(key), (int, float)) or p[key] < 0:
                raise ValueError(f"prices: models.{model}.{key} must be a non-negative number")
    return models


def estimate_cost(usage: Usage | None, model: str, prices: dict) -> float | None:
    """Estimated USD for one call, or None if usage or a price for this model is missing."""
    p = prices.get(model)
    if usage is None or p is None:
        return None
    return usage.input_tokens / 1_000_000 * p["input_per_mtok"] + usage.output_tokens / 1_000_000 * p["output_per_mtok"]
