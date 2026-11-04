import pytest

from agent.config import ROOT
from agent.cost import estimate_cost, load_prices
from agent.llm.base import Usage

PRICES = {"m": {"input_per_mtok": 0.10, "output_per_mtok": 0.50}}


def test_estimate_cost():
    assert estimate_cost(Usage(1_000_000, 0), "m", PRICES) == pytest.approx(0.10)
    assert estimate_cost(Usage(0, 1_000_000), "m", PRICES) == pytest.approx(0.50)
    assert estimate_cost(Usage(3000, 400), "m", PRICES) == pytest.approx(0.0003 + 0.0002)


def test_no_estimate_without_price_or_usage():
    assert estimate_cost(Usage(10, 10), "unpriced", PRICES) is None
    assert estimate_cost(None, "m", PRICES) is None


def test_load_prices(tmp_path):
    path = tmp_path / "prices.toml"
    path.write_text('[models."m"]\ninput_per_mtok = 1.5\noutput_per_mtok = 7\n')
    assert load_prices(path) == {"m": {"input_per_mtok": 1.5, "output_per_mtok": 7}}


def test_load_prices_rejects_bad_values(tmp_path):
    path = tmp_path / "prices.toml"
    path.write_text('[models."m"]\ninput_per_mtok = "cheap"\noutput_per_mtok = 7\n')
    with pytest.raises(ValueError):
        load_prices(path)


def test_missing_prices_file_means_no_prices(tmp_path):
    assert load_prices(tmp_path / "absent.toml") == {}


def test_shipped_prices_file_has_no_active_prices():
    # Prices ship commented out until the user verifies them.
    assert load_prices(ROOT / "prices.toml") == {}
