"""Tests for the pure stats logic (no network) — especially corrupt-tick handling."""
import numpy as np
import pandas as pd

from finplan.tools.market_data import compute_asset_stats


def _price_series(values, start="2016-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx)


def test_corrupt_tick_is_filtered_from_volatility():
    rng = np.random.default_rng(0)
    prices = 100 * (1 + rng.normal(0, 0.005, 750)).cumprod()
    series = _price_series(prices)
    series.iloc[400] *= 100  # inject an unadjusted-split style glitch

    stats = compute_asset_stats(series, "TEST")

    assert stats["cleaned_outliers"] >= 1
    assert stats["volatility"] < 0.5  # glitch did not poison the estimate


def test_clean_series_flags_no_outliers():
    rng = np.random.default_rng(1)
    prices = 100 * (1 + rng.normal(0.0003, 0.01, 750)).cumprod()

    stats = compute_asset_stats(_price_series(prices), "TEST")

    assert stats["cleaned_outliers"] == 0
    assert 0 < stats["volatility"] < 1


def test_cagr_recovers_a_known_growth_rate():
    # Exactly 10% per year for 5 years (daily-compounded) -> CAGR ~ 10%.
    days = 252 * 5
    daily = 1.10 ** (1 / 252)
    prices = 100 * daily ** np.arange(days)
    stats = compute_asset_stats(_price_series(prices), "TEST")
    assert abs(stats["cagr"] - 0.10) < 0.01


def test_too_short_history_raises():
    import pytest

    with pytest.raises(ValueError):
        compute_asset_stats(_price_series([100.0]), "TEST")
