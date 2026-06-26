"""Market-data tool — real historical performance via yfinance.

We derive expected return (CAGR) and risk (annualized volatility) from actual
price history, so the agent reasons over real numbers instead of guessing.

Yahoo's free data occasionally contains corrupt ticks (unadjusted splits, bad
prints). :func:`compute_asset_stats` filters those out so a single glitch can't
poison the volatility estimate — see the unit tests.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252

# No broad index/ETF moves >50% in a day; larger jumps are data errors.
_MAX_SANE_DAILY_RETURN = 0.5

# Handy defaults for an Indian-market plan (Yahoo Finance symbols).
DEFAULT_TICKERS = {
    "equity": "^NSEI",           # Nifty 50 index
    "equity_midcap": "^NSEIMDCP50",
    "gold": "GOLDBEES.NS",       # Nippon gold ETF
    "debt": "LTGILTBEES.NS",     # long-term gilt ETF (realistic bond proxy)
}


@dataclass
class AssetStats:
    """Return/risk summary for one instrument (all rates are decimals)."""

    ticker: str
    name: str
    last_price: float
    cagr: float             # annualized return, e.g. 0.12 == 12%
    volatility: float       # annualized stdev of daily returns
    years: float            # span of history actually used
    samples: int            # number of price observations
    cleaned_outliers: int   # corrupt daily ticks excluded from volatility

    def as_dict(self) -> dict:
        return asdict(self)


def compute_asset_stats(close: pd.Series, ticker: str, name: str | None = None) -> dict:
    """Compute return/risk statistics from a close-price series (pure, no I/O).

    Volatility is computed after removing corrupt ticks (``|daily return| >
    50%``), so unadjusted splits / bad prints don't blow up the estimate.

    Raises
    ------
    ValueError
        If fewer than two usable prices are present.
    """
    close = close.dropna()
    if len(close) < 2:
        raise ValueError(f"Insufficient price history for {ticker!r}")

    span_days = (close.index[-1] - close.index[0]).days
    years = max(span_days / 365.25, 1e-9)
    cagr = (close.iloc[-1] / close.iloc[0]) ** (1 / years) - 1

    daily = close.pct_change().dropna()
    clean = daily[daily.abs() <= _MAX_SANE_DAILY_RETURN]
    volatility = float(clean.std() * np.sqrt(TRADING_DAYS))

    return AssetStats(
        ticker=ticker,
        name=name or ticker,
        last_price=float(close.iloc[-1]),
        cagr=float(cagr),
        volatility=volatility,
        years=round(years, 2),
        samples=int(len(close)),
        cleaned_outliers=int(len(daily) - len(clean)),
    ).as_dict()


def get_asset_data(ticker: str, period: str = "10y") -> dict:
    """Fetch price history for ``ticker`` and return return/risk statistics.

    Parameters
    ----------
    ticker:
        Yahoo Finance symbol, e.g. ``"^NSEI"`` (Nifty 50) or ``"GOLDBEES.NS"``.
    period:
        History window understood by yfinance (``"5y"``, ``"10y"``, ``"max"``…).

    Returns
    -------
    dict
        JSON-friendly stats (see :class:`AssetStats`).

    Raises
    ------
    ValueError
        If no usable price history is returned.
    """
    t = yf.Ticker(ticker)
    hist = t.history(period=period, auto_adjust=True)
    if hist.empty or "Close" not in hist:
        raise ValueError(f"No price data for ticker {ticker!r}")

    # Friendly name is best-effort: .info is an extra (flaky) network call.
    name = ticker
    try:
        name = t.info.get("shortName") or ticker
    except Exception:  # noqa: BLE001 - never fail the tool over a label
        pass

    return compute_asset_stats(hist["Close"], ticker, name)
