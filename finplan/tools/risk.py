"""Risk-profiling and asset-allocation tools.

Pure, rule-based logic so allocations are transparent and explainable. The agent
combines the recommended weights with real return/risk stats (see
:mod:`finplan.tools.market_data`) to estimate portfolio performance.
"""
from __future__ import annotations

_LEVELS = {1: "conservative", 2: "moderate", 3: "aggressive"}
_TOLERANCE = {"low": 1, "medium": 2, "high": 3}

# Accept the natural financial vocabulary an LLM (or user) is likely to use,
# not just our internal enum — robust tools beat brittle ones.
_TOLERANCE_SYNONYMS = {
    "low": "low", "conservative": "low", "cautious": "low", "safe": "low",
    "medium": "medium", "moderate": "medium", "balanced": "medium", "mid": "medium",
    "high": "high", "aggressive": "high", "growth": "high", "risky": "high",
}
_PROFILE_SYNONYMS = {
    "conservative": "conservative", "low": "conservative", "cautious": "conservative",
    "moderate": "moderate", "medium": "moderate", "balanced": "moderate",
    "aggressive": "aggressive", "high": "aggressive", "growth": "aggressive",
}

# Base allocations by risk profile (percent; equity / debt / gold).
_BASE_ALLOCATION = {
    "conservative": {"equity": 30, "debt": 60, "gold": 10},
    "moderate": {"equity": 60, "debt": 30, "gold": 10},
    "aggressive": {"equity": 80, "debt": 10, "gold": 10},
}


def _capacity_from_horizon(horizon_years: float) -> int:
    """Objective risk *capacity* implied by the time horizon."""
    # Longer horizon = more ability to ride out volatility, so higher capacity.
    # <3y: conservative, 3-7y: moderate, >7y: aggressive (1/2/3 map to _LEVELS).
    if horizon_years < 3:
        return 1
    if horizon_years <= 7:
        return 2
    return 3


def assess_risk_profile(horizon_years: float, stated_tolerance: str = "medium") -> dict:
    """Combine risk *capacity* (horizon) and *willingness* (stated) into a profile.

    Uses the prudent rule ``recommended = min(capacity, willingness)`` so we
    never advise more risk than either the horizon or the user supports.
    """
    tol = _TOLERANCE_SYNONYMS.get(stated_tolerance.lower().strip())
    if tol is None:
        raise ValueError(
            f"stated_tolerance {stated_tolerance!r} not recognized; use "
            "low/medium/high (or conservative/moderate/aggressive)"
        )

    # capacity = what the horizon can support; willingness = what the user wants.
    # Take the LOWER of the two — never push more risk than both agree on.
    capacity = _capacity_from_horizon(horizon_years)
    willingness = _TOLERANCE[tol]
    level = min(capacity, willingness)

    return {
        "profile": _LEVELS[level],
        "capacity": _LEVELS[capacity],
        "willingness": _LEVELS[willingness],
        "horizon_years": horizon_years,
        "rationale": (
            f"Horizon of {horizon_years:g}y implies {_LEVELS[capacity]} capacity; "
            f"stated tolerance is {_LEVELS[willingness]}. "
            f"Taking the lower of the two -> {_LEVELS[level]}."
        ),
    }


def recommend_allocation(profile: str, horizon_years: float) -> dict:
    """Return an equity/debt/gold split for a profile, de-risked for short horizons."""
    profile = _PROFILE_SYNONYMS.get(profile.lower().strip(), profile.lower().strip())
    if profile not in _BASE_ALLOCATION:
        raise ValueError(
            f"profile must be one of {list(_BASE_ALLOCATION)} (or low/medium/high)"
        )

    alloc = dict(_BASE_ALLOCATION[profile])

    # Horizon-based equity ceiling: short-term money shouldn't sit in equities
    # (no time to recover from a crash). The ceiling rises with the horizon.
    if horizon_years < 3:
        ceiling = 30
    elif horizon_years < 5:
        ceiling = 50
    elif horizon_years < 7:
        ceiling = 70
    else:
        ceiling = 100

    # If the profile's equity exceeds the ceiling, move the excess into debt
    # so the weights still sum to 100.
    if alloc["equity"] > ceiling:
        moved = alloc["equity"] - ceiling
        alloc["equity"] = ceiling
        alloc["debt"] += moved

    capped = alloc["equity"] == ceiling and ceiling < 100
    return {
        "profile": profile,
        "allocation": alloc,
        "equity_ceiling": ceiling,
        "note": (
            f"Equity capped at {ceiling}% for a {horizon_years:g}y horizon."
            if capped
            else "Standard allocation for this profile."
        ),
    }


def blended_portfolio_stats(allocation: dict, stats_by_class: dict) -> dict:
    """Combine per-asset-class return/risk into portfolio-level estimates.

    Parameters
    ----------
    allocation:
        Percent weights, e.g. ``{"equity": 60, "debt": 30, "gold": 10}``.
    stats_by_class:
        ``{"equity": {"cagr": .., "volatility": ..}, ...}`` (see market_data).

    Notes
    -----
    The volatility figure is a *conservative upper bound*: it assumes perfect
    correlation between asset classes and so ignores the diversification
    benefit. Labelled accordingly so the agent doesn't overstate precision.
    """
    weights = {k: v / 100 for k, v in allocation.items()}
    if abs(sum(weights.values()) - 1.0) > 1e-6:
        raise ValueError("allocation percentages must sum to 100")

    # Expected return is a clean weighted average. Volatility is too — but a
    # weighted average of vols assumes perfect correlation (no diversification
    # benefit), so it's an UPPER BOUND, named accordingly to avoid overstating.
    exp_return = sum(weights[k] * stats_by_class[k]["cagr"] for k in weights)
    vol_upper = sum(weights[k] * stats_by_class[k]["volatility"] for k in weights)

    return {
        "expected_return": round(exp_return, 4),
        "volatility_upper_bound": round(vol_upper, 4),
    }
