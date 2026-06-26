"""Composite planning pipeline — the full deterministic workflow in one call.

Chaining every step here (not via the LLM) guarantees each step consumes the
previous step's EXACT output, so no intermediate value (expected return,
volatility, allocation) is ever guessed by the model. It also collapses a plan
from ~8 LLM round-trips to ~2 — faster and far easier on free-tier rate limits.
"""
from __future__ import annotations

from functools import lru_cache

from finplan.parsing import format_inr
from finplan.tools.feasibility import check_feasibility, inflation_adjusted_goal
from finplan.tools.market_data import DEFAULT_TICKERS, get_asset_data
from finplan.tools.projection import project_growth
from finplan.tools.risk import (
    assess_risk_profile,
    blended_portfolio_stats,
    recommend_allocation,
)
from finplan.tools.simulation import monte_carlo_simulation
from finplan.tools.tax import apply_ltcg_tax

# Conservative fallbacks if a ticker can't be fetched (e.g. no network).
_FALLBACK = {
    "equity": {"cagr": 0.11, "volatility": 0.16},
    "debt": {"cagr": 0.065, "volatility": 0.03},
    "gold": {"cagr": 0.085, "volatility": 0.13},
}


@lru_cache(maxsize=8)
def _class_stats(asset_class: str, period: str = "10y") -> tuple[float, float, str]:
    """Cached real return/risk for one asset class, with graceful fallback."""
    try:
        d = get_asset_data(DEFAULT_TICKERS[asset_class], period=period)
        return d["cagr"], d["volatility"], f"{DEFAULT_TICKERS[asset_class]} ({d['years']}y live)"
    except Exception:  # noqa: BLE001 - degrade gracefully, never crash
        f = _FALLBACK[asset_class]
        return f["cagr"], f["volatility"], f"assumed ({DEFAULT_TICKERS[asset_class]} unavailable)"


def generate_plan(
    initial: float,
    monthly: float,
    years: float,
    goal: float,
    risk_tolerance: str = "medium",
    inflation: float = 0.06,
) -> dict:
    """Run the entire goal-based plan deterministically and return every figure.

    Each step uses the previous step's exact output — risk profile drives the
    allocation, the allocation + real market data drive the blended return, and
    that return drives the projection, Monte-Carlo, and feasibility checks.
    """
    # 1. Risk profile from horizon + stated tolerance.
    profile_info = assess_risk_profile(years, risk_tolerance)
    profile = profile_info["profile"]

    # 2. Allocation driven by that profile (+ horizon de-risking).
    alloc_info = recommend_allocation(profile, years)
    allocation = alloc_info["allocation"]

    # 3. Real return/risk per asset class (cached; falls back if yfinance fails).
    market_data = {}
    for cls in ("equity", "debt", "gold"):
        cagr, vol, source = _class_stats(cls)
        market_data[cls] = {"cagr": cagr, "volatility": vol, "source": source}

    # 4. Blend the per-class stats by the allocation weights.
    blended = blended_portfolio_stats(allocation, market_data)
    expected_return = blended["expected_return"]
    volatility = blended["volatility_upper_bound"]

    # 5. Everything downstream consumes the SAME blended return/volatility — this
    #    chaining is the whole reason the pipeline lives in Python, not the LLM.
    projection = project_growth(initial, monthly, years, expected_return)
    mc = monte_carlo_simulation(initial, monthly, years, expected_return, volatility, goal=goal)
    feasibility = check_feasibility(initial, monthly, years, expected_return, goal)
    infl_goal = inflation_adjusted_goal(goal, years, inflation)
    tax = apply_ltcg_tax(projection["future_value"], projection["total_invested"])

    # 6. Pre-format every user-facing number into a string the LLM quotes verbatim
    #    (so the model never regroups digits or rescales lakh/crore itself).
    diff = feasibility["difference"]
    levers = feasibility["to_reach_goal"]
    req_ret = levers["required_annual_return"]
    summary = {
        "expected_return": f"{expected_return * 100:.2f}%",
        "volatility": f"{volatility * 100:.2f}%",
        "projected_corpus": format_inr(projection["future_value"]),
        "goal": format_inr(goal),
        "median_outcome": format_inr(mc["median"]),
        "range_p10_to_p90": f"{format_inr(mc['p10'])} to {format_inr(mc['p90'])}",
        "probability_of_goal": f"{mc['probability_of_reaching_goal'] * 100:.1f}%",
        "status": "on track" if feasibility["on_track"] else "shortfall",
        "gap": format_inr(abs(diff)) + (" surplus" if diff >= 0 else " short"),
        "required_monthly_sip": format_inr(levers["required_monthly"]),
        "required_years": levers["required_years"],
        "required_annual_return": f"{req_ret * 100:.2f}%" if req_ret is not None else "n/a",
        "inflation_adjusted_goal": format_inr(infl_goal),
        "post_tax_corpus": format_inr(tax["post_tax_corpus"]),
        "estimated_ltcg_tax": format_inr(tax["estimated_tax"]),
    }

    return {
        "inputs": {
            "initial": initial, "monthly": monthly, "years": years,
            "goal": goal, "risk_tolerance": risk_tolerance,
        },
        "risk_profile": profile,
        "risk_rationale": profile_info["rationale"],
        "allocation": allocation,
        "allocation_note": alloc_info["note"],
        "market_data": market_data,
        "expected_return": expected_return,
        "volatility": volatility,
        "projection": projection,
        "monte_carlo": mc,
        "feasibility": feasibility,
        "inflation_adjusted_goal": infl_goal,
        "tax": tax,
        "summary": summary,
    }


def generate_multi_goal_plan(goals: list[dict]) -> dict:
    """Plan several goals at once by running :func:`generate_plan` per goal.

    Each goal dict carries the same inputs as a single plan (``initial``,
    ``monthly``, ``years``, ``goal``, optional ``risk_tolerance``). Goals are
    planned INDEPENDENTLY and the contributions/corpora summed — there is no
    shared-budget optimization (kept simple on purpose). Returns the full plan
    per goal plus combined totals the LLM can quote verbatim.
    """
    plans = [
        generate_plan(
            g["initial"], g["monthly"], g["years"], g["goal"],
            g.get("risk_tolerance", "medium"),
        )
        for g in goals
    ]

    total_initial = sum(p["inputs"]["initial"] for p in plans)
    total_monthly = sum(p["inputs"]["monthly"] for p in plans)
    combined_post_tax = sum(p["tax"]["post_tax_corpus"] for p in plans)

    summary = {
        "num_goals": len(plans),
        "total_initial": format_inr(total_initial),
        "total_monthly_sip": format_inr(total_monthly),
        "combined_post_tax_corpus": format_inr(combined_post_tax),
        "all_on_track": all(p["feasibility"]["on_track"] for p in plans),
        "per_goal": [
            {
                "goal": p["summary"]["goal"],
                "monthly_sip": format_inr(p["inputs"]["monthly"]),
                "projected_corpus": p["summary"]["projected_corpus"],
                "post_tax_corpus": p["summary"]["post_tax_corpus"],
                "status": p["summary"]["status"],
            }
            for p in plans
        ],
    }

    return {"goals": plans, "summary": summary}
