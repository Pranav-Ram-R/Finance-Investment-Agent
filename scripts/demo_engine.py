"""End-to-end demo of the deterministic engine (no LLM, no API key needed).

Chains every tool the way the agent will: risk profile -> allocation ->
real market data -> blended stats -> projection -> Monte Carlo -> feasibility.

Run from the project root:
    python -m scripts.demo_engine
"""
from __future__ import annotations

import sys

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

try:  # make ₹ / unicode safe on any console
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

# Conservative fallbacks if a ticker can't be fetched (e.g. no network).
FALLBACK = {
    "equity": {"cagr": 0.11, "volatility": 0.16},
    "debt": {"cagr": 0.065, "volatility": 0.03},
    "gold": {"cagr": 0.085, "volatility": 0.13},
}


def gather_asset_stats(period: str = "10y") -> dict:
    """Fetch real return/risk per asset class, falling back to assumptions."""
    # Mirrors planner._class_stats: try live data per class, but never let a
    # missing ticker or offline run break the demo — fall back to assumed numbers.
    stats = {}
    for cls in ("equity", "debt", "gold"):
        ticker = DEFAULT_TICKERS[cls]
        try:
            d = get_asset_data(ticker, period=period)
            stats[cls] = {
                "cagr": d["cagr"],
                "volatility": d["volatility"],
                "source": f"{ticker} ({d['years']}y live)",
            }
        except Exception:  # noqa: BLE001 - graceful degradation
            stats[cls] = {**FALLBACK[cls], "source": f"assumed ({ticker} unavailable)"}
    return stats


def main() -> None:
    # Walks the same 9-step pipeline as planner.generate_plan, but prints each
    # step so you can see the engine working with NO LLM and NO API key.
    # ---- user inputs ----
    initial, monthly, years, goal, tolerance = 200_000, 15_000, 12, 5_000_000, "medium"
    print(
        f"INPUT: Rs {initial:,} lump + Rs {monthly:,}/mo for {years}y | "
        f"goal Rs {goal:,} | tolerance={tolerance}\n"
    )

    prof = assess_risk_profile(years, tolerance)
    print(f"1) RISK PROFILE : {prof['profile']}  ({prof['rationale']})")

    alloc = recommend_allocation(prof["profile"], years)
    print(f"2) ALLOCATION   : {alloc['allocation']}  ({alloc['note']})")

    stats = gather_asset_stats()
    print("3) MARKET DATA  :")
    for cls, s in stats.items():
        print(f"     {cls:7s} {s['cagr']*100:5.2f}% CAGR  {s['volatility']*100:5.2f}% vol   [{s['source']}]")

    blended = blended_portfolio_stats(alloc["allocation"], stats)
    er, vol = blended["expected_return"], blended["volatility_upper_bound"]
    print(f"4) BLENDED      : {er*100:.2f}% expected return, <= {vol*100:.2f}% volatility")

    proj = project_growth(initial, monthly, years, er)
    print(
        f"5) PROJECTION   : Rs {proj['future_value']:,.0f} "
        f"(invested Rs {proj['total_invested']:,.0f}, gain Rs {proj['total_gain']:,.0f})"
    )

    mc = monte_carlo_simulation(initial, monthly, years, er, vol, goal=goal)
    print(
        f"6) MONTE CARLO  : p10 Rs {mc['p10']:,.0f} | median Rs {mc['median']:,.0f} | "
        f"p90 Rs {mc['p90']:,.0f}  ->  {mc['probability_of_reaching_goal']*100:.1f}% chance of goal"
    )

    feas = check_feasibility(initial, monthly, years, er, goal)
    status = "ON TRACK" if feas["on_track"] else "SHORTFALL"
    o = feas["to_reach_goal"]
    req_ret = f"{o['required_annual_return']*100:.1f}%" if o["required_annual_return"] else "n/a"
    print(f"7) FEASIBILITY  : {status} (difference Rs {feas['difference']:,.0f})")
    print(
        f"     to reach goal -> SIP Rs {o['required_monthly']:,.0f}/mo  OR  "
        f"{o['required_years']}y horizon  OR  {req_ret} annual return"
    )

    infl = inflation_adjusted_goal(goal, years)
    print(f"8) INFLATION    : Rs {goal:,} today ~= Rs {infl:,.0f} in {years}y at 6% (the real target)")

    tax = apply_ltcg_tax(proj["future_value"], proj["total_invested"])
    print(
        f"9) AFTER-TAX    : Rs {tax['post_tax_corpus']:,.0f} post-tax "
        f"(LTCG Rs {tax['estimated_tax']:,.0f} = 12.5% over Rs 1,25,000 exemption)"
    )


if __name__ == "__main__":
    main()
