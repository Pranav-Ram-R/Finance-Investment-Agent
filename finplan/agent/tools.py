"""LangChain tools — the bridge between the LLM and the deterministic engine.

The primary tool is ``generate_plan``: a single deterministic call that runs the
whole pipeline with every value correctly chained (so the model never supplies
or guesses financial figures). The granular tools remain for "what-if" tweaks.

Money amounts are parsed in Python via :func:`finplan.parsing.parse_amount`, so
the model can pass "50 lakh" verbatim and never has to do unit conversion.
"""
from __future__ import annotations

from langchain_core.tools import tool

from finplan.memory import store
from finplan.parsing import parse_amount
from finplan.planner import _class_stats
from finplan.planner import generate_multi_goal_plan as _generate_multi_goal_plan
from finplan.planner import generate_plan as _generate_plan
from finplan.tools.feasibility import check_feasibility as _check_feasibility
from finplan.tools.feasibility import inflation_adjusted_goal as _inflation_adjusted_goal
from finplan.tools.news import get_news_sentiment as _get_news_sentiment
from finplan.tools.projection import future_value
from finplan.tools.projection import project_growth as _project_growth
from finplan.tools.risk import assess_risk_profile as _assess_risk_profile
from finplan.tools.risk import blended_portfolio_stats
from finplan.tools.risk import recommend_allocation as _recommend_allocation
from finplan.tools.simulation import monte_carlo_simulation
from finplan.tools.tax import apply_ltcg_tax as _apply_ltcg_tax

Amount = float | str  # money args accept a number or text like "50 lakh"

# Identifies the active user for persistent memory. The app can override this
# (e.g. from a Streamlit session); single-user default keeps demos simple.
CURRENT_USER = "default_user"


# --------------------------------------------------------------------------- #
# Primary tool — build a complete plan in one deterministic call
# --------------------------------------------------------------------------- #
@tool
def generate_plan(
    initial: Amount, monthly: Amount, years: float, goal: Amount, risk_tolerance: str = "medium"
) -> dict:
    """Build a COMPLETE goal-based investment plan in one call — use this first.

    Pass the user's raw inputs. For money amounts, pass them EXACTLY as the user
    said them, including words like 'lakh'/'crore' (e.g. initial="2 lakh",
    monthly="15000", goal="50 lakh") — the tool converts to rupees, so never do
    that arithmetic yourself. ``risk_tolerance`` is 'low'/'medium'/'high' (or
    conservative/moderate/aggressive).

    Runs the entire pipeline — risk profiling, allocation, REAL market data,
    blended return/risk, projection, Monte-Carlo, feasibility, inflation — and
    returns every figure, each computed from the previous step.
    """
    return _generate_plan(
        parse_amount(initial), parse_amount(monthly), years, parse_amount(goal), risk_tolerance
    )


@tool
def generate_multi_goal_plan(goals: list[dict]) -> dict:
    """Plan MULTIPLE goals at once — use when the user has two or more goals.

    Pass a list of goal objects, each with: initial, monthly, years, goal, and
    risk_tolerance ('low'/'medium'/'high'). Money may be text like "5 lakh".
    Example:
        [{"initial": "2 lakh", "monthly": "15000", "years": 12, "goal": "50 lakh", "risk_tolerance": "medium"},
         {"initial": 0, "monthly": "5000", "years": 5, "goal": "5 lakh", "risk_tolerance": "low"}]

    Builds a full plan for each goal and returns them plus combined totals
    (total monthly SIP, combined post-tax corpus). Each goal is planned
    independently — there is no shared-budget split.
    """
    # Parse each goal's money fields here (the wrapper's job) so the inner
    # planner receives plain floats, exactly like the single-goal path.
    parsed = [
        {
            "initial": parse_amount(g["initial"]),
            "monthly": parse_amount(g["monthly"]),
            "years": g["years"],
            "goal": parse_amount(g["goal"]),
            "risk_tolerance": g.get("risk_tolerance", "medium"),
        }
        for g in goals
    ]
    return _generate_multi_goal_plan(parsed)


# --------------------------------------------------------------------------- #
# Granular tools (for "what-if" follow-ups)
# --------------------------------------------------------------------------- #
@tool
def assess_risk_profile(horizon_years: float, stated_tolerance: str = "medium") -> dict:
    """Determine the investor's risk profile (conservative/moderate/aggressive)
    from a time horizon (years) and stated tolerance ('low'/'medium'/'high')."""
    return _assess_risk_profile(horizon_years, stated_tolerance)


@tool
def recommend_allocation(profile: str, horizon_years: float) -> dict:
    """Recommend an equity/debt/gold split for a risk profile and horizon.
    ``profile`` is 'conservative', 'moderate', or 'aggressive'."""
    return _recommend_allocation(profile, horizon_years)


@tool
def get_portfolio_market_data(equity_pct: int, debt_pct: int, gold_pct: int) -> dict:
    """Get the blended expected annual return and volatility for an allocation.

    Pass equity/debt/gold percentages (must sum to 100). Fetches REAL 10-year
    return/risk per asset class and blends them.
    """
    allocation = {"equity": equity_pct, "debt": debt_pct, "gold": gold_pct}
    per_class, sources = {}, {}
    for cls in ("equity", "debt", "gold"):
        cagr, vol, src = _class_stats(cls)
        per_class[cls] = {"cagr": cagr, "volatility": vol}
        sources[cls] = src
    blended = blended_portfolio_stats(allocation, per_class)
    return {"allocation": allocation, "per_class": per_class, "sources": sources, **blended}


@tool
def project_growth(initial: Amount, monthly: Amount, years: float, annual_return: float) -> dict:
    """Project the future value of a lump sum plus monthly SIP. ``annual_return``
    is a decimal (e.g. 0.106). Money amounts may be text like "2 lakh"."""
    return _project_growth(parse_amount(initial), parse_amount(monthly), years, annual_return)


@tool
def run_monte_carlo(
    initial: Amount, monthly: Amount, years: float,
    annual_return: float, annual_volatility: float, goal: Amount,
) -> dict:
    """Monte-Carlo simulation of the plan. Returns p10/median/p90, mean, and the
    probability of reaching ``goal`` — use it to communicate the RANGE of outcomes."""
    return monte_carlo_simulation(
        parse_amount(initial), parse_amount(monthly), years,
        annual_return, annual_volatility, goal=parse_amount(goal),
    )


@tool
def check_feasibility(
    initial: Amount, monthly: Amount, years: float, annual_return: float, goal: Amount
) -> dict:
    """Check whether the plan reaches the goal and solve for the three levers to
    close any gap: required monthly SIP, required horizon, required annual return."""
    return _check_feasibility(
        parse_amount(initial), parse_amount(monthly), years, annual_return, parse_amount(goal)
    )


@tool
def inflation_adjusted_goal(goal: Amount, years: float, inflation: float = 0.06) -> dict:
    """Convert a goal in today's money into the future rupees actually needed
    (default 6% inflation). ``goal`` may be text like "50 lakh"."""
    g = parse_amount(goal)
    return {"nominal_goal_today": g, "years": years, "inflation": inflation,
            "future_value_needed": _inflation_adjusted_goal(g, years, inflation)}


@tool
def estimate_ltcg_tax(corpus: Amount, total_invested: Amount) -> dict:
    """Estimate Long-Term Capital Gains tax and the post-tax corpus on redemption.

    Uses the equity LTCG rule (12.5% on gains above a ₹1.25 lakh exemption). Use
    for what-if questions about take-home value after tax. Money amounts may be
    text like "50 lakh". This is a simplified estimate, not tax advice."""
    return _apply_ltcg_tax(parse_amount(corpus), parse_amount(total_invested))


@tool
def get_news_sentiment(ticker: str = "^NSEI") -> dict:
    """Read the current market "mood" from recent headlines for a ticker.

    ``ticker`` is a Yahoo Finance symbol (default '^NSEI', the Nifty 50; e.g.
    'RELIANCE.NS' for a stock). Returns a positive/neutral/negative label, a
    score, and the headlines used. This is QUALITATIVE context only — it does
    NOT change any projection, return, or allocation. Use it when the user asks
    about current market sentiment or news, and present it as soft context."""
    return _get_news_sentiment(ticker)


# --------------------------------------------------------------------------- #
# Memory tools (persistent across sessions)
# --------------------------------------------------------------------------- #
@tool
def save_plan(
    initial: Amount, monthly: Amount, years: float, goal: Amount, profile: str,
    equity_pct: int, debt_pct: int, gold_pct: int,
    expected_return: float, projected_value: Amount,
) -> dict:
    """Save the finalized plan so it can be recalled in a future session."""
    return store.save_plan(
        CURRENT_USER,
        initial=parse_amount(initial), monthly=parse_amount(monthly), years=years,
        goal=parse_amount(goal), profile=profile,
        allocation={"equity": equity_pct, "debt": debt_pct, "gold": gold_pct},
        expected_return=expected_return, projected_value=parse_amount(projected_value),
    )


@tool
def get_saved_plan() -> dict:
    """Retrieve the user's previously saved plan from an earlier session, if any."""
    return store.get_plan(CURRENT_USER) or {"status": "no_saved_plan"}


@tool
def log_contribution(amount_invested_so_far: Amount, current_portfolio_value: Amount) -> dict:
    """Record the user's actual progress: total invested so far and current value."""
    return store.log_contribution(
        CURRENT_USER, parse_amount(amount_invested_so_far), parse_amount(current_portfolio_value)
    )


@tool
def check_progress() -> dict:
    """Compare the user's latest logged value against the saved plan's expected
    path — tells them whether they are on track, ahead, or behind, with the gap."""
    plan = store.get_plan(CURRENT_USER)
    if plan is None:
        return {"status": "no_saved_plan"}
    history = store.get_progress(CURRENT_USER)
    if not history:
        return {"status": "no_progress_logged"}

    latest = history[-1]
    invested, current = latest["amount"], latest["current_value"]
    monthly = plan["monthly"] or 0
    # Infer how far along the plan we are from how much has been invested:
    # (total invested - initial lump) / monthly SIP ≈ months elapsed.
    elapsed_months = max(0, round((invested - plan["initial"]) / monthly)) if monthly else 0
    # Re-run the SAME future-value math to get where the plan EXPECTED them to be by now.
    expected = future_value(plan["initial"], monthly, elapsed_months / 12, plan["expected_return"])

    return {
        "invested_so_far": round(invested, 2),
        "current_value": round(current, 2),
        "expected_value_now": round(expected, 2),
        "difference": round(current - expected, 2),
        "elapsed_months": elapsed_months,
        # 5% tolerance band — markets wobble, so "slightly behind" still counts as on track.
        "on_track": current >= expected * 0.95,
    }


ALL_TOOLS = [
    generate_plan,            # primary: full plan in one deterministic call
    generate_multi_goal_plan, # primary: several goals at once, aggregated
    assess_risk_profile,
    recommend_allocation,
    get_portfolio_market_data,
    project_growth,
    run_monte_carlo,
    check_feasibility,
    inflation_adjusted_goal,
    estimate_ltcg_tax,
    get_news_sentiment,
    save_plan,
    get_saved_plan,
    log_contribution,
    check_progress,
]
