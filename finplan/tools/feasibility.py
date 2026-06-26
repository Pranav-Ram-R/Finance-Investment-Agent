"""Feasibility tools — does the plan reach the goal, and if not, what closes the gap?

The solvers invert the future-value math to find the *required* monthly SIP,
horizon, or return. This is the core advisory reasoning, computed exactly.
"""
from __future__ import annotations

from finplan.tools.projection import future_value


def _monthly_rate(annual_return: float) -> float:
    return (1 + annual_return) ** (1 / 12) - 1


def _required_monthly(initial: float, goal: float, years: float, annual_return: float) -> float:
    """SIP needed to hit ``goal`` (holding initial, years, return fixed)."""
    months = int(round(years * 12))
    r = _monthly_rate(annual_return)
    fv_lump = initial * (1 + r) ** months
    needed = goal - fv_lump
    if needed <= 0:  # the lump sum alone already reaches the goal
        return 0.0
    if r == 0:
        return needed / months
    annuity_factor = ((1 + r) ** months - 1) / r
    return needed / annuity_factor


def _required_years(
    initial: float, monthly: float, goal: float, annual_return: float, max_months: int = 1200
) -> float | None:
    """Years needed to hit ``goal`` (holding contributions, return fixed)."""
    r = _monthly_rate(annual_return)
    bal = float(initial)
    if bal >= goal:
        return 0.0
    for m in range(1, max_months + 1):
        bal = bal * (1 + r) + monthly
        if bal >= goal:
            return round(m / 12, 1)
    return None


def _required_return(
    initial: float, monthly: float, years: float, goal: float, lo: float = -0.5, hi: float = 1.0
) -> float | None:
    """Annual return needed to hit ``goal`` (bisection; FV is monotonic in return)."""
    f = lambda ar: future_value(initial, monthly, years, ar) - goal  # noqa: E731
    if f(hi) < 0:
        return None  # unreachable by return alone within a sane ceiling (100%/yr)
    if f(lo) > 0:
        return lo  # reachable even with deeply negative returns
    for _ in range(100):
        mid = (lo + hi) / 2
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 4)


def inflation_adjusted_goal(goal: float, years: float, inflation: float = 0.06) -> float:
    """Future rupees required to match ``goal`` stated in today's money."""
    return round(goal * (1 + inflation) ** years, 2)


def check_feasibility(
    initial: float, monthly: float, years: float, annual_return: float, goal: float
) -> dict:
    """Compare the projected corpus to the goal and solve for ways to close any gap.

    Returns the projected value, the surplus/shortfall, an ``on_track`` flag, and
    three independent levers to reach the goal exactly: the required monthly SIP,
    the required horizon, and the required annual return.
    """
    projected = future_value(initial, monthly, years, annual_return)
    difference = projected - goal

    return {
        "goal": round(goal, 2),
        "projected_value": round(projected, 2),
        "difference": round(difference, 2),  # positive = surplus, negative = shortfall
        "on_track": projected >= goal,
        "to_reach_goal": {
            "required_monthly": round(_required_monthly(initial, goal, years, annual_return), 2),
            "required_years": _required_years(initial, monthly, goal, annual_return),
            "required_annual_return": _required_return(initial, monthly, years, goal),
        },
    }
