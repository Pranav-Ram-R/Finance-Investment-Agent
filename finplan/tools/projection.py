"""Growth-projection tools — exact future-value math for lump sum + SIP.

All arithmetic lives here (never in the LLM) so projections are numerically
exact and auditable. Uses monthly compounding; SIP contributions are assumed
at month end (ordinary annuity).
"""
from __future__ import annotations


def _monthly_rate(annual_return: float) -> float:
    """Convert an annual return to its equivalent monthly compounding rate."""
    # 12th root, NOT annual/12: this way 12 months compound back to exactly
    # annual_return (e.g. 12% -> 0.949%/mo, not 1%/mo).
    return (1 + annual_return) ** (1 / 12) - 1


def _fv_months(initial: float, monthly: float, months: int, monthly_rate: float) -> float:
    """Future value after ``months`` given a per-month rate (unrounded)."""
    fv_lump = initial * (1 + monthly_rate) ** months
    if monthly_rate == 0:
        fv_sip = monthly * months
    else:
        fv_sip = monthly * (((1 + monthly_rate) ** months - 1) / monthly_rate)
    return fv_lump + fv_sip


def future_value(initial: float, monthly: float, years: float, annual_return: float) -> float:
    """Exact (unrounded) future value of a lump sum plus monthly SIP.

    Shared by :func:`project_growth` and the feasibility solvers, so there is a
    single source of truth for the math.
    """
    months = int(round(years * 12))
    return _fv_months(initial, monthly, months, _monthly_rate(annual_return))


def project_growth(
    initial: float,
    monthly: float,
    years: float,
    annual_return: float,
) -> dict:
    """Project the future value of a lump sum plus a monthly SIP.

    Parameters
    ----------
    initial:
        Lump-sum amount invested today.
    monthly:
        Recurring monthly contribution (SIP).
    years:
        Investment horizon in years.
    annual_return:
        Assumed annual return as a decimal (e.g. ``0.12`` for 12%).

    Returns
    -------
    dict
        ``future_value``, a lump/SIP breakdown, ``total_invested``,
        ``total_gain``, the assumed return, and a year-by-year ``trajectory``.
    """
    months = int(round(years * 12))
    r = _monthly_rate(annual_return)

    fv_lump = initial * (1 + r) ** months
    fv_sip = monthly * months if r == 0 else monthly * (((1 + r) ** months - 1) / r)
    total = fv_lump + fv_sip
    total_invested = initial + monthly * months

    trajectory = [
        {"year": y, "value": round(_fv_months(initial, monthly, y * 12, r), 2)}
        for y in range(1, int(years) + 1)
    ]

    return {
        "future_value": round(total, 2),
        "from_lumpsum": round(fv_lump, 2),
        "from_sip": round(fv_sip, 2),
        "total_invested": round(total_invested, 2),
        "total_gain": round(total - total_invested, 2),
        "assumed_annual_return": annual_return,
        "trajectory": trajectory,
    }
