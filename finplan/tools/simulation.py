"""Monte-Carlo simulation tool — models return uncertainty, not a single guess.

Simulates many possible market paths to produce a *range* of outcomes
(p10-p90) and the probability of reaching a goal. This is what turns a point
estimate into honest, risk-aware reasoning.
"""
from __future__ import annotations

import numpy as np


def monte_carlo_simulation(
    initial: float,
    monthly: float,
    years: float,
    annual_return: float,
    annual_volatility: float,
    goal: float | None = None,
    n_sims: int = 2000,
    seed: int | None = 42,
) -> dict:
    """Simulate ``n_sims`` investment paths and summarize the outcome distribution.

    Monthly returns are drawn from a normal distribution implied by the annual
    return and volatility. SIP contributions are added at month end (matching
    :func:`finplan.tools.projection.project_growth`).

    Returns percentiles (p10/p25/p50/p75/p90), the mean, and — if ``goal`` is
    given — the probability of reaching it. ``seed`` is fixed by default so
    results are reproducible (important for testing and demos).
    """
    months = int(round(years * 12))
    monthly_mean = (1 + annual_return) ** (1 / 12) - 1
    # √time scaling: variance grows linearly with time, so stdev scales by √12.
    monthly_vol = annual_volatility / np.sqrt(12)

    rng = np.random.default_rng(seed)
    # One big draw of every monthly return for every path (n_sims × months).
    draws = rng.normal(monthly_mean, monthly_vol, size=(n_sims, months))

    balances = np.full(n_sims, float(initial))
    for m in range(months):
        # Vectorized: steps ALL n_sims paths forward one month at once (no per-path loop).
        balances = balances * (1 + draws[:, m]) + monthly

    p10, p25, p50, p75, p90 = (
        float(x) for x in np.percentile(balances, [10, 25, 50, 75, 90])
    )

    result = {
        "p10": round(p10, 2),
        "p25": round(p25, 2),
        "median": round(p50, 2),
        "p75": round(p75, 2),
        "p90": round(p90, 2),
        "mean": round(float(balances.mean()), 2),
        "n_sims": n_sims,
    }
    if goal is not None:
        result["goal"] = goal
        result["probability_of_reaching_goal"] = round(
            float((balances >= goal).mean()), 4
        )
    return result
