"""Tests for the composite planner — especially that values are chained, not guessed."""
import finplan.planner as planner
from finplan.planner import generate_multi_goal_plan, generate_plan

# Fixed per-class stats so the test is deterministic and offline.
_FIXED = {
    "equity": (0.12, 0.16, "fixed"),
    "debt": (0.07, 0.03, "fixed"),
    "gold": (0.08, 0.13, "fixed"),
}


def test_generate_plan_chains_values(monkeypatch):
    monkeypatch.setattr(planner, "_class_stats", lambda cls, period="10y": _FIXED[cls])
    plan = generate_plan(200_000, 15_000, 12, 5_000_000, "moderate")

    # moderate + 12y -> 60/30/10
    assert plan["allocation"] == {"equity": 60, "debt": 30, "gold": 10}

    # blended return = 0.6*0.12 + 0.3*0.07 + 0.1*0.08 = 0.101
    assert plan["expected_return"] == 0.101

    # the projection MUST use the chained blended return, not a round-number guess
    assert plan["projection"]["assumed_annual_return"] == 0.101
    assert plan["monte_carlo"]["goal"] == 5_000_000
    assert "required_monthly" in plan["feasibility"]["to_reach_goal"]
    assert plan["inflation_adjusted_goal"] > plan["inputs"]["goal"]

    # LTCG tax is computed from the projection and surfaced in the summary
    assert plan["tax"]["post_tax_corpus"] <= plan["projection"]["future_value"]

    # pre-formatted, ready-to-quote display strings
    s = plan["summary"]
    assert s["projected_corpus"].startswith("₹")
    assert s["required_annual_return"].endswith("%")
    assert s["status"] in ("on track", "shortfall")
    assert s["post_tax_corpus"].startswith("₹")
    assert s["estimated_ltcg_tax"].startswith("₹")


def test_generate_plan_respects_risk_tolerance(monkeypatch):
    monkeypatch.setattr(planner, "_class_stats", lambda cls, period="10y": _FIXED[cls])
    conservative = generate_plan(0, 10_000, 12, 1_000_000, "low")
    aggressive = generate_plan(0, 10_000, 12, 1_000_000, "high")
    assert conservative["allocation"]["equity"] < aggressive["allocation"]["equity"]


def test_multi_goal_aggregates_independent_plans(monkeypatch):
    monkeypatch.setattr(planner, "_class_stats", lambda cls, period="10y": _FIXED[cls])
    goals = [
        {"initial": 200_000, "monthly": 15_000, "years": 12, "goal": 5_000_000, "risk_tolerance": "moderate"},
        {"initial": 0, "monthly": 5_000, "years": 5, "goal": 500_000, "risk_tolerance": "low"},
    ]
    multi = generate_multi_goal_plan(goals)

    assert len(multi["goals"]) == 2
    # totals are exactly the sum of the two independent single-goal plans
    one = generate_plan(200_000, 15_000, 12, 5_000_000, "moderate")
    two = generate_plan(0, 5_000, 5, 500_000, "low")
    expected_post_tax = one["tax"]["post_tax_corpus"] + two["tax"]["post_tax_corpus"]
    from finplan.parsing import format_inr
    assert multi["summary"]["total_monthly_sip"] == format_inr(20_000)
    assert multi["summary"]["combined_post_tax_corpus"] == format_inr(expected_post_tax)
    assert multi["summary"]["num_goals"] == 2
