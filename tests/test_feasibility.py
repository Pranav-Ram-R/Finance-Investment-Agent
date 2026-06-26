import math

from finplan.tools.feasibility import check_feasibility, inflation_adjusted_goal
from finplan.tools.projection import future_value


def test_on_track_when_projection_exceeds_goal():
    r = check_feasibility(100_000, 20_000, 10, 0.12, goal=1_000_000)
    assert r["on_track"] is True
    assert r["difference"] > 0


def test_shortfall_is_flagged_and_solved():
    r = check_feasibility(100_000, 10_000, 12, 0.10, goal=5_000_000)
    assert r["on_track"] is False
    assert r["difference"] < 0
    opts = r["to_reach_goal"]
    assert opts["required_monthly"] > 10_000
    assert opts["required_years"] > 12


def test_required_monthly_actually_hits_goal():
    goal = 5_000_000
    req = check_feasibility(100_000, 10_000, 12, 0.10, goal=goal)["to_reach_goal"]["required_monthly"]
    achieved = future_value(100_000, req, 12, 0.10)
    assert math.isclose(achieved, goal, rel_tol=1e-3)


def test_required_return_actually_hits_goal():
    goal = 5_000_000
    req = check_feasibility(100_000, 10_000, 12, 0.10, goal=goal)["to_reach_goal"][
        "required_annual_return"
    ]
    assert req is not None
    achieved = future_value(100_000, 10_000, 12, req)
    assert math.isclose(achieved, goal, rel_tol=1e-3)


def test_inflation_adjusted_goal():
    assert inflation_adjusted_goal(1_000_000, 10, 0.06) == round(1_000_000 * 1.06**10, 2)
