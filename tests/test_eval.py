"""Eval-harness tests — the pure scoring functions, offline (no LLM)."""
from finplan.eval.dataset import GOLDEN, EvalCase
from finplan.eval.runner import (
    check_inputs,
    check_numeric_grounding,
    check_tool_selection,
    expected_ground_str,
)


def test_tool_selection():
    assert check_tool_selection(["generate_plan"], "generate_plan")
    assert not check_tool_selection(["assess_risk_profile"], "generate_plan")


def test_numeric_grounding():
    assert check_numeric_grounding("Projected corpus is ₹47,83,173 in 12y.", "₹47,83,173")
    assert not check_numeric_grounding("no figure quoted", "₹47,83,173")
    assert not check_numeric_grounding("anything", "")  # empty expected must fail


def test_check_inputs_single_goal():
    case = EvalCase(
        "x", "p", "generate_plan",
        {"initial": 200_000, "monthly": 15_000, "years": 12, "goal": 5_000_000},
    )
    good = {"inputs": {"initial": 200_000.0, "monthly": 15_000.0, "years": 12, "goal": 5_000_000.0}}
    bad = {"inputs": {"initial": 100_000, "monthly": 15_000, "years": 12, "goal": 5_000_000}}
    assert check_inputs(good, case)
    assert not check_inputs(bad, case)
    assert not check_inputs({}, case)          # no inputs at all
    assert not check_inputs("not a dict", case)


def test_check_inputs_multi_goal():
    case = EvalCase("m", "p", "generate_multi_goal_plan", {"num_goals": 2})
    assert check_inputs({"summary": {"num_goals": 2}}, case)
    assert not check_inputs({"summary": {"num_goals": 3}}, case)


def test_expected_ground_str_picks_right_field():
    single_case = EvalCase("x", "p", "generate_plan", {})
    assert expected_ground_str({"summary": {"projected_corpus": "₹47,83,173"}}, single_case) == "₹47,83,173"

    multi_case = EvalCase("m", "p", "generate_multi_goal_plan", {"num_goals": 2})
    assert expected_ground_str(
        {"summary": {"combined_post_tax_corpus": "₹42,72,974"}}, multi_case
    ) == "₹42,72,974"

    # Missing output -> a sentinel that can never appear in a reply (fails grounding).
    assert expected_ground_str({}, single_case) == "\x00"


def test_golden_set_is_wellformed():
    assert len(GOLDEN) >= 3
    for c in GOLDEN:
        assert c.prompt
        assert c.expected_tool in ("generate_plan", "generate_multi_goal_plan")
