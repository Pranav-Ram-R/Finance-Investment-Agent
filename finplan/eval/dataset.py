"""Golden evaluation cases — natural-language prompts + what a correct run means.

Kept deliberately small so a full run is a handful of LLM calls. ``expected_inputs``
holds the rupee/year values the agent should extract (after parse_amount); for a
multi-goal case it carries ``{"num_goals": N}`` instead.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    name: str
    prompt: str
    expected_tool: str
    expected_inputs: dict


GOLDEN: list[EvalCase] = [
    EvalCase(
        name="basic_moderate",
        prompt=(
            "I have 2 lakh now, can invest 15000 a month for 12 years, "
            "my goal is 50 lakh, moderate risk."
        ),
        expected_tool="generate_plan",
        expected_inputs={
            "initial": 200_000,
            "monthly": 15_000,
            "years": 12,
            "goal": 5_000_000,
        },
    ),
    EvalCase(
        name="crore_aggressive_no_lumpsum",
        prompt="Nothing upfront, 25k per month, 20 years, target 1 crore, aggressive.",
        expected_tool="generate_plan",
        expected_inputs={
            "initial": 0,
            "monthly": 25_000,
            "years": 20,
            "goal": 10_000_000,
        },
    ),
    EvalCase(
        name="conservative_short_horizon",
        prompt=(
            "I can put 5 lakh now and 10k monthly for 5 years, I want 12 lakh, low risk."
        ),
        expected_tool="generate_plan",
        expected_inputs={
            "initial": 500_000,
            "monthly": 10_000,
            "years": 5,
            "goal": 1_200_000,
        },
    ),
    EvalCase(
        name="two_goals",
        prompt=(
            "I have two goals and nothing upfront: 50 lakh for retirement in 20 years "
            "putting in 30k a month, and 10 lakh for a car in 5 years at 12k a month."
        ),
        expected_tool="generate_multi_goal_plan",
        expected_inputs={"num_goals": 2},
    ),
]
