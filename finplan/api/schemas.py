"""Pydantic models for the FinPlan API.

Money fields are typed ``float | str`` and normalized to plain rupees by the
existing :func:`finplan.parsing.parse_amount` — so a client may send ``"50 lakh"``,
``"₹15,000"`` or ``200000`` and the unit conversion still happens in Python, never
in an LLM. This mirrors the deterministic-numbers principle used everywhere else.
"""
from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

from finplan.parsing import parse_amount


def _to_rupees(value: Any) -> float:
    """Coerce a number or Indian-money string (e.g. ``"2 lakh"``) to rupees."""
    return parse_amount(value)


# A money amount. The BeforeValidator accepts a number OR text (e.g. "2 lakh")
# at runtime and normalizes it to rupees; the resolved type is a plain float, so
# handlers and the engine receive a number (never a raw string to reparse).
Money = Annotated[float, BeforeValidator(_to_rupees)]


class PlanRequest(BaseModel):
    """Inputs for a single goal-based plan."""

    initial: Money = Field(0, description="Lump sum to invest now. Number or text like '2 lakh'.")
    monthly: Money = Field(0, description="Monthly SIP. Number or text like '15000' / '20k'.")
    years: float = Field(..., gt=0, description="Investment horizon in years.")
    goal: Money = Field(..., description="Target corpus. Number or text like '50 lakh'.")
    risk_tolerance: str = Field(
        "medium", description="low / medium / high (or conservative / moderate / aggressive)."
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "initial": "2 lakh",
                "monthly": "15000",
                "years": 12,
                "goal": "50 lakh",
                "risk_tolerance": "medium",
            }
        }
    )


class GoalItem(BaseModel):
    """One goal within a multi-goal request."""

    initial: Money = Field(0, description="Lump sum for this goal.")
    monthly: Money = Field(0, description="Monthly SIP for this goal.")
    years: float = Field(..., gt=0, description="Horizon in years for this goal.")
    goal: Money = Field(..., description="Target corpus for this goal.")
    risk_tolerance: str = Field("medium", description="low / medium / high.")


class MultiGoalRequest(BaseModel):
    """Two or more goals planned independently and aggregated."""

    goals: list[GoalItem] = Field(..., min_length=1)


class ChatRequest(BaseModel):
    """One natural-language turn against the LLM agent."""

    message: str = Field(..., description="The user's message, in plain language.")
    thread_id: str = Field(
        "api-default", description="Conversation id; reuse it to preserve memory across turns."
    )


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #
class HealthResponse(BaseModel):
    status: str
    version: str
    models: dict[str, str]


class PlanSummary(BaseModel):
    """The pre-formatted, ready-to-display figures (strings + one nullable float)."""

    expected_return: str
    volatility: str
    projected_corpus: str
    goal: str
    median_outcome: str
    range_p10_to_p90: str
    probability_of_goal: str
    status: str
    gap: str
    required_monthly_sip: str
    required_years: float | None
    required_annual_return: str
    inflation_adjusted_goal: str
    post_tax_corpus: str
    estimated_ltcg_tax: str


class PlanResponse(BaseModel):
    """Full plan. Declares the headline fields for the OpenAPI schema; the rest of
    the engine's output (projection, monte_carlo, feasibility, market_data, tax…)
    is passed through via ``extra="allow"`` so nothing is dropped."""

    model_config = ConfigDict(extra="allow")

    risk_profile: str
    allocation: dict[str, float]
    expected_return: float
    volatility: float
    summary: PlanSummary


class ChatResponse(BaseModel):
    reply: str
    trace: list[str]
    tool_outputs: dict[str, Any]
