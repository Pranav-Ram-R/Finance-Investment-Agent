"""FastAPI service exposing FinPlan's deterministic engine and LLM agent over HTTP.

Two classes of endpoint, mirroring the project's split:
  * ``/plan`` and ``/plan/multi`` — **deterministic**; the Python engine computes
    every figure, so they need **no LLM key** (a public demo can't burn free-tier
    quota). yfinance does blocking network I/O, so the engine runs in a thread to
    keep the async event loop free.
  * ``/chat`` — drives the LangChain agent and therefore requires a provider key;
    returns the reply plus the tool-call trace and captured tool outputs.

Run:  uvicorn finplan.api.main:app --reload   →   http://localhost:8000/docs
"""
from __future__ import annotations

import asyncio
import json
import os
from functools import lru_cache
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from finplan.agent.planner_agent import build_agent, content_to_text
from finplan.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MultiGoalRequest,
    PlanRequest,
    PlanResponse,
)
from finplan.config import describe_config
from finplan.planner import generate_multi_goal_plan as _generate_multi_goal_plan
from finplan.planner import generate_plan as _generate_plan

VERSION = "1.0.0"

app = FastAPI(
    title="FinPlan API",
    version=VERSION,
    description=(
        "Goal-based investment planning engine + LLM agent for Indian retail "
        "investors. `/plan` and `/plan/multi` are fully deterministic and need no "
        "LLM key; `/chat` drives the conversational agent (requires a provider key)."
    ),
)


# --------------------------------------------------------------------------- #
# Meta
# --------------------------------------------------------------------------- #
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send browsers straight to the interactive OpenAPI docs."""
    return RedirectResponse(url="/docs")


@app.get("/healthz", response_model=HealthResponse, tags=["meta"])
def healthz() -> HealthResponse:
    """Liveness probe — no network, no LLM. Reports the resolved model config."""
    return HealthResponse(status="ok", version=VERSION, models=describe_config())


# --------------------------------------------------------------------------- #
# Deterministic planning (no API key required)
# --------------------------------------------------------------------------- #
@app.post("/plan", response_model=PlanResponse, tags=["planning"])
async def create_plan(req: PlanRequest) -> dict:
    """Build a complete goal-based plan. Deterministic — needs no LLM key.

    Money fields accept text like ``"50 lakh"``; they are normalized to rupees by
    the request model before this handler runs.
    """
    # generate_plan fetches market data (blocking I/O), so run it off the event loop.
    return await asyncio.to_thread(
        _generate_plan, req.initial, req.monthly, req.years, req.goal, req.risk_tolerance
    )


@app.post("/plan/multi", tags=["planning"])
async def create_multi_goal_plan(req: MultiGoalRequest) -> dict:
    """Plan two or more goals independently and return combined totals."""
    goals = [g.model_dump() for g in req.goals]
    return await asyncio.to_thread(_generate_multi_goal_plan, goals)


# --------------------------------------------------------------------------- #
# Conversational agent (requires an LLM provider key)
# --------------------------------------------------------------------------- #
def _has_key() -> bool:
    """True only if a real provider key is set (rejects the .env.example placeholder)."""
    return any(
        (os.getenv(k) or "") and "your-" not in (os.getenv(k) or "")
        for k in ("GOOGLE_API_KEY", "GROQ_API_KEY")
    )


@lru_cache(maxsize=1)
def _agent():
    """Build the agent once and reuse it (same pattern as the Streamlit app)."""
    return build_agent()


def _capture_turn(messages: list) -> tuple[list[str], dict[str, Any]]:
    """Extract the latest turn's tool-call trace and parsed tool outputs.

    ``invoke`` returns the whole checkpointed history; scope to everything from the
    last human message onward so stale outputs from earlier turns are excluded
    (the same slicing the Streamlit UI does).
    """
    last_user = 0
    for i, m in enumerate(messages):
        if m.__class__.__name__ == "HumanMessage" or getattr(m, "type", None) == "human":
            last_user = i

    trace: list[str] = []
    captured: dict[str, Any] = {}
    for m in messages[last_user:]:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                trace.append(f"→ {tc['name']}({json.dumps(tc['args'])})")
        elif m.__class__.__name__ == "ToolMessage":
            try:
                captured[m.name] = json.loads(m.content)
            except Exception:  # noqa: BLE001 - non-JSON tool output is fine to skip
                pass
            snippet = str(m.content)
            trace.append(f"    {m.name} ⇒ {snippet[:150]}{'…' if len(snippet) > 150 else ''}")
    return trace, captured


def _run_agent_turn(message: str, thread_id: str) -> ChatResponse:
    result = _agent().invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    trace, captured = _capture_turn(result["messages"])
    reply = content_to_text(result["messages"][-1].content)
    return ChatResponse(reply=reply, trace=trace, tool_outputs=captured)


@app.post("/chat", response_model=ChatResponse, tags=["agent"])
async def chat(req: ChatRequest) -> ChatResponse:
    """One conversational turn with the planning agent. Requires an LLM provider key."""
    if not _has_key():
        raise HTTPException(
            status_code=503,
            detail=(
                "No LLM provider key set. Add GROQ_API_KEY or GOOGLE_API_KEY to run "
                "the agent. The deterministic /plan endpoint works without one."
            ),
        )
    # The agent call is blocking (LLM + tools); run it off the event loop.
    return await asyncio.to_thread(_run_agent_turn, req.message, req.thread_id)
