"""API tests — offline and deterministic (no network, no LLM key).

The /plan endpoints call the same engine as tests/test_planner.py, so we reuse
that file's trick: monkeypatch ``finplan.planner._class_stats`` to fixed per-class
stats, avoiding any yfinance call. The /chat live path is opt-in (real LLM calls),
gated behind RUN_LIVE_AGENT_TEST exactly like tests/test_agent_smoke.py.
"""
import os

import pytest
from fastapi.testclient import TestClient

import finplan.planner as planner
from finplan.api.main import app

# Same fixed stats as tests/test_planner.py so the chained numbers are identical.
_FIXED = {
    "equity": (0.12, 0.16, "fixed"),
    "debt": (0.07, 0.03, "fixed"),
    "gold": (0.08, 0.13, "fixed"),
}

client = TestClient(app)


@pytest.fixture(autouse=True)
def _offline_stats(monkeypatch):
    # Every asset class returns fixed stats -> no network, fully deterministic.
    monkeypatch.setattr(planner, "_class_stats", lambda cls, period="10y": _FIXED[cls])


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"]
    assert isinstance(body["models"], dict)


def test_plan_happy_path_accepts_text_money():
    r = client.post(
        "/plan",
        json={
            "initial": "2 lakh",
            "monthly": "15000",
            "years": 12,
            "goal": "50 lakh",
            "risk_tolerance": "medium",
        },
    )
    assert r.status_code == 200
    body = r.json()
    # Same chaining the engine test asserts: moderate + 12y -> 60/30/10, blend 0.101.
    assert body["allocation"] == {"equity": 60, "debt": 30, "gold": 10}
    assert body["expected_return"] == pytest.approx(0.101)
    s = body["summary"]
    assert s["projected_corpus"].startswith("₹")
    assert s["status"] in ("on track", "shortfall")
    assert s["post_tax_corpus"].startswith("₹")
    # extra="allow" passes the rest of the engine output through untouched.
    assert "monte_carlo" in body and "feasibility" in body


def test_plan_validation_error_when_goal_missing():
    r = client.post("/plan", json={"monthly": "15000", "years": 12})
    assert r.status_code == 422


def test_multi_goal_plan_aggregates():
    r = client.post(
        "/plan/multi",
        json={
            "goals": [
                {"initial": "2 lakh", "monthly": "15000", "years": 12, "goal": "50 lakh"},
                {"initial": 0, "monthly": "5000", "years": 5, "goal": "5 lakh", "risk_tolerance": "low"},
            ]
        },
    )
    assert r.status_code == 200
    summary = r.json()["summary"]
    assert summary["num_goals"] == 2
    assert summary["total_monthly_sip"].startswith("₹")


def test_chat_without_key_returns_503(monkeypatch):
    # Force both providers to look like the placeholder so no LLM call is attempted.
    for k in ("GROQ_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.setenv(k, "your-placeholder-key")
    r = client.post("/chat", json={"message": "Hi"})
    assert r.status_code == 503


@pytest.mark.skipif(
    not os.getenv("RUN_LIVE_AGENT_TEST"), reason="opt-in: makes real LLM calls"
)
def test_chat_live_turn():
    r = client.post("/chat", json={"message": "Hello", "thread_id": "api-test"})
    assert r.status_code == 200
    assert r.json()["reply"]
