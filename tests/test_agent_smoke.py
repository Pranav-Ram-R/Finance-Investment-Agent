"""Live smoke test for the agent. Skipped automatically when no API key is set."""
import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # so a key in .env is picked up before the skip check

_HAS_KEY = bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GROQ_API_KEY"))

# Opt-in: this test makes real LLM + network calls, so it stays out of routine
# `pytest` runs. Enable it with:  RUN_LIVE_AGENT_TEST=1 pytest
pytestmark = pytest.mark.skipif(
    os.getenv("RUN_LIVE_AGENT_TEST") != "1" or not _HAS_KEY,
    reason="set RUN_LIVE_AGENT_TEST=1 (and an API key) to run the live agent test",
)


def test_agent_produces_a_plan():
    from finplan.agent.planner_agent import build_agent, run_turn

    agent = build_agent()
    reply = run_turn(
        agent,
        "I have 2 lakh now, can invest 15000 per month for 12 years, "
        "goal is 50 lakh, moderate risk. Build me a plan.",
        thread_id="pytest",
    )
    assert isinstance(reply, str) and len(reply) > 50
