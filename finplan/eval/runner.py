"""Scoring functions + the evaluation loop.

The three ``check_*`` functions are pure (no LLM, no I/O) and unit-tested offline.
:func:`run_eval` and :func:`score_case` drive the real agent and therefore need a
provider key; run them via ``python -m finplan.eval``.
"""
from __future__ import annotations

import json
import re
from typing import Any

from finplan.agent.planner_agent import content_to_text
from finplan.eval.dataset import GOLDEN, EvalCase


# --------------------------------------------------------------------------- #
# Pure checks (offline-testable)
# --------------------------------------------------------------------------- #
def check_tool_selection(called_tools: list[str], expected_tool: str) -> bool:
    """Did the agent call the expected primary tool at least once?"""
    return expected_tool in called_tools


def check_numeric_grounding(reply: str, expected_str: str) -> bool:
    """Does the reply quote the engine's pre-formatted figure verbatim?"""
    return bool(expected_str) and expected_str in reply


def check_inputs(tool_output: Any, case: EvalCase) -> bool:
    """Did the tool receive the intended inputs (parsed to rupees / years)?"""
    if not isinstance(tool_output, dict):
        return False
    if case.expected_tool == "generate_multi_goal_plan":
        num = tool_output.get("summary", {}).get("num_goals")
        return num == case.expected_inputs.get("num_goals")
    inputs = tool_output.get("inputs", {})
    for key, expected in case.expected_inputs.items():
        actual = inputs.get(key)
        if actual is None or abs(float(actual) - float(expected)) > 1.0:  # ₹1 tolerance
            return False
    return True


def expected_ground_str(tool_output: Any, case: EvalCase) -> str:
    """The exact display string the reply is expected to quote (from this run)."""
    summary = tool_output.get("summary", {}) if isinstance(tool_output, dict) else {}
    key = (
        "combined_post_tax_corpus"
        if case.expected_tool == "generate_multi_goal_plan"
        else "projected_corpus"
    )
    # "\x00" is a sentinel that can't appear in a reply, so a missing tool output
    # (tool never called) reliably fails the grounding check instead of matching "".
    return summary.get(key, "\x00")


# --------------------------------------------------------------------------- #
# Running the agent (needs an LLM key)
# --------------------------------------------------------------------------- #
def _capture(messages: list) -> tuple[list[str], dict[str, Any]]:
    """Collect the tool names the agent called and their parsed JSON outputs."""
    called: list[str] = []
    outputs: dict[str, Any] = {}
    for m in messages:
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            called.extend(tc["name"] for tc in tool_calls)
        elif m.__class__.__name__ == "ToolMessage":
            try:
                outputs[m.name] = json.loads(m.content)
            except Exception:  # noqa: BLE001 - skip non-JSON tool output
                pass
    return called, outputs


def score_case(agent: Any, case: EvalCase) -> dict[str, Any]:
    """Run one case through the agent and score the three checks."""
    result = agent.invoke(
        {"messages": [{"role": "user", "content": case.prompt}]},
        config={"configurable": {"thread_id": f"eval-{case.name}"}},
    )
    messages = result["messages"]
    called, outputs = _capture(messages)
    reply = content_to_text(messages[-1].content)
    out = outputs.get(case.expected_tool, {})

    tool_ok = check_tool_selection(called, case.expected_tool)
    inputs_ok = check_inputs(out, case)
    grounded = check_numeric_grounding(reply, expected_ground_str(out, case))
    return {
        "name": case.name,
        "tool_ok": tool_ok,
        "inputs_ok": inputs_ok,
        "grounded": grounded,
        "passed": tool_ok and inputs_ok and grounded,
        "called": called,
        "reply": reply,
    }


def run_eval(agent: Any = None, cases: list[EvalCase] | None = None) -> dict[str, Any]:
    """Score every case and return per-case results plus aggregate pass counts."""
    if agent is None:
        from finplan.agent.planner_agent import build_agent

        agent = build_agent()
    cases = cases or GOLDEN
    results = [score_case(agent, c) for c in cases]
    n = len(results)
    summary = {
        "n": n,
        "tool_pass": sum(r["tool_ok"] for r in results),
        "inputs_pass": sum(r["inputs_ok"] for r in results),
        "grounded_pass": sum(r["grounded"] for r in results),
        "overall_pass": sum(r["passed"] for r in results),
    }
    return {"results": results, "summary": summary}


def judge_explanation(reply: str) -> int | None:
    """Optional LLM-as-judge: rate explanation clarity 1-5 (None if unavailable)."""
    from finplan.config import get_chat_model

    model = get_chat_model("advisor")
    prompt = (
        "Rate from 1 to 5 how clear and helpful this investment-plan explanation is "
        "for a beginner (5 = excellent). Reply with ONLY the integer.\n\n" + reply
    )
    try:
        text = content_to_text(model.invoke(prompt).content)
    except Exception:  # noqa: BLE001 - judging is best-effort
        return None
    match = re.search(r"[1-5]", text)
    return int(match.group()) if match else None
