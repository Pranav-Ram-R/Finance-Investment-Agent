"""FinPlan — Streamlit UI for the Goal-Based Investment Planner agent.

A chat interface over the LangChain agent, plus charts auto-generated from the
agent's tool outputs (allocation, projection, Monte-Carlo) and an expandable
tool-call trace so you can see exactly how the agent reasoned.

Run:  streamlit run app.py
"""
from __future__ import annotations

import json
import os
import uuid

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))  # load .env before importing the agent

import plotly.graph_objects as go
import streamlit as st

from finplan.agent.planner_agent import build_agent, content_to_text
from finplan.config import describe_config

st.set_page_config(page_title="FinPlan — Goal-Based Investment Planner", page_icon="📈", layout="wide")


def _has_key() -> bool:
    return any(
        (os.getenv(k) or "") and "your-" not in (os.getenv(k) or "")
        for k in ("GOOGLE_API_KEY", "GROQ_API_KEY")
    )


@st.cache_resource
def get_agent():
    return build_agent()


def _current_turn_messages(messages: list) -> list:
    """Slice the full thread history down to just the latest turn.

    The agent is checkpointed, so ``invoke`` returns the ENTIRE conversation,
    not only the new messages. The current turn is everything from the most
    recent user message onward; scoping to it keeps stale tool outputs from
    earlier turns out of the charts and trace (reading the whole history left
    the charts stuck on the first plan and never updating on follow-ups).
    """
    last_user = 0
    for i, m in enumerate(messages):
        if m.__class__.__name__ == "HumanMessage" or getattr(m, "type", None) == "human":
            last_user = i
    return messages[last_user:]


def invoke_agent(user_msg: str, thread_id: str):
    """Run one turn; return (reply_text, captured_tool_outputs, trace_lines).

    Only the latest turn's tool outputs/trace are returned — see
    :func:`_current_turn_messages`.
    """
    result = get_agent().invoke(
        {"messages": [{"role": "user", "content": user_msg}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    captured: dict = {}
    trace: list[str] = []
    for m in _current_turn_messages(result["messages"]):
        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                trace.append(f"→ {tc['name']}({json.dumps(tc['args'])})")
        elif m.__class__.__name__ == "ToolMessage":
            try:
                captured[m.name] = json.loads(m.content)
            except Exception:  # noqa: BLE001
                pass
            snippet = str(m.content)
            trace.append(f"    {m.name} ⇒ {snippet[:150]}{'…' if len(snippet) > 150 else ''}")
    return content_to_text(result["messages"][-1].content), captured, trace


def extract_chart_components(cap: dict) -> dict:
    """Pull the chart pieces (alloc/proj/mc/feas) out of one turn's tool outputs.

    A full plan comes from ``generate_plan``; a follow-up "what-if" turn instead
    calls the granular tools. Granular outputs are applied AFTER the plan so a
    recomputed projection/Monte-Carlo/feasibility overrides the now-stale plan
    value, while pieces the turn didn't touch (e.g. allocation) are left absent
    and preserved by the caller's per-component merge.
    """
    out: dict = {}
    gp = cap.get("generate_plan")
    if gp:
        out["alloc"] = gp.get("allocation")
        out["proj"] = gp.get("projection")
        out["mc"] = gp.get("monte_carlo")
        out["feas"] = gp.get("feasibility")
        out["tax"] = gp.get("tax")
    granular_alloc = (cap.get("get_portfolio_market_data") or cap.get("recommend_allocation") or {}).get("allocation")
    if granular_alloc:
        out["alloc"] = granular_alloc
    if cap.get("project_growth"):
        out["proj"] = cap["project_growth"]
    if cap.get("run_monte_carlo"):
        out["mc"] = cap["run_monte_carlo"]
    if cap.get("check_feasibility"):
        out["feas"] = cap["check_feasibility"]
    return {k: v for k, v in out.items() if v}


def render_charts(comp: dict) -> None:
    """Render allocation pie, projection-vs-goal chart, and headline metrics."""
    alloc, proj = comp.get("alloc"), comp.get("proj")
    mc, feas = comp.get("mc"), comp.get("feas")
    goal = (feas or {}).get("goal") or (mc or {}).get("goal")

    if feas:
        c1, c2, c3 = st.columns(3)
        c1.metric("Projected corpus", f"₹{feas['projected_value']:,.0f}")
        c2.metric("Goal", f"₹{feas['goal']:,.0f}")
        c3.metric(
            "On track?",
            "Yes ✅" if feas["on_track"] else "Short ⚠️",
            delta=f"₹{feas['difference']:,.0f}",
        )

    tax = comp.get("tax")
    if tax:
        st.caption(
            f"After estimated LTCG tax (12.5% over a ₹1.25L exemption): "
            f"**₹{tax['post_tax_corpus']:,.0f}** post-tax corpus "
            f"(est. tax ₹{tax['estimated_tax']:,.0f}). Not tax advice."
        )

    left, right = st.columns(2)
    if alloc:
        fig = go.Figure(go.Pie(labels=list(alloc), values=list(alloc.values()), hole=0.45))
        fig.update_layout(title="Recommended allocation", margin=dict(t=40, b=0))
        left.plotly_chart(fig, use_container_width=True)

    if proj:
        traj = proj["trajectory"]
        xs = [p["year"] for p in traj]
        ys = [p["value"] for p in traj]
        fig = go.Figure()
        fig.add_scatter(x=xs, y=ys, mode="lines+markers", name="Projected (expected return)")
        if mc:
            yr = xs[-1] if xs else proj.get("trajectory", [{}])[-1].get("year", 0)
            fig.add_scatter(
                x=[yr, yr], y=[mc["p10"], mc["p90"]], mode="lines",
                name="Monte-Carlo p10–p90", line=dict(width=10), opacity=0.4,
            )
            fig.add_scatter(x=[yr], y=[mc["median"]], mode="markers",
                            name="MC median", marker=dict(size=12, symbol="diamond"))
        if goal:
            fig.add_hline(y=goal, line_dash="dash", annotation_text="Goal")
        fig.update_layout(title="Projected growth vs goal", xaxis_title="Year",
                          yaxis_title="₹", margin=dict(t=40, b=0))
        right.plotly_chart(fig, use_container_width=True)

    if mc and goal:
        st.info(
            f"**Monte-Carlo:** {mc['probability_of_reaching_goal'] * 100:.0f}% chance of reaching "
            f"₹{goal:,.0f}. Range at horizon: ₹{mc['p10']:,.0f} (p10) → "
            f"₹{mc['median']:,.0f} (median) → ₹{mc['p90']:,.0f} (p90)."
        )


def render_multi_goal(multi: dict) -> None:
    """Render the combined totals + per-goal table for a multi-goal plan."""
    s = multi["summary"]
    st.subheader(f"Multi-goal plan — {s['num_goals']} goals")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total monthly SIP", s["total_monthly_sip"])
    c2.metric("Combined post-tax corpus", s["combined_post_tax_corpus"])
    c3.metric("All on track?", "Yes ✅" if s["all_on_track"] else "Some short ⚠️")
    st.table(s["per_goal"])


def render_sentiment(sent: dict) -> None:
    """Render the 'market mood' read as soft context (never affects the numbers)."""
    if sent.get("status") != "ok":
        st.info("📰 No recent headlines available for a sentiment read right now.")
        return
    emoji = {"positive": "🟢", "neutral": "⚪", "negative": "🔴"}.get(sent["label"], "⚪")
    headlines = "\n".join(f"- {h['headline']}" for h in sent["headlines"][:5])
    st.info(
        f"📰 **Market mood for {sent['ticker']}: {emoji} {sent['label']}** "
        f"(score {sent['score']:+.2f}, from recent headlines — informational only, "
        f"it does not change the plan's numbers):\n{headlines}"
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
st.title("📈 FinPlan — Goal-Based Investment Planner")
st.caption("LangChain agent · real market data · every number is computed by a tool, not guessed.")

with st.sidebar:
    st.subheader("⚙️ Models")
    for role, spec in describe_config().items():
        st.markdown(f"**{role}** — `{spec}`")
    st.divider()
    st.subheader("💡 What you can ask")
    st.markdown(
        "**Start a plan** — describe your goal in plain words:\n"
        "> *I have ₹2 lakh now, can invest ₹15,000/month for 12 years, "
        "my goal is ₹50 lakh, moderate risk.*\n\n"
        "Amounts can be lakh/crore, ₹ symbols, or plain numbers — no need to "
        "convert. Missing a detail? It'll ask.\n\n"
        "**Try a what-if** (same chat):\n"
        "- *What if I bump the SIP to ₹20k?*\n"
        "- *What if I extend to 15 years?*\n"
        "- *What return would I need to hit the goal?*\n"
        "- *Adjust my target for 6% inflation.*\n\n"
        "**Save & track** across sessions:\n"
        "- *Save this plan.*  /  *What was my plan again?*\n"
        "- *I've invested ₹1.8L, portfolio's at ₹2.1L — am I on track?*"
    )
    st.divider()
    if st.button("🔄 New conversation"):
        st.session_state.clear()
        st.rerun()
    st.caption("Educational tool, not financial advice.")

if not _has_key():
    st.error(
        "No API key found. Add `GOOGLE_API_KEY` (or `GROQ_API_KEY`) to your `.env`, "
        "then restart. Get a free key at https://aistudio.google.com/app/apikey"
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.charts = None
    st.session_state.trace = None
    st.session_state.thread_id = f"st-{uuid.uuid4().hex[:8]}"

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Describe your financial goal…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Planning… (profiling risk, fetching market data, simulating outcomes)"):
            reply, captured, trace = invoke_agent(prompt, st.session_state.thread_id)
        st.markdown(reply)
    st.session_state.messages.append({"role": "assistant", "content": reply})
    fresh = extract_chart_components(captured)
    if fresh:
        # Merge per-component so a what-if turn updates only what it recomputed
        # (e.g. projection) while keeping pieces it didn't touch (e.g. allocation).
        components = dict(st.session_state.get("charts") or {})
        components.update(fresh)
        st.session_state.charts = components
    if captured.get("generate_multi_goal_plan"):
        st.session_state.multi = captured["generate_multi_goal_plan"]
    if captured.get("get_news_sentiment"):
        st.session_state.sentiment = captured["get_news_sentiment"]
    if trace:  # keep the last tool-call trace visible across pure-chat turns
        st.session_state.trace = trace

if st.session_state.get("charts"):
    st.divider()
    render_charts(st.session_state.charts)

if st.session_state.get("multi"):
    st.divider()
    render_multi_goal(st.session_state.multi)

if st.session_state.get("sentiment"):
    render_sentiment(st.session_state.sentiment)

if st.session_state.get("trace"):
    with st.expander("🔍 How the agent reasoned (tool-call trace)"):
        st.code("\n".join(st.session_state.trace), language="text")
