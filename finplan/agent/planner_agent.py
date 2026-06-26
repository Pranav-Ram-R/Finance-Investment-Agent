"""Builds the FinPlan agent: a LangChain v1 tool-calling agent (on LangGraph).

The agent wires together:
  * the swappable orchestrator model (see :mod:`finplan.config`),
  * the deterministic financial tools (see :mod:`finplan.agent.tools`),
  * the workflow-enforcing system prompt, and
  * a checkpointer for multi-turn conversation memory.
"""
from __future__ import annotations

from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from finplan.agent.prompts import SYSTEM_PROMPT
from finplan.agent.tools import ALL_TOOLS
from finplan.config import get_chat_model


def build_agent(checkpointer=None):
    """Create the compiled FinPlan agent.

    Parameters
    ----------
    checkpointer:
        LangGraph checkpointer for conversation memory. Defaults to an in-memory
        saver (per-process); swap for a persistent one to remember chats across
        restarts.
    """
    model = get_chat_model("orchestrator")
    return create_agent(
        model,
        ALL_TOOLS,
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )


def content_to_text(content) -> str:
    """Flatten a message's content to plain text.

    Different providers return content differently: some give a plain string,
    others (e.g. Gemini 2.5, Anthropic) return a list of content blocks like
    ``[{"type": "text", "text": "..."}]``. This normalizes both.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and "text" in block
        ]
        return "\n".join(parts).strip()
    return str(content)


def run_turn(agent, message: str, thread_id: str = "default") -> str:
    """Send one user message and return the agent's final text reply.

    ``thread_id`` selects the conversation; reusing it preserves memory across
    turns. A new id starts a fresh conversation.
    """
    result = agent.invoke(
        {"messages": [{"role": "user", "content": message}]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return content_to_text(result["messages"][-1].content)
