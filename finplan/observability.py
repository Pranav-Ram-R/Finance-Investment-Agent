"""Lightweight observability for the agent loop — token usage, latency, tool calls.

A single :class:`MetricsCallbackHandler` (a LangChain callback) accumulates, per
turn, how many LLM calls and tool calls happened and how many input/output tokens
were spent, while emitting a structured log line for each event. The API attaches
one handler per ``/chat`` turn and returns its :meth:`summary` so a client (or the
Streamlit trace) can show cost/latency alongside the answer.

Token accounting is provider-agnostic: it prefers LangChain's normalized
``usage_metadata`` on the response message and falls back to the raw
``llm_output`` usage block, so it works across Groq / Gemini / others.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("finplan.metrics")


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a simple structured handler once (safe to call repeatedly)."""
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(level)


def _extract_usage(response: Any) -> tuple[int, int]:
    """Return (input_tokens, output_tokens) from an LLM response, best-effort."""
    # Preferred: the normalized usage_metadata LangChain puts on the AIMessage.
    try:
        for batch in response.generations:
            for gen in batch:
                msg = getattr(gen, "message", None)
                usage = getattr(msg, "usage_metadata", None) if msg is not None else None
                if usage:
                    return int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0))
    except Exception:  # noqa: BLE001 - metrics must never break a turn
        pass
    # Fallback: raw provider usage in llm_output (OpenAI/Groq-style keys).
    out = getattr(response, "llm_output", None) or {}
    tu = out.get("token_usage") or out.get("usage") or {}
    inp = tu.get("prompt_tokens") or tu.get("input_tokens") or 0
    outp = tu.get("completion_tokens") or tu.get("output_tokens") or 0
    return int(inp or 0), int(outp or 0)


class MetricsCallbackHandler(BaseCallbackHandler):
    """Accumulates per-turn LLM/tool/token metrics and logs each event."""

    def __init__(self) -> None:
        self.llm_calls = 0
        self.tool_calls: list[str] = []
        self.input_tokens = 0
        self.output_tokens = 0
        self._llm_started_at: dict[Any, float] = {}

    # -- LLM ---------------------------------------------------------------- #
    def on_llm_start(self, serialized, prompts, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        self._llm_started_at[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id=None, **kwargs) -> None:  # noqa: ANN001
        self.llm_calls += 1
        inp, outp = _extract_usage(response)
        self.input_tokens += inp
        self.output_tokens += outp
        started = self._llm_started_at.pop(run_id, None)
        dt = f"{time.perf_counter() - started:.2f}s" if started is not None else "?"
        logger.info("llm_end call=%d in=%d out=%d dur=%s", self.llm_calls, inp, outp, dt)

    # -- Tools -------------------------------------------------------------- #
    def on_tool_start(self, serialized, input_str, **kwargs) -> None:  # noqa: ANN001
        name = (serialized or {}).get("name", "tool")
        self.tool_calls.append(name)
        logger.info("tool_start name=%s", name)

    def summary(self) -> dict[str, Any]:
        """The per-turn metrics, ready to return to a client."""
        return {
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
        }
