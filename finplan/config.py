"""Central LLM configuration for the Goal-Based Investment Planner.

Every model in the app is created through :func:`get_chat_model`, so swapping
providers or models is a one-line change in your ``.env`` file — no code edits.

Spec format is ``"provider:model_id"`` (LangChain's ``init_chat_model``
convention), e.g. ``"google_genai:gemini-2.5-flash"`` or
``"groq:llama-3.3-70b-versatile"``.

Roles
-----
orchestrator : fast, reliable tool-caller that drives the agent loop.
advisor      : (optionally heavier) model that writes the final recommendation.

Override a role by setting ``ORCHESTRATOR_MODEL`` / ``ADVISOR_MODEL`` in ``.env``
to either a preset alias (see :data:`PRESETS`) or a full ``provider:model_id``.
"""
from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

# Free / low-cost defaults (override per-role in .env).
DEFAULTS: dict[str, str] = {
    "orchestrator": "google_genai:gemini-2.5-flash",
    "advisor": "google_genai:gemini-2.5-flash",
}

# Convenience aliases so .env can say e.g. ORCHESTRATOR_MODEL=llama-70b.
# NOTE: provider model IDs drift over time — verify against the provider's
# current model list (Google AI Studio / console.groq.com/docs/models / etc.).
PRESETS: dict[str, str] = {
    "gemini-flash": "google_genai:gemini-2.5-flash",
    "gemini-pro": "google_genai:gemini-2.5-pro",
    "llama-70b": "groq:llama-3.3-70b-versatile",
    "llama-8b": "groq:llama-3.1-8b-instant",
    "deepseek-r1": "groq:deepseek-r1-distill-llama-70b",
    "qwen": "groq:qwen-2.5-32b",
    "mistral": "mistralai:mistral-small-latest",
    "deepseek-chat": "deepseek:deepseek-chat",
}

# Reasoning models reject sampling params like ``temperature``; don't pass them.
_REASONING_HINTS = ("deepseek-r1", "reasoner", "o1", "o3", "qwq")


def resolve_spec(spec: str) -> str:
    """Map a preset alias to a full ``provider:model_id`` spec.

    Unknown values pass through unchanged, so a full spec also works.
    """
    return PRESETS.get(spec, spec)


@lru_cache(maxsize=8)
def get_chat_model(role: str = "orchestrator", temperature: float = 0.1):
    """Return a configured LangChain chat model for ``role``.

    The model is read from the ``{ROLE}_MODEL`` env var, falling back to
    :data:`DEFAULTS`. The result is cached so repeated calls reuse one client.
    """
    if role not in DEFAULTS:
        raise ValueError(f"Unknown role {role!r}; expected one of {list(DEFAULTS)}")

    # Read {ROLE}_MODEL from .env (e.g. ORCHESTRATOR_MODEL), fall back to DEFAULTS,
    # then expand any preset alias to a full provider:model_id.
    spec = resolve_spec(os.getenv(f"{role.upper()}_MODEL", DEFAULTS[role]))

    # Reasoning models (o1, deepseek-r1…) reject a temperature arg — omit it for them.
    kwargs: dict[str, object] = {}
    if not any(hint in spec.lower() for hint in _REASONING_HINTS):
        kwargs["temperature"] = temperature

    # init_chat_model picks the right provider client from the "provider:" prefix.
    return init_chat_model(spec, **kwargs)


def describe_config() -> dict[str, str]:
    """Return the resolved spec for each role (for logs / an 'About' panel)."""
    return {
        role: resolve_spec(os.getenv(f"{role.upper()}_MODEL", default))
        for role, default in DEFAULTS.items()
    }
