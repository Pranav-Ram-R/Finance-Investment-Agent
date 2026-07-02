"""HTTP layer — a FastAPI service exposing the engine and agent over REST.

  * ``schemas.py`` — Pydantic request/response models (money fields accept text
    like ``"50 lakh"`` and are normalized to rupees in Python, never by an LLM).
  * ``main.py``    — the FastAPI app. ``/plan`` and ``/plan/multi`` are fully
    deterministic (no LLM key needed); ``/chat`` drives the LangChain agent.

Run:  uvicorn finplan.api.main:app --reload
"""
