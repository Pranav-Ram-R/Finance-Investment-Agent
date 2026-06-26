"""News-sentiment tool — a qualitative "market mood" from recent headlines.

This is deliberately kept SEPARATE from the numeric engine. Sentiment is soft,
informational context; it must never feed into returns, allocations, or
projections — those stay fully deterministic and defensible. (An interviewer
question this pre-empts: "does the news move your numbers?" — no, by design.)

Headlines come from yfinance (already a dependency, no new API key); scoring is a
small built-in finance lexicon — no LLM, so the score is reproducible. If Yahoo
returns nothing, we degrade gracefully to a "no recent news" result, mirroring the
fallback pattern used elsewhere in the engine.
"""
from __future__ import annotations

import re

import yfinance as yf

# Tiny finance lexicons — enough to read the mood of a headline, easy to explain.
_POSITIVE = {
    "surge", "surges", "gain", "gains", "rally", "rallies", "jump", "jumps",
    "rise", "rises", "soar", "soars", "record", "high", "highs", "boost",
    "upgrade", "beat", "beats", "profit", "profits", "growth", "bullish",
    "outperform", "recovery", "rebound", "strong", "positive", "optimism", "up",
}
_NEGATIVE = {
    "fall", "falls", "drop", "drops", "plunge", "plunges", "slump", "slumps",
    "crash", "decline", "declines", "loss", "losses", "downgrade", "miss",
    "misses", "weak", "bearish", "selloff", "fear", "fears", "recession",
    "slowdown", "cut", "cuts", "negative", "tumble", "tumbles", "down",
    "worry", "worries", "plunged", "sink", "sinks",
}

# Above this the mood is "positive", below its negative the mood is "negative".
_THRESHOLD = 0.15


def score_headlines(headlines: list[str]) -> dict:
    """Score headlines with the finance lexicon (pure, no I/O).

    Returns an overall ``label`` (positive/neutral/negative), a normalized
    ``score`` in [-1, 1], the positive/negative hit counts, and a per-headline
    breakdown. The score is (pos - neg) / (pos + neg), or 0 when nothing matched.
    """
    pos_total = neg_total = 0
    per_headline = []
    for h in headlines:
        words = re.findall(r"[a-z]+", h.lower())
        p = sum(w in _POSITIVE for w in words)
        n = sum(w in _NEGATIVE for w in words)
        pos_total += p
        neg_total += n
        per_headline.append({"headline": h, "score": p - n})

    hits = pos_total + neg_total
    score = 0.0 if hits == 0 else (pos_total - neg_total) / hits
    label = "positive" if score > _THRESHOLD else "negative" if score < -_THRESHOLD else "neutral"

    return {
        "label": label,
        "score": round(score, 3),
        "positive_hits": pos_total,
        "negative_hits": neg_total,
        "headlines": per_headline,
    }


def _extract_title(item: dict) -> str | None:
    """Pull a headline from a yfinance news item across schema versions."""
    # Newer yfinance nests the story under "content"; older returns it flat.
    content = item.get("content") or {}
    return content.get("title") or item.get("title")


def get_news_sentiment(ticker: str = "^NSEI", limit: int = 8) -> dict:
    """Fetch recent headlines for ``ticker`` and return an aggregate sentiment.

    ``ticker`` is a Yahoo Finance symbol (default ``^NSEI``, the Nifty 50).
    Degrades gracefully to a ``no_recent_news`` result if the fetch fails or
    Yahoo returns nothing, so the tool never crashes a conversation.
    """
    try:
        items = yf.Ticker(ticker).news or []
        headlines = [t for it in items[:limit] if (t := _extract_title(it))]
    except Exception:  # noqa: BLE001 - never fail the tool over a flaky feed
        headlines = []

    if not headlines:
        return {
            "ticker": ticker,
            "status": "no_recent_news",
            "label": "unavailable",
            "headlines": [],
        }

    result = score_headlines(headlines)
    result["ticker"] = ticker
    result["status"] = "ok"
    return result
