"""Deterministic parsing of Indian money amounts.

LLMs (especially smaller ones) are unreliable at converting "50 lakh" to
5,000,000 — they drop digits. So we never ask the model to do the arithmetic:
it passes the amount verbatim and this function converts it, in Python.
"""
from __future__ import annotations

import re

# Order matters: check longer/﻿more specific suffixes first (crore before cr).
_UNITS = [
    ("crores", 1e7), ("crore", 1e7), ("cr", 1e7),
    ("lakhs", 1e5), ("lakh", 1e5), ("lacs", 1e5), ("lac", 1e5),
    ("million", 1e6), ("thousand", 1e3), ("k", 1e3),
]


def parse_amount(value: float | int | str) -> float:
    """Convert a money amount to plain rupees.

    Accepts a number (returned as-is) or a string such as ``"50 lakh"``,
    ``"1.5 crore"``, ``"₹2,00,000"``, ``"50k"`` or ``"15000"``.
    """
    if isinstance(value, (int, float)):
        return float(value)

    s = str(value).strip().lower().replace(",", "")
    for sym in ("₹", "inr", "rs.", "rs"):
        s = s.replace(sym, "")
    s = s.strip()

    multiplier = 1.0
    for unit, factor in _UNITS:
        if s.endswith(unit):
            multiplier = factor
            s = s[: -len(unit)].strip()
            break

    match = re.search(r"-?\d+(\.\d+)?", s)
    if not match:
        raise ValueError(f"could not parse a money amount from {value!r}")
    return float(match.group()) * multiplier


def _indian_group(n: int) -> str:
    """Group digits the Indian way: 4851170 -> '48,51,170'."""
    s = str(abs(n))
    if len(s) <= 3:
        body = s
    else:
        # Indian grouping is the last 3 digits, then 2-digit groups (12,34,56,789),
        # unlike the Western 3-3-3. Peel the last 3, then chunk the rest in pairs.
        last3, rest = s[-3:], s[:-3]
        groups = []
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        body = ",".join(groups) + "," + last3
    return ("-" if n < 0 else "") + body


def format_inr(amount: float) -> str:
    """Format a rupee amount with Indian digit grouping, e.g. '₹48,51,170'.

    Used to pre-format figures so the LLM copies ready-made strings instead of
    (unreliably) regrouping or rescaling numbers itself.
    """
    return "₹" + _indian_group(round(amount))
