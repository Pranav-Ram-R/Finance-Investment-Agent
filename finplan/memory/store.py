"""SQLite-backed persistent memory: user plans and contribution progress.

Two kinds of memory power the agent:
  * conversation memory (handled by the agent's LangGraph checkpointer), and
  * this durable store, which survives restarts so a user can save a plan and
    return weeks later to check progress.

Pure stdlib (``sqlite3``) — no external deps, fully unit-testable with a temp DB.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parents[2] / "finplan_memory.sqlite"


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _connect(db_path=None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path or DEFAULT_DB))
    # Row factory lets us read columns by name (row["goal"]) instead of by index.
    conn.row_factory = sqlite3.Row
    # CREATE IF NOT EXISTS makes connect idempotent — first run sets up the schema,
    # later runs are no-ops. Tests pass a temp db_path for full isolation.
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS plans (
            user_id          TEXT PRIMARY KEY,
            initial          REAL,
            monthly          REAL,
            years            REAL,
            goal             REAL,
            profile          TEXT,
            allocation       TEXT,   -- JSON {"equity":.., "debt":.., "gold":..}
            expected_return  REAL,
            projected_value  REAL,
            updated_at       TEXT
        );
        CREATE TABLE IF NOT EXISTS progress (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       TEXT,
            logged_at     TEXT,
            amount        REAL,   -- total invested so far
            current_value REAL    -- current portfolio value
        );
        """
    )
    conn.commit()
    return conn


def save_plan(
    user_id: str,
    *,
    initial: float,
    monthly: float,
    years: float,
    goal: float,
    profile: str,
    allocation: dict,
    expected_return: float,
    projected_value: float,
    db_path=None,
) -> dict:
    """Insert or update (upsert) the user's plan."""
    conn = _connect(db_path)
    # One plan per user (user_id is the PK), so re-saving overwrites via the
    # ON CONFLICT clause rather than inserting a duplicate row.
    conn.execute(
        """
        INSERT INTO plans (user_id, initial, monthly, years, goal, profile,
                           allocation, expected_return, projected_value, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(user_id) DO UPDATE SET
            initial=excluded.initial, monthly=excluded.monthly, years=excluded.years,
            goal=excluded.goal, profile=excluded.profile, allocation=excluded.allocation,
            expected_return=excluded.expected_return,
            projected_value=excluded.projected_value, updated_at=excluded.updated_at
        """,
        (
            user_id, initial, monthly, years, goal, profile,
            json.dumps(allocation), expected_return, projected_value, _now(),
        ),
    )
    conn.commit()
    conn.close()
    return {"status": "saved", "user_id": user_id}


def get_plan(user_id: str, db_path=None) -> dict | None:
    """Return the saved plan for ``user_id``, or ``None`` if there isn't one."""
    conn = _connect(db_path)
    row = conn.execute("SELECT * FROM plans WHERE user_id=?", (user_id,)).fetchone()
    conn.close()
    if row is None:
        return None
    plan = dict(row)
    plan["allocation"] = json.loads(plan["allocation"])
    return plan


def log_contribution(user_id: str, amount: float, current_value: float, db_path=None) -> dict:
    """Append a progress entry (total invested so far + current portfolio value)."""
    conn = _connect(db_path)
    conn.execute(
        "INSERT INTO progress (user_id, logged_at, amount, current_value) VALUES (?,?,?,?)",
        (user_id, _now(), amount, current_value),
    )
    conn.commit()
    conn.close()
    return {"status": "logged", "amount": amount, "current_value": current_value}


def get_progress(user_id: str, db_path=None) -> list[dict]:
    """Return all progress entries for ``user_id``, oldest first."""
    conn = _connect(db_path)
    rows = conn.execute(
        "SELECT * FROM progress WHERE user_id=? ORDER BY id", (user_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
