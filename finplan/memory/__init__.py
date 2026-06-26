"""Memory package — persistent user plan and progress log.

See ``store.py`` for the SQLite-backed store (saved plans + contribution
progress) that survives restarts, complementing the agent's in-process
conversation checkpointer.
"""
