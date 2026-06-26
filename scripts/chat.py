"""Interactive terminal chat with the FinPlan agent.

Requires an LLM API key in ``.env`` (e.g. GOOGLE_API_KEY or GROQ_API_KEY).
Run from the project root:
    python -m scripts.chat
"""
from __future__ import annotations

import sys

from finplan.agent.planner_agent import build_agent, run_turn


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    print("FinPlan agent ready. Describe your goal, or type 'quit' to exit.\n")
    print("e.g. \"I have 2 lakh now, can invest 15000/month for 12 years, "
          "want 50 lakh, moderate risk.\"\n")

    agent = build_agent()
    # A fixed thread id means every turn shares one conversation (so follow-up
    # "what-if" questions remember the plan built earlier in the session).
    thread = "cli-session"
    while True:
        try:
            msg = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if msg.lower() in {"quit", "exit"}:
            break
        if not msg:
            continue
        print(f"\nfinplan > {run_turn(agent, msg, thread_id=thread)}\n")


if __name__ == "__main__":
    main()
