"""CLI entry: ``python -m finplan.eval [--judge]``

Runs the golden set through the live agent (needs a provider key) and prints a
pass-rate report for tool-selection, input-extraction, and numeric-grounding.
With ``--judge`` it also asks the advisor model to rate each explanation 1-5.
"""
from __future__ import annotations

import os
import sys

from finplan.eval.runner import judge_explanation, run_eval


def _has_key() -> bool:
    return any(
        (os.getenv(k) or "") and "your-" not in (os.getenv(k) or "")
        for k in ("GOOGLE_API_KEY", "GROQ_API_KEY")
    )


def _mark(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> int:
    if not _has_key():
        print("No LLM key set (GROQ_API_KEY / GOOGLE_API_KEY); the eval runs the live agent.")
        return 2

    report = run_eval()
    s = report["summary"]
    print(f"\nFinPlan agent eval — {s['n']} cases")
    print("-" * 60)
    print(f"{'case':<30}{'tool':<8}{'inputs':<8}{'grounded':<8}")
    for r in report["results"]:
        print(
            f"{r['name']:<30}{_mark(r['tool_ok']):<8}"
            f"{_mark(r['inputs_ok']):<8}{_mark(r['grounded']):<8}"
        )
    print("-" * 60)
    print(f"tool-selection : {s['tool_pass']}/{s['n']}")
    print(f"input-extract  : {s['inputs_pass']}/{s['n']}")
    print(f"numeric-ground : {s['grounded_pass']}/{s['n']}")
    print(f"OVERALL (all 3): {s['overall_pass']}/{s['n']}")

    if "--judge" in sys.argv:
        scores = [j for r in report["results"] if (j := judge_explanation(r["reply"])) is not None]
        if scores:
            print(f"LLM-judge avg  : {sum(scores) / len(scores):.1f}/5 (n={len(scores)})")

    return 0 if s["overall_pass"] == s["n"] else 1


if __name__ == "__main__":
    sys.exit(main())
