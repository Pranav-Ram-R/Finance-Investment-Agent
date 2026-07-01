"""Agent evaluation harness.

Measures whether the LLM agent behaves correctly on a golden set of prompts,
against three deterministic checks (no human labels needed):

  1. **Tool selection** — did it call the expected primary tool?
  2. **Input extraction** — did the parsed tool inputs match the intended rupees /
     years (verifying parse_amount handled lakh/crore text)?
  3. **Numeric grounding** — does the final reply quote the engine's pre-formatted
     figure *verbatim* (the project's core "LLM never formats numbers" rule)?

An optional LLM-as-judge scores explanation quality. Running the agent needs a
provider key, so :func:`runner.run_eval` is invoked via ``python -m finplan.eval``;
the pure scoring functions are unit-tested offline.
"""
