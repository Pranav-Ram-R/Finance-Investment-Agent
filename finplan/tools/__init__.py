"""Financial tools package.

Each module holds pure, deterministic functions (no LLM calls) so the math is
exact and unit-testable. The agent layer wraps these as LangChain tools.

Imports are intentionally NOT re-exported here to keep import side-effects (and
test dependencies) minimal — import the specific module you need.
"""
