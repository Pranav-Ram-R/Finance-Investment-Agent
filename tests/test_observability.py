"""Observability tests — offline, using synthetic LLM responses (no real calls)."""
from types import SimpleNamespace

from finplan.observability import MetricsCallbackHandler, _extract_usage


def _response(input_tokens: int, output_tokens: int):
    """A minimal stand-in for a chat model's LLMResult carrying usage_metadata."""
    msg = SimpleNamespace(usage_metadata={"input_tokens": input_tokens, "output_tokens": output_tokens})
    gen = SimpleNamespace(message=msg)
    return SimpleNamespace(generations=[[gen]], llm_output=None)


def test_extract_usage_prefers_usage_metadata():
    assert _extract_usage(_response(12, 7)) == (12, 7)


def test_extract_usage_falls_back_to_llm_output():
    resp = SimpleNamespace(
        generations=[], llm_output={"token_usage": {"prompt_tokens": 3, "completion_tokens": 4}}
    )
    assert _extract_usage(resp) == (3, 4)


def test_handler_accumulates_tokens_and_calls():
    h = MetricsCallbackHandler()
    h.on_tool_start({"name": "generate_plan"}, "")
    h.on_llm_end(_response(10, 5), run_id="a")
    h.on_llm_end(_response(2, 3), run_id="b")

    s = h.summary()
    assert s["llm_calls"] == 2
    assert s["tool_calls"] == ["generate_plan"]
    assert s["input_tokens"] == 12
    assert s["output_tokens"] == 8
    assert s["total_tokens"] == 20
