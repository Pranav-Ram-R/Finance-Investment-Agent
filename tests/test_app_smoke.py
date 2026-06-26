"""Offline smoke test for the Streamlit UI.

Uses Streamlit's in-process AppTest to run app.py top-to-bottom in a simulated
runtime — catches syntax/runtime errors without a browser or any LLM call.
"""
import os

from streamlit.testing.v1 import AppTest

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_app_boots_without_error():
    at = AppTest.from_file(APP).run(timeout=60)
    assert not at.exception
    assert any("FinPlan" in t.value for t in at.title)
