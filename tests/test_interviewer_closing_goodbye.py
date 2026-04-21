"""Tests for the LLM-backed `_looks_like_closing_goodbye` classifier.

Covers:
- Fast-path short-circuits (empty string, message containing '?').
- LLM JSON responses mapped to True/False.
- Graceful False on LLM exception or unparseable output.
- Lazy construction + reuse of the small judge engine.
"""
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mocked_engines():
    """Patch both get_engine + invoke_engine as they're referenced inside
    src.agents.interviewer.interviewer after `from ... import get_engine, invoke_engine`.
    """
    with patch("src.agents.interviewer.interviewer.get_engine") as mock_get, \
         patch("src.agents.interviewer.interviewer.invoke_engine") as mock_invoke:
        yield mock_get, mock_invoke


@pytest.mark.asyncio
async def test_empty_text_returns_false_without_llm_call(interviewer, mocked_engines):
    _, mock_invoke = mocked_engines
    assert await interviewer._looks_like_closing_goodbye("") is False
    assert await interviewer._looks_like_closing_goodbye("   ") is False
    mock_invoke.assert_not_called()


@pytest.mark.asyncio
async def test_text_with_question_mark_returns_false_without_llm_call(interviewer, mocked_engines):
    _, mock_invoke = mocked_engines
    result = await interviewer._looks_like_closing_goodbye(
        "Thanks for sharing — how did that land with your team?"
    )
    assert result is False
    mock_invoke.assert_not_called()


@pytest.mark.asyncio
async def test_llm_closing_true_returns_true(interviewer, mocked_engines):
    _, mock_invoke = mocked_engines
    mock_invoke.return_value = MagicMock(
        content='{"is_closing": true, "reason": "signals wrap-up"}'
    )

    result = await interviewer._looks_like_closing_goodbye(
        "Thanks so much for walking me through this — we're all set here."
    )
    assert result is True
    mock_invoke.assert_called_once()


@pytest.mark.asyncio
async def test_llm_closing_false_returns_false(interviewer, mocked_engines):
    """Pure acknowledgment mid-session should NOT be flagged as closing."""
    _, mock_invoke = mocked_engines
    mock_invoke.return_value = MagicMock(
        content='{"is_closing": false, "reason": "ack only, no wrap-up"}'
    )

    result = await interviewer._looks_like_closing_goodbye(
        "Thanks for laying that out so clearly — that gives a really crisp "
        "picture of how your week actually gets spent."
    )
    assert result is False
    mock_invoke.assert_called_once()


@pytest.mark.asyncio
async def test_llm_exception_returns_false(interviewer, mocked_engines):
    _, mock_invoke = mocked_engines
    mock_invoke.side_effect = RuntimeError("engine boom")

    result = await interviewer._looks_like_closing_goodbye(
        "Appreciate the time — that's everything I needed."
    )
    assert result is False


@pytest.mark.asyncio
async def test_unparseable_llm_output_returns_false(interviewer, mocked_engines):
    _, mock_invoke = mocked_engines
    mock_invoke.return_value = MagicMock(content="not json at all")

    result = await interviewer._looks_like_closing_goodbye(
        "Thanks — we'll wrap here."
    )
    assert result is False


@pytest.mark.asyncio
async def test_judge_engine_constructed_once_and_reused(interviewer, mocked_engines):
    mock_get, mock_invoke = mocked_engines
    mock_get.return_value = MagicMock(name="small_engine")
    mock_invoke.return_value = MagicMock(content='{"is_closing": false}')

    await interviewer._looks_like_closing_goodbye("Thanks.")
    await interviewer._looks_like_closing_goodbye("Appreciate it.")
    await interviewer._looks_like_closing_goodbye("All good.")

    assert mock_get.call_count == 1
    # Engine arg passed to invoke_engine should be the cached one
    engines_used = {call.args[0] for call in mock_invoke.call_args_list}
    assert engines_used == {mock_get.return_value}


@pytest.mark.asyncio
async def test_judge_engine_respects_env_override(interviewer, mocked_engines, monkeypatch):
    mock_get, mock_invoke = mocked_engines
    monkeypatch.setenv("INTERVIEWER_JUDGE_MODEL", "claude-3-5-haiku")
    # Force fresh construction (fixture already ran __init__, which left _judge_engine=None)
    interviewer._judge_engine = None
    mock_invoke.return_value = MagicMock(content='{"is_closing": false}')

    await interviewer._looks_like_closing_goodbye("Thanks.")

    mock_get.assert_called_once()
    assert mock_get.call_args.kwargs.get("model_name") == "claude-3-5-haiku"
