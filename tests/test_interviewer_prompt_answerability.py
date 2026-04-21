from unittest.mock import AsyncMock

import pytest


def _prime_prompt_methods(fake_session):
    agenda = fake_session.session_agenda
    agenda.get_user_portrait_str.return_value = "portrait"
    agenda.get_last_meeting_summary_str.return_value = "prior summary"
    agenda.get_questions_and_notes_str.return_value = "topics"


def test_get_recent_user_answers_filters_and_limits(interviewer, fake_session):
    _prime_prompt_methods(fake_session)

    interviewer.add_event(sender="User", tag="message", content="first")
    interviewer.add_event(sender="Interviewer", tag="message", content="question")
    interviewer.add_event(sender="system", tag="recall", content="memory")
    interviewer.add_event(sender="User", tag="message", content="second")
    interviewer.add_event(sender="User", tag="message", content="third")

    recent = interviewer._get_recent_user_answers(limit=2)
    assert recent == ["second", "third"]


def test_prompt_includes_recent_user_answers_block(interviewer, fake_session):
    _prime_prompt_methods(fake_session)
    # Force normal prompt variant (not introduction).
    interviewer.add_event(
        sender="Interviewer",
        tag="message",
        content="What does your week look like?",
    )
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I teach business analytics in the fall and do research year-round.",
    )

    prompt = interviewer._get_prompt()

    assert "<recent_user_answers>" in prompt
    assert "I teach business analytics in the fall and do research year-round." in prompt


def test_prompt_includes_no_domain_reask_and_no_repeat_probe_rules(interviewer, fake_session):
    _prime_prompt_methods(fake_session)
    interviewer.add_event(
        sender="Interviewer",
        tag="message",
        content="What field are you in?",
    )
    interviewer.add_event(
        sender="User",
        tag="message",
        content="Information systems.",
    )

    prompt = interviewer._get_prompt()

    assert "once the participant names a broad field/domain" in prompt
    assert "you keep asking this" in prompt


def test_repetitive_breadth_probe_is_rewritten_to_specific_followup(interviewer, fake_session):
    _prime_prompt_methods(fake_session)
    interviewer.add_event(
        sender="Interviewer",
        tag="message",
        content="Outside of teaching and research, what else do you do now and then?",
    )
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I'm in meetings a lot to direct students on research tasks.",
    )

    rewritten = interviewer._rewrite_repetitive_breadth_probe(
        "Besides that, what else is on your plate?",
    )

    assert "what else" not in rewritten.lower()
    assert "anything else" not in rewritten.lower()
    assert rewritten.startswith("You mentioned ")
    assert "What's the main outcome you're aiming for there?" in rewritten


def test_first_breadth_probe_is_not_rewritten(interviewer, fake_session):
    _prime_prompt_methods(fake_session)
    rewritten = interviewer._rewrite_repetitive_breadth_probe(
        "Besides teaching and research, what else is on your plate?",
    )
    assert rewritten == "Besides teaching and research, what else is on your plate?"


def test_semantic_duplicate_gate_skips_low_risk_question(interviewer, fake_session, monkeypatch):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "auto")
    interviewer.add_event(
        sender="Interviewer",
        tag="message",
        content="What does your week look like?",
    )
    should_run = interviewer._should_run_semantic_duplicate_llm(
        "How long have you been in this postdoc role?",
    )
    assert should_run is False


def test_semantic_duplicate_gate_runs_for_breadth_probe(interviewer, fake_session, monkeypatch):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "auto")
    should_run = interviewer._should_run_semantic_duplicate_llm(
        "Beyond that, is there anything else you do less often?",
    )
    assert should_run is True


def test_semantic_duplicate_gate_runs_for_task_list_collection_subtopic(
    interviewer, fake_session, monkeypatch
):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "auto")
    should_run = interviewer._should_run_semantic_duplicate_llm(
        "How long have you been in this postdoc role?",
        subtopic_id="2.2",
    )
    assert should_run is True


@pytest.mark.asyncio
async def test_handle_response_skips_semantic_duplicate_llm_on_low_risk_turn(
    interviewer, fake_session, monkeypatch
):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "auto")
    interviewer._check_semantic_duplicate = AsyncMock(return_value=(False, ""))
    interviewer.add_event(
        sender="Interviewer",
        tag="message",
        content="What does your week look like?",
    )

    await interviewer._handle_response(
        "How long have you been in this postdoc role?",
        subtopic_id="1.2",
    )

    interviewer._check_semantic_duplicate.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_response_runs_semantic_duplicate_llm_for_task_list_collection(
    interviewer, fake_session, monkeypatch
):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "auto")
    interviewer._check_semantic_duplicate = AsyncMock(return_value=(False, ""))
    interviewer.add_event(
        sender="Interviewer",
        tag="message",
        content="What does your week look like?",
    )

    await interviewer._handle_response(
        "How long have you been in this postdoc role?",
        subtopic_id="2.2",
    )

    interviewer._check_semantic_duplicate.assert_awaited_once()
