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


def test_multi_quant_question_is_rewritten_to_single_qualitative_followup(interviewer, fake_session):
    _prime_prompt_methods(fake_session)
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I met with trainees and coordinators to align priorities for experiments and paperwork.",
    )

    rewritten = interviewer._rewrite_multi_quant_question(
        "On a recent Tuesday, roughly how many separate meetings did you have with postdocs, "
        "PhD students, and coordinators, and about how long do they usually run?",
        subtopic_id="2.2",
    )

    assert "how many" not in rewritten.lower()
    assert "how long" not in rewritten.lower()
    assert rewritten.count("?") == 1
    assert "focus on" in rewritten.lower()


def test_multi_quant_question_in_time_allocation_subtopic_rewrites_to_single_split(interviewer, fake_session):
    _prime_prompt_methods(fake_session)
    rewritten = interviewer._rewrite_multi_quant_question(
        "Roughly how many blocks do you have and how long is each one?",
        subtopic_id="2.4",
    )
    assert rewritten == "Roughly how is your time split across the main tasks you mentioned this week?"


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


def test_inferability_gate_runs_for_task_list_collection_subtopic(
    interviewer, fake_session, monkeypatch
):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_INFERABILITY_MODE", "auto")
    interviewer._inferability_mode = "auto"
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I usually prepare slide decks and supporting figures for those presentations.",
    )
    should_run = interviewer._should_run_inferability_llm(
        "For that presentation prep work, what's the main thing you're producing?",
        subtopic_id="2.2",
    )
    assert should_run is True


def test_inferability_gate_skips_when_mode_off(interviewer, fake_session, monkeypatch):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_INFERABILITY_MODE", "off")
    interviewer._inferability_mode = "off"
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I prepare slide decks for those talks.",
    )
    should_run = interviewer._should_run_inferability_llm(
        "For that presentation prep work, what's the main thing you're producing?",
        subtopic_id="2.2",
    )
    assert should_run is False


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


@pytest.mark.asyncio
async def test_handle_response_rewrites_inferable_question_before_sending(
    interviewer, fake_session, monkeypatch
):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "off")
    monkeypatch.setenv("INTERVIEWER_INFERABILITY_MODE", "always")
    interviewer._inferability_mode = "always"
    interviewer._check_inferable_question = AsyncMock(
        return_value=(
            True,
            "Shifting gears a bit — roughly how is your time split across those tasks in a typical week?",
        )
    )
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I mostly produce slide decks and speaking notes for those presentations.",
    )

    await interviewer._handle_response(
        "For that presentation prep work, what's the main thing you're producing?",
        subtopic_id="2.2",
    )

    interviewer._check_inferable_question.assert_awaited_once()
    assert fake_session.chat_history[-1]["content"] == (
        "Shifting gears a bit — roughly how is your time split across those tasks in a typical week?"
    )


@pytest.mark.asyncio
async def test_handle_response_rewrites_multi_quant_question_before_sending(
    interviewer, fake_session, monkeypatch
):
    _prime_prompt_methods(fake_session)
    monkeypatch.setenv("INTERVIEWER_SEMANTIC_DUP_MODE", "off")
    monkeypatch.setenv("INTERVIEWER_INFERABILITY_MODE", "off")
    interviewer._inferability_mode = "off"
    interviewer.add_event(
        sender="User",
        tag="message",
        content="I had several meetings with postdocs and coordinators to unblock ongoing studies.",
    )

    await interviewer._handle_response(
        "On a recent Tuesday, roughly how many separate meetings did you have with postdocs, "
        "PhD students, and coordinators, and about how long do they usually run?",
        subtopic_id="2.2",
    )

    sent = fake_session.chat_history[-1]["content"]
    assert "how many" not in sent.lower()
    assert "how long" not in sent.lower()
    assert sent.count("?") == 1
