"""Regression tests for /api/task-followup behavior and prompt construction."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.main_flask as main_flask


def _msg(role: str, content: str, msg_type: str = "conversation"):
    return SimpleNamespace(
        role=role,
        content=content,
        type=msg_type,
        timestamp=datetime.now(),
    )


def _register_session(session_token: str):
    session = SimpleNamespace(
        chat_history=[
            _msg("User", "I am a PhD student who does AI research."),
            _msg("Interviewer", "Are there any tasks you can think of that we haven't covered?"),
        ],
        end_with_thankyou=MagicMock(),
        user_id="u-task-followup",
        session_id=999,
    )
    wrapper = SimpleNamespace(
        session_token=session_token,
        interview_session=session,
        user_id=session.user_id,
    )
    main_flask.active_sessions[session_token] = wrapper
    return session, wrapper


@pytest.fixture
def client():
    main_flask.active_sessions.clear()
    main_flask.chat_history_offsets.clear()
    main_flask.session_audio_cache.clear()
    main_flask.last_messages_by_session.clear()
    main_flask.pending_turns_by_session.clear()
    main_flask.delivered_turn_messages_by_session.clear()
    main_flask.task_followup_history_by_session.clear()

    main_flask.app.config["TESTING"] = True
    with main_flask.app.test_client() as test_client:
        yield test_client

    main_flask.active_sessions.clear()
    main_flask.chat_history_offsets.clear()
    main_flask.session_audio_cache.clear()
    main_flask.last_messages_by_session.clear()
    main_flask.pending_turns_by_session.clear()
    main_flask.delivered_turn_messages_by_session.clear()
    main_flask.task_followup_history_by_session.clear()


def test_task_followup_prompt_requires_understanding_based_followup(client):
    token = "tok-followup-style"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"You mentioned grant writing - what part takes the most time?","task_name":"Write grant proposals to secure project funding","done":false}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "I write grant proposals.",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is False
    assert body["task_name"] == "Write grant proposals to secure project funding"

    prompt = mock_invoke.call_args.args[1]
    assert "CASE B — participant is adding a NEW task" in prompt
    assert "Ask exactly one follow-up question about that task." in prompt
    assert "NEVER ask a follow-up question about the task. NEVER probe for more detail." not in prompt


def test_task_followup_prompt_includes_recent_turn_context(client):
    token = "tok-followup-history"
    _register_session(token)
    prompts = []

    def _fake_invoke(_engine, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return SimpleNamespace(
                content='{"reply":"You mentioned CS seminars - what outcome are you usually aiming for?","task_name":"Attend CS seminars to stay current with research","done":false}'
            )
        return SimpleNamespace(
            content='{"reply":"That helps - what other tasks are part of your week?","done":false}'
        )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=_fake_invoke,
    ):
        first = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "I attend CS seminars.",
                "prior_tasks": [],
                "phase": "probing",
            },
        ).get_json()
        assert first["success"] is True

        second = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "Mostly for networking and staying current.",
                "prior_tasks": ["Attend CS seminars to stay current with research"],
                "phase": "probing",
            },
        ).get_json()
        assert second["success"] is True

    assert len(prompts) == 2
    second_prompt = prompts[1]
    assert "Recent task-followup conversation (oldest to newest):" in second_prompt
    assert "Participant: I attend CS seminars." in second_prompt
    assert "Interviewer: You mentioned CS seminars - what outcome are you usually aiming for?" in second_prompt

