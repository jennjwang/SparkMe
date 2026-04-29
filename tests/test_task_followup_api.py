"""Regression tests for /api/task-followup behavior and prompt construction."""

import json
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


def _register_session(
    session_token: str,
    latest_interviewer: str = "What other tasks are part of your work?",
):
    session = SimpleNamespace(
        chat_history=[
            _msg("User", "I am a PhD student who does AI research."),
            _msg("Interviewer", latest_interviewer),
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
    main_flask.task_followup_turns_by_session.clear()

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
    main_flask.task_followup_turns_by_session.clear()


def test_final_task_probe_advances_without_followup(client):
    token = "tok-final-probe"
    session, _ = _register_session(
        token,
        "Are there any tasks you can think of that we haven't covered?",
    )

    fake_response = SimpleNamespace(
        content='{"decision":"closure","task_name":"","clarification_question":"","reason":"participant says the list is complete"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "In those pretty much you have all we do",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is True
    assert body["reply"] == ""
    assert body["task_name"] is None
    mock_invoke.assert_called_once()
    assert session.chat_history[-1].role == "User"
    assert session.chat_history[-1].content == "In those pretty much you have all we do"
    assert session._completeness_probe_sent is False
    assert session._post_probe_followup_pending is False
    assert session._final_task_probe_loop_active is False


def test_final_task_probe_loops_when_new_task_is_clear(client, tmp_path, monkeypatch):
    monkeypatch.setenv("LOGS_DIR", str(tmp_path))
    token = "tok-final-probe-clear-task"
    session, _ = _register_session(
        token,
        "Are there any tasks you can think of that we haven't covered?",
    )
    session.session_agenda = SimpleNamespace(
        user_portrait={"Task Inventory": ["Build web pages"]}
    )
    responses = [
        SimpleNamespace(
            content='{"decision":"clear_task","task_name":"Fix software bugs; answer client questions","clarification_question":"","reason":"participant named clear action-object tasks"}'
        ),
        SimpleNamespace(
            content='{"reply":"I will add fixing software bugs and answering client questions. What other work should we add to the list?"}'
        ),
    ]

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=responses,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "Solving bugs and answering specific questions for the client",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is False
    assert body["reply"] == "I will add fixing software bugs and answering client questions. What other work should we add to the list?"
    assert body["task_name"] == "Fix software bugs; answer client questions"
    assert mock_invoke.call_count == 2
    assert session._completeness_probe_sent is False
    assert session._post_probe_followup_pending is False
    assert session._final_task_probe_loop_active is True
    assert session.chat_history[-1].role == "Interviewer"
    assert session.chat_history[-1].content == "I will add fixing software bugs and answering client questions. What other work should we add to the list?"
    assert session.session_agenda.user_portrait["Task Inventory"] == [
        "Build web pages",
        "Fix software bugs",
        "answer client questions",
    ]
    portrait_path = tmp_path / session.user_id / "user_portrait.json"
    saved_portrait = json.loads(portrait_path.read_text())
    assert (
        saved_portrait["Task Inventory"]
        == session.session_agenda.user_portrait["Task Inventory"]
    )
    classifier_prompt = mock_invoke.call_args_list[0].args[1]
    assert "Recent loop dialogue" in classifier_prompt
    assert "Are there any tasks you can think of that we haven't covered?" in classifier_prompt
    generator_prompt = mock_invoke.call_args_list[1].args[1]
    assert "Do not repeat or closely paraphrase any recent interviewer question from the loop" in generator_prompt
    assert "Accepted missed task to add: Fix software bugs; answer client questions" in generator_prompt
    assert "start with a brief neutral acknowledgement" in generator_prompt


def test_final_task_probe_generates_varied_loop_question_after_clear_task(client):
    token = "tok-final-probe-generated-loop"
    session, _ = _register_session(
        token,
        "Are there any tasks you can think of that we haven't covered?",
    )
    responses = [
        SimpleNamespace(
            content='{"decision":"clear_task","task_name":"Answer client questions","clarification_question":"","reason":"participant named a clear task"}'
        ),
        SimpleNamespace(
            content='{"reply":"I will add answering client questions. Is there other work that belongs on the task list?"}'
        ),
    ]

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=responses,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "Answering client questions",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is False
    assert body["reply"] == "I will add answering client questions. Is there other work that belongs on the task list?"
    assert body["task_name"] == "Answer client questions"
    assert mock_invoke.call_count == 2
    assert session.chat_history[-1].content == "I will add answering client questions. Is there other work that belongs on the task list?"
    generator_prompt = mock_invoke.call_args_list[1].args[1]
    assert "Recent loop dialogue" in generator_prompt
    assert "Answering client questions" in generator_prompt
    assert "Accepted missed task to add: Answer client questions" in generator_prompt
    assert "Do not repeat or closely paraphrase any recent interviewer question from the loop" in generator_prompt
    assert "Avoid generic repeated templates" in generator_prompt


def test_final_task_probe_only_follows_up_for_action_object_clarification(client):
    token = "tok-final-probe-clarify"
    session, _ = _register_session(
        token,
        "Are there any tasks you can think of that we haven't covered?",
    )
    fake_response = SimpleNamespace(
        content='{"decision":"needs_clarification","task_name":"","clarification_question":"What concrete work does that teamwork involve?","reason":"teamwork solution lacks action and object"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "It's more a teamwork solution",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is False
    assert body["reply"] == "What concrete work does that teamwork involve?"
    assert body["task_name"] is None
    assert mock_invoke.call_count == 1
    assert session._final_task_probe_clarification_pending is True
    assert session.chat_history[-1].role == "Interviewer"
    assert session.chat_history[-1].content == "What concrete work does that teamwork involve?"
    prompt = mock_invoke.call_args_list[0].args[1]
    assert "must NOT include examples, suggestions, or option lists" in prompt


def test_final_task_probe_unclear_input_does_not_advance(client):
    token = "tok-final-probe-unclear"
    session, _ = _register_session(
        token,
        "Are there any tasks you can think of that we haven't covered?",
    )
    fake_response = SimpleNamespace(
        content='{"decision":"unclear_response","task_name":"","clarification_question":"I did not catch that. Are you adding another task, or are you finished with the task list?","reason":"reply looks like accidental text"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "efwwef",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is False
    assert body["reply"] == "I did not catch that. Are you adding another task, or are you finished with the task list?"
    assert body["task_name"] is None
    assert mock_invoke.call_count == 1
    assert session._final_task_probe_loop_active is True
    assert session._final_task_probe_clarification_pending is False
    assert session.chat_history[-1].role == "Interviewer"
    assert session.chat_history[-1].content == body["reply"]
    prompt = mock_invoke.call_args.args[1]
    assert "unclear_response" in prompt
    assert "gibberish" in prompt
    assert "does not answer whether there are more tasks" in prompt


def test_final_task_probe_clarification_answer_returns_to_missed_task_loop(client):
    token = "tok-final-probe-clarified"
    session, _ = _register_session(
        token,
        "What concrete work does that teamwork involve?",
    )
    session._final_task_probe_clarification_pending = True
    fake_response = SimpleNamespace(
        content='{"decision":"clear_task","task_name":"Coordinate team responses to client questions","clarification_question":"","reason":"participant clarified action and object"}'
    )
    generated_response = SimpleNamespace(
        content='{"reply":"I will add coordinating team responses to client questions. Any other tasks we should include?"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=[fake_response, generated_response],
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "We coordinate team responses to client questions",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is False
    assert body["reply"] == "I will add coordinating team responses to client questions. Any other tasks we should include?"
    assert body["task_name"] == "Coordinate team responses to client questions"
    assert mock_invoke.call_count == 2
    assert session._final_task_probe_clarification_pending is False
    assert session._final_task_probe_loop_active is True
    assert session.chat_history[-1].role == "Interviewer"
    assert session.chat_history[-1].content == "I will add coordinating team responses to client questions. Any other tasks we should include?"


def test_final_task_probe_loop_ends_when_user_says_nothing_more(client):
    token = "tok-final-probe-loop-close"
    session, _ = _register_session(
        token,
        "Any other tasks we should include?",
    )
    session._final_task_probe_loop_active = True
    fake_response = SimpleNamespace(
        content='{"decision":"closure","task_name":"","clarification_question":"","reason":"participant says there is nothing more"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "Nothing else",
                "prior_tasks": ["Fix software bugs"],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is True
    assert body["reply"] == ""
    assert body["task_name"] is None
    assert mock_invoke.call_count == 1
    assert session._final_task_probe_loop_active is False


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


def test_ai_task_followup_prompt_stays_ai_scoped(client):
    token = "tok-ai-followup-scope"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"What do you usually check before applying AI-suggested bug fixes?","task_name":"Draft bug fixes using AI to speed implementation","done":false}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "I ask AI for suggested bug fixes.",
                "prior_tasks": ["Review AI-generated code suggestions"],
                "phase": "ai_extras",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    prompt = mock_invoke.call_args.args[1]
    assert "collecting an AI-related task inventory" in prompt
    assert "AI-related tasks already collected:" in prompt
    assert "This phase is ONLY about AI-related tasks" in prompt
    assert "Do NOT broaden to general work tasks" in prompt
    assert "vary the wording and keep it brief" in prompt
    assert "brief neutral acknowledgement" in prompt
    assert "Do not include more than a brief neutral acknowledgement" in prompt
    assert "do not use one every turn" in prompt
    assert "Do NOT infer benefits they did not say" in prompt
    assert "Do NOT reuse the same wording from the previous interviewer turn" in prompt
    assert "Interviewer: Any other AI-related tasks come to mind?" in prompt
    assert "Interviewer: Are there any tasks you can think of that we haven't covered?" not in prompt
    assert "briefly reflects understanding" not in prompt
    assert "Respond briefly to the answer" not in prompt


def test_ai_open_done_suppresses_sycophantic_acknowledgement(client):
    token = "tok-ai-open-neutral"
    session, _ = _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"Got it, using AI to move faster through experiments makes a lot of sense for a PhD.","done":true}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "research projects, a lot of experiments. it doesn't really change it",
                "phase": "ai_open",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is True
    assert body["reply"] == ""
    assert body["open_answer"] == "research projects, a lot of experiments. it doesn't really change it"
    assert session.chat_history[-1].role == "User"
    prompt = mock_invoke.call_args.args[1]
    assert 'Return an EMPTY reply (reply: "")' in prompt
    assert "brief neutral acknowledgement" in prompt
    assert "Do NOT infer benefits they did not say" in prompt
    assert "If they say AI does not change their work, do not contradict" in prompt


def test_ai_task_followup_fallback_question_is_ai_scoped(client):
    token = "tok-ai-followup-fallback"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"That helps","done":false}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ):
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "Mostly for checking generated summaries.",
                "prior_tasks": [],
                "phase": "ai_extras",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["reply"] == "Any other AI-related tasks come to mind?"
    assert "That helps" not in body["reply"]
    assert "other tasks are part of your work" not in body["reply"]


def test_ai_task_followup_strips_evaluative_preamble(client):
    token = "tok-ai-followup-no-preamble"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"Nice that it mostly holds up without heavy revision, so are there other ways AI shows up in your PhD work?","done":false}'
    )
    edited_response = SimpleNamespace(
        content='{"reply":"Are there other ways AI shows up in your PhD work?"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=[fake_response, edited_response],
    ):
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "it lands close enough",
                "prior_tasks": ["Draft grant proposals from scratch using AI"],
                "phase": "ai_extras",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["reply"] == "Are there other ways AI shows up in your PhD work?"
    assert "Nice" not in body["reply"]
    assert "holds up" not in body["reply"]


def test_ai_task_followup_keeps_brief_neutral_acknowledgement(client):
    token = "tok-ai-followup-neutral-ack"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"Got it, are there other ways AI shows up in your PhD work?","done":false}'
    )
    edited_response = SimpleNamespace(
        content='{"reply":"Got it, are there other ways AI shows up in your PhD work?"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=[fake_response, edited_response],
    ):
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "it lands close enough",
                "prior_tasks": ["Draft grant proposals from scratch using AI"],
                "phase": "ai_extras",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["reply"] == "Got it, are there other ways AI shows up in your PhD work?"


def test_ai_task_followup_strips_hyphenated_preamble(client):
    token = "tok-ai-followup-no-hyphen-preamble"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"Using AI for grant proposals sounds useful - what parts do you usually draft with it?","done":false}'
    )
    edited_response = SimpleNamespace(
        content='{"reply":"What parts do you usually draft with it?"}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        side_effect=[fake_response, edited_response],
    ):
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "grant proposals yes",
                "prior_tasks": ["Draft grant proposals using AI"],
                "phase": "ai_extras",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["reply"] == "What parts do you usually draft with it?"
    assert "sounds useful" not in body["reply"]


def test_ai_task_followup_done_prompt_requests_graceful_close(client):
    token = "tok-ai-followup-done"
    session, _ = _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"Thanks, that gives me a complete picture of the AI-related tasks we needed to cover.","done":true}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ) as mock_invoke:
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "that's it",
                "prior_tasks": ["Review AI-generated citations"],
                "phase": "ai_extras",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert body["done"] is True
    assert body["reply"] == ""
    session.end_with_thankyou.assert_called_once()

    prompt = mock_invoke.call_args.args[1]
    assert 'Return JSON: {"reply": "", "done": true}' in prompt


def test_task_followup_reply_strips_em_dash_characters(client):
    token = "tok-followup-no-dash"
    _register_session(token)

    fake_response = SimpleNamespace(
        content='{"reply":"Coding with AI assistance is a big shift — are you mostly using it for implementation?","task_name":"Code software features using AI assistance to ship faster","done":false}'
    )

    with patch("src.utils.llm.engines.get_engine", return_value=MagicMock()), patch(
        "src.utils.llm.engines.invoke_engine",
        return_value=fake_response,
    ):
        res = client.post(
            "/api/task-followup",
            json={
                "session_token": token,
                "task_text": "I use AI to code a lot more now.",
                "prior_tasks": [],
                "phase": "probing",
            },
        )

    body = res.get_json()
    assert res.status_code == 200
    assert body["success"] is True
    assert "—" not in body["reply"]
    assert "–" not in body["reply"]
