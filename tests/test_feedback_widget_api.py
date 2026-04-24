"""API-level tests for feedback widget delivery and submission flow."""

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import src.main_flask as main_flask
from src.interview_session.session_models import MessageType


def _message(mid: str, role: str, content: str, msg_type: str):
    return SimpleNamespace(
        id=mid,
        role=role,
        content=content,
        type=msg_type,
        timestamp=datetime.now(),
    )


def _register_session(session_token: str, session, *, with_loop: bool):
    wrapper = SimpleNamespace(
        session_token=session_token,
        interview_session=session,
        user_id=getattr(session, "user_id", "test_user"),
    )
    if with_loop:
        wrapper.loop = MagicMock()
    main_flask.active_sessions[session_token] = wrapper
    return wrapper


@pytest.fixture
def client():
    main_flask.active_sessions.clear()
    main_flask.chat_history_offsets.clear()
    main_flask.session_audio_cache.clear()
    main_flask.last_messages_by_session.clear()
    main_flask.task_followup_history_by_session.clear()

    main_flask.app.config["TESTING"] = True
    with main_flask.app.test_client() as test_client:
        yield test_client

    main_flask.active_sessions.clear()
    main_flask.chat_history_offsets.clear()
    main_flask.session_audio_cache.clear()
    main_flask.last_messages_by_session.clear()
    main_flask.task_followup_history_by_session.clear()


class TestEndSessionApi:
    def test_end_session_dispatches_feedback_widget_on_session_loop(self, client):
        token = "tok-end-loop"
        session = SimpleNamespace(
            trigger_feedback_widget=MagicMock(),
            session_id=123,
            user_id="u123",
        )
        wrapper = _register_session(token, session, with_loop=True)

        res = client.post("/api/end-session", json={"session_token": token})
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True
        assert body["awaiting_feedback"] is True
        wrapper.loop.call_soon_threadsafe.assert_called_once_with(
            session.trigger_feedback_widget
        )
        session.trigger_feedback_widget.assert_not_called()

    def test_end_session_calls_trigger_directly_without_loop(self, client):
        token = "tok-end-direct"
        session = SimpleNamespace(
            trigger_feedback_widget=MagicMock(),
            session_id=456,
            user_id="u456",
        )
        _register_session(token, session, with_loop=False)

        res = client.post("/api/end-session", json={"session_token": token})
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True
        assert body["awaiting_feedback"] is True
        session.trigger_feedback_widget.assert_called_once_with()


class TestSubmitFeedbackApi:
    def test_submit_feedback_rejects_missing_feedback_payload(self, client):
        token = "tok-submit-invalid"
        session = SimpleNamespace(
            submit_feedback=MagicMock(),
            session_id=789,
            user_id="u789",
        )
        _register_session(token, session, with_loop=False)

        res = client.post("/api/submit-feedback", json={"session_token": token})
        body = res.get_json()

        assert res.status_code == 400
        assert body["success"] is False
        assert "session_token and feedback required" in body["error"]
        session.submit_feedback.assert_not_called()

    def test_submit_feedback_dispatches_on_session_loop(self, client):
        token = "tok-submit-loop"
        feedback = {"rating": 5, "comment": "Great session."}
        session = SimpleNamespace(
            submit_feedback=MagicMock(),
            session_id=1001,
            user_id="u1001",
        )
        wrapper = _register_session(token, session, with_loop=True)

        res = client.post(
            "/api/submit-feedback",
            json={"session_token": token, "feedback": feedback},
        )
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True
        wrapper.loop.call_soon_threadsafe.assert_called_once_with(
            session.submit_feedback,
            feedback,
        )
        session.submit_feedback.assert_not_called()

    def test_submit_feedback_calls_submit_directly_without_loop(self, client):
        token = "tok-submit-direct"
        feedback = {"rating": 4}
        session = SimpleNamespace(
            submit_feedback=MagicMock(),
            session_id=1002,
            user_id="u1002",
        )
        _register_session(token, session, with_loop=False)

        res = client.post(
            "/api/submit-feedback",
            json={"session_token": token, "feedback": feedback},
        )
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True
        session.submit_feedback.assert_called_once_with(feedback)


class TestGetMessagesReplayForWidgets:
    def test_full_history_replays_feedback_and_profile_widgets(self, client):
        token = "tok-full-replay"

        session = SimpleNamespace(
            user=SimpleNamespace(get_new_messages=lambda: []),
            chat_history=[
                _message(
                    "m1",
                    "User",
                    "Conversation item",
                    MessageType.CONVERSATION,
                ),
                _message(
                    "m2",
                    "Interviewer",
                    "",
                    MessageType.TIME_SPLIT_WIDGET,
                ),
                _message(
                    "m3",
                    "Interviewer",
                    "",
                    MessageType.FEEDBACK_WIDGET,
                ),
                _message(
                    "m4",
                    "Interviewer",
                    "",
                    MessageType.PROFILE_CONFIRM_WIDGET,
                ),
            ],
            session_completed=False,
            session_in_progress=True,
            turns=1,
            max_turns=20,
        )
        _register_session(token, session, with_loop=False)

        res = client.get(f"/api/get-messages?session_token={token}&full=true")
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True

        returned_types = [m["type"] for m in body["messages"]]
        assert MessageType.CONVERSATION in returned_types
        assert MessageType.FEEDBACK_WIDGET in returned_types
        assert MessageType.PROFILE_CONFIRM_WIDGET in returned_types
        assert MessageType.TIME_SPLIT_WIDGET not in returned_types


class TestOrganizeTasksApi:
    def test_organize_tasks_rejects_non_list_tasks(self, client):
        token = "tok-organize-bad"
        session = SimpleNamespace(session_id=2001, user_id="u2001")
        _register_session(token, session, with_loop=False)

        res = client.post(
            "/api/organize-tasks",
            json={"session_token": token, "tasks": "not-a-list"},
        )
        body = res.get_json()

        assert res.status_code == 400
        assert body["success"] is False
        assert "tasks must be a list" in body["error"]

    def test_organize_tasks_forwards_grouping_feedback(self, client):
        token = "tok-organize-feedback"
        session = SimpleNamespace(session_id=2002, user_id="u2002")
        _register_session(token, session, with_loop=False)

        mock_tree = [{"name": "writing docs", "children": []}]
        with patch.object(
            main_flask,
            "_organize_tasks_cached",
            return_value=(mock_tree, False),
        ) as mock_cached:
            res = client.post(
                "/api/organize-tasks",
                json={
                    "session_token": token,
                    "tasks": ["writing docs", "meeting teammates"],
                    "grouping_feedback": "group meeting tasks under collaboration",
                },
            )
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True
        assert body["grouping_feedback_applied"] is True
        assert body["acknowledgement"] == "Thanks - I regrouped the subtasks using your feedback."
        assert body["message"] == body["acknowledgement"]
        assert body["tree"] == mock_tree
        mock_cached.assert_called_once_with(
            ["writing docs", "meeting teammates"],
            grouping_feedback="group meeting tasks under collaboration",
        )

    def test_organize_tasks_returns_generic_acknowledgement_without_feedback(self, client):
        token = "tok-organize-no-feedback"
        session = SimpleNamespace(session_id=2003, user_id="u2003")
        _register_session(token, session, with_loop=False)

        mock_tree = [{"name": "task a", "children": []}]
        with patch.object(
            main_flask,
            "_organize_tasks_cached",
            return_value=(mock_tree, True),
        ) as mock_cached:
            res = client.post(
                "/api/organize-tasks",
                json={
                    "session_token": token,
                    "tasks": ["task a", "task b"],
                },
            )
        body = res.get_json()

        assert res.status_code == 200
        assert body["success"] is True
        assert body["grouping_feedback_applied"] is False
        assert body["acknowledgement"] == "Thanks - I organized your subtasks."
        assert body["message"] == body["acknowledgement"]
        assert body["cached"] is True
        assert body["tree"] == mock_tree
        mock_cached.assert_called_once_with(
            ["task a", "task b"],
            grouping_feedback="",
        )

    def test_cached_organizer_enables_screening(self, client):
        # Regression: task list organization should run with screen=True so
        # context-only entries (e.g., "working from home") can be filtered.
        main_flask._TASK_TREE_CACHE.clear()
        main_flask._TASK_TREE_INFLIGHT.clear()

        tasks = ["working from home", "writing grants to secure funding"]
        with patch(
            "src.utils.task_hierarchy.organize_tasks",
            create=True,
            autospec=True,
            return_value=[{"name": "writing grants to secure funding", "children": []}],
        ) as mock_organize:
            main_flask._organize_tasks_cached(tasks, grouping_feedback="")

        mock_organize.assert_called_once_with(
            tasks,
            model_name=main_flask._TASK_HIERARCHY_MODEL_NAME,
            screen=True,
            grouping_feedback="",
            append_uncovered_tasks=False,
        )
