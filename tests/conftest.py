"""
Shared fixtures for widget-trigger tests.

Strategy: avoid real LLM/FAISS/logging I/O by patching at the boundary.
The fixtures build a real InterviewTopicManager from topics_intake.json so
coverage logic runs against genuine subtopic objects.
"""
import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.content.session_agenda.interview_topic_manager import InterviewTopicManager
from src.content.memory_bank.memory_bank_base import MemoryBankBase
from src.interview_session.session_models import MessageType

INTAKE_CONFIG = Path(__file__).parent.parent / "configs" / "topics_intake.json"


class _StubMemoryBank(MemoryBankBase):
    """Minimal in-memory stub that satisfies the Recall tool's isinstance check."""
    memories = []

    def add_memory(self, memory): pass
    def update_memory(self, memory_id, **kwargs): pass
    def search_memories(self, query, k=5): return []
    def _save_implementation_specific(self, path): pass
    def _load_implementation_specific(self, user_id, base_path=None): pass
    def save_to_file(self, user_id): pass
    def set_session_id(self, session_id): pass

    @classmethod
    def load_from_file(cls, user_id): return cls()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_topic_manager() -> InterviewTopicManager:
    with open(INTAKE_CONFIG) as f:
        plan = json.load(f)
    return InterviewTopicManager.init_from_interview_plan(plan)


def make_fake_session(topic_manager: InterviewTopicManager):
    """Return a minimal fake InterviewSession with real topic coverage state."""
    chat_history = []

    def _add_message(role="", content="", message_type=MessageType.CONVERSATION, metadata={}):
        chat_history.append({
            "role": role,
            "content": content,
            "message_type": message_type,
        })

    session = MagicMock()
    session.memory_bank = _StubMemoryBank()
    session.session_type = "intake"
    session.session_in_progress = True
    session._session_ending = False
    session._feedback_widget_sent = False
    session._profile_confirm_widget_sent = False
    session._widget_pending_subtopic_id = None
    session.chat_history = chat_history
    session.add_message_to_chat_history.side_effect = _add_message

    # Wire real trigger methods so the flag + message logic actually runs
    def _trigger_profile_confirm():
        if session._profile_confirm_widget_sent:
            return
        session._profile_confirm_widget_sent = True
        _add_message(role="Interviewer", content="", message_type=MessageType.PROFILE_CONFIRM_WIDGET)

    def _trigger_feedback():
        if session._feedback_widget_sent:
            return
        session._feedback_widget_sent = True
        _add_message(role="Interviewer", content="", message_type=MessageType.FEEDBACK_WIDGET)

    session.trigger_profile_confirm_widget.side_effect = _trigger_profile_confirm
    session.trigger_feedback_widget.side_effect = _trigger_feedback

    # Real topic manager so coverage checks work
    agenda = MagicMock()
    agenda.interview_topic_manager = topic_manager
    agenda.user_portrait = {}
    agenda.available_time_minutes = None
    session.session_agenda = agenda

    # Scribe stub — needed by _refresh_portrait_before_widget
    scribe = MagicMock()
    scribe.processing_in_progress = False
    session.session_scribe = scribe

    session._portrait_update_lock = asyncio.Lock()
    # async methods called by _on_message_body
    session.wait_if_paused = AsyncMock()
    session._generate_and_save_user_portrait_inner = AsyncMock()

    # Strategic planner stub (used by _should_include_strategic_questions)
    strategic_state = MagicMock()
    strategic_state.strategic_question_suggestions = []
    strategic_state.last_planning_turn = 0
    planner = MagicMock()
    planner.strategic_state = strategic_state
    planner.rollout_horizon = 3
    session.strategic_planner = planner

    return session


@pytest.fixture(autouse=True)
def silence_session_logger():
    """Prevent SessionLogger.log_to_file from raising RuntimeError in tests."""
    with patch("src.utils.logger.session_logger.SessionLogger.log_to_file", return_value=None):
        yield


@pytest.fixture
def topic_manager():
    return load_topic_manager()


@pytest.fixture
def fake_session(topic_manager):
    return make_fake_session(topic_manager)


@pytest.fixture
def interviewer(fake_session):
    """Interviewer with mocked engine — no real LLM calls."""
    with patch("src.agents.base_agent.get_engine", return_value=MagicMock()):
        from src.agents.interviewer.interviewer import Interviewer, InterviewerConfig
        iv = Interviewer(
            config=InterviewerConfig(
                user_id="test_user",
                tts={},
                interview_description="test session",
            ),
            interview_session=fake_session,
        )
    # Patch engine so call_engine_async never hits a real API
    iv.call_engine_async = AsyncMock()
    # Patch portrait refresh to be a no-op (avoids event-loop complexity)
    iv._refresh_portrait_before_widget = AsyncMock()
    return iv
