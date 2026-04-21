"""
Tests for widget trigger logic.

Covers three widgets:
  - profile_confirm_widget  (after first topic covered)
  - feedback_widget         (end_conversation tool + safety net)
  - time_split_widget       (time-allocation subtopic question)

Each test uses a real InterviewTopicManager so coverage flags reflect genuine
subtopic objects, but all LLM / IO calls are mocked out.
"""
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from src.interview_session.session_models import MessageType
from tests.conftest import make_fake_session, load_topic_manager


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def widget_types(session):
    return [m["message_type"] for m in session.chat_history]


def mark_subtopic_covered(topic_manager, subtopic_id: str, covered: bool = True):
    for topic in topic_manager:
        st = topic.get_subtopic(subtopic_id)
        if st is not None:
            st.is_covered = covered
            return
    raise ValueError(f"subtopic_id {subtopic_id!r} not found in topic_manager")


def mark_first_topic_covered(topic_manager):
    first_topic = list(topic_manager)[0]
    for st in first_topic.required_subtopics.values():
        st.is_covered = True


def mark_all_topics_covered(topic_manager):
    for topic in topic_manager:
        for st in topic.required_subtopics.values():
            st.is_covered = True


def find_time_allocation_subtopic_id(topic_manager) -> str:
    for topic in topic_manager:
        for st in topic.required_subtopics.values():
            if "time allocation" in str(st.description).lower():
                return str(st.subtopic_id)
    raise ValueError("time-allocation subtopic not found")


def find_non_time_allocation_subtopic_id(topic_manager) -> str:
    for topic in topic_manager:
        for st in topic.required_subtopics.values():
            if "time allocation" not in str(st.description).lower():
                return str(st.subtopic_id)
    raise ValueError("non-time-allocation subtopic not found")


def find_non_first_topic_subtopic_id(topic_manager) -> str:
    topics = list(topic_manager)
    if len(topics) < 2:
        raise ValueError("expected at least two topics")
    for topic in topics[1:]:
        for st in topic.required_subtopics.values():
            return str(st.subtopic_id)
    raise ValueError("non-first-topic subtopic not found")


# ---------------------------------------------------------------------------
# Unit: trigger_profile_confirm_widget
# ---------------------------------------------------------------------------

class TestProfileConfirmWidget:

    def test_emits_message_on_first_call(self, fake_session):
        from src.interview_session.interview_session import InterviewSession
        # Call the real method directly via side_effect wiring in fixture
        fake_session.trigger_profile_confirm_widget()
        assert MessageType.PROFILE_CONFIRM_WIDGET in widget_types(fake_session)

    def test_idempotent_second_call_ignored(self, fake_session):
        fake_session.trigger_profile_confirm_widget()
        fake_session.trigger_profile_confirm_widget()
        count = widget_types(fake_session).count(MessageType.PROFILE_CONFIRM_WIDGET)
        assert count == 1

    def test_sets_sent_flag(self, fake_session):
        fake_session.trigger_profile_confirm_widget()
        assert fake_session._profile_confirm_widget_sent is True


# ---------------------------------------------------------------------------
# Unit: trigger_feedback_widget
# ---------------------------------------------------------------------------

class TestFeedbackWidget:

    def test_emits_message(self, fake_session):
        fake_session.trigger_feedback_widget()
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)

    def test_idempotent(self, fake_session):
        fake_session.trigger_feedback_widget()
        fake_session.trigger_feedback_widget()
        count = widget_types(fake_session).count(MessageType.FEEDBACK_WIDGET)
        assert count == 1

    def test_sets_sent_flag(self, fake_session):
        fake_session.trigger_feedback_widget()
        assert fake_session._feedback_widget_sent is True


# ---------------------------------------------------------------------------
# Integration: _handle_response fires profile_confirm_widget
# ---------------------------------------------------------------------------

class TestHandleResponseProfileConfirm:

    @pytest.mark.asyncio
    async def test_fires_when_first_topic_covered(self, interviewer, fake_session, topic_manager):
        mark_first_topic_covered(topic_manager)
        await interviewer._handle_response("Walk me through a typical week.", subtopic_id="2.1")
        assert MessageType.PROFILE_CONFIRM_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_does_not_fire_when_first_topic_not_covered(self, interviewer, fake_session):
        # All subtopics start uncovered — no widget should fire
        await interviewer._handle_response("What's your role?", subtopic_id="1.1")
        assert MessageType.PROFILE_CONFIRM_WIDGET not in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_fires_when_transitioning_to_non_first_topic_even_if_coverage_lags(
        self, interviewer, fake_session, topic_manager
    ):
        # Regression: coverage can lag briefly while scribe work completes.
        # If the outgoing question is already outside Topic 1, emit the widget.
        subtopic_id = find_non_first_topic_subtopic_id(topic_manager)
        await interviewer._handle_response(
            "Walk me through a typical week.", subtopic_id=subtopic_id
        )
        assert MessageType.PROFILE_CONFIRM_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_widget_appears_before_conversation_message(self, interviewer, fake_session, topic_manager):
        mark_first_topic_covered(topic_manager)
        await interviewer._handle_response("Walk me through a typical week.", subtopic_id="2.1")
        types = widget_types(fake_session)
        widget_idx = types.index(MessageType.PROFILE_CONFIRM_WIDGET)
        # The conversation message added right after must come later
        conv_indices = [i for i, t in enumerate(types) if t == MessageType.CONVERSATION]
        assert conv_indices, "expected at least one conversation message"
        assert widget_idx < conv_indices[-1]

    @pytest.mark.asyncio
    async def test_does_not_fire_for_weekly_session(self, interviewer, fake_session, topic_manager):
        fake_session.session_type = "weekly"
        mark_first_topic_covered(topic_manager)
        await interviewer._handle_response("What did you work on this week?", subtopic_id="2.1")
        assert MessageType.PROFILE_CONFIRM_WIDGET not in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_fires_only_once_across_multiple_calls(self, interviewer, fake_session, topic_manager):
        mark_first_topic_covered(topic_manager)
        await interviewer._handle_response("Walk me through a typical week.", subtopic_id="2.1")
        await interviewer._handle_response("What tasks take the most time?", subtopic_id="2.2")
        count = widget_types(fake_session).count(MessageType.PROFILE_CONFIRM_WIDGET)
        assert count == 1

    def test_fast_opening_role_subtopic_lookup_matches_current_wording(
        self, interviewer, topic_manager
    ):
        sid = interviewer._get_role_title_subtopic_id()
        assert sid
        first_topic = list(topic_manager)[0]
        first_topic_ids = {str(st.subtopic_id) for st in first_topic.required_subtopics.values()}
        assert sid in first_topic_ids

    @pytest.mark.asyncio
    async def test_on_message_waits_for_scribe_before_profile_confirm_transition(
        self, interviewer, fake_session, topic_manager
    ):
        # Simulate the transition turn:
        # - first topic is almost covered (1.1 + 1.2 done, 1.3 pending)
        # - LLM labels the outgoing question as 1.3 even though it's a task-inventory ask
        # - scribe marks 1.3 covered slightly later
        # Regression: if interviewer early-exits before scribe completes, the
        # profile widget appears one turn late.
        mark_subtopic_covered(topic_manager, "1.1", covered=True)
        mark_subtopic_covered(topic_manager, "1.2", covered=True)
        mark_subtopic_covered(topic_manager, "1.3", covered=False)
        fake_session.session_scribe.processing_in_progress = True

        async def finish_scribe_later():
            await asyncio.sleep(1.25)
            mark_subtopic_covered(topic_manager, "1.3", covered=True)
            fake_session.session_scribe.processing_in_progress = False

        settle_task = asyncio.create_task(finish_scribe_later())

        interviewer.call_engine_async.return_value = (
            "<tool_calls>"
            "<respond_to_user>"
            "<subtopic_id>1.3</subtopic_id>"
            "<response>Can you walk me through what a typical work week looks like for you?</response>"
            "</respond_to_user>"
            "</tool_calls>"
        )

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime

        user_msg = Message(
            id="u1",
            type=MessageType.CONVERSATION,
            role="User",
            content="I work in information systems with an AI/labor focus.",
            timestamp=datetime.now(),
            metadata={},
        )
        await interviewer._on_message_body(user_msg)
        await settle_task

        types = widget_types(fake_session)
        assert MessageType.PROFILE_CONFIRM_WIDGET in types
        widget_idx = types.index(MessageType.PROFILE_CONFIRM_WIDGET)
        conv_indices = [i for i, t in enumerate(types) if t == MessageType.CONVERSATION]
        assert conv_indices, "expected interviewer conversation message"
        assert widget_idx < conv_indices[-1]

    @pytest.mark.asyncio
    async def test_profile_confirm_waits_for_coverage_but_not_memory_tail(
        self, interviewer, fake_session, topic_manager
    ):
        # Start in a near-transition state: first topic almost done.
        mark_subtopic_covered(topic_manager, "1.1", covered=True)
        mark_subtopic_covered(topic_manager, "1.2", covered=True)
        mark_subtopic_covered(topic_manager, "1.3", covered=False)

        scribe = fake_session.session_scribe
        scribe.processing_in_progress = True
        # New fast-path signal used during profile-confirm transition.
        scribe.coverage_processing_in_progress = True

        async def complete_coverage_then_leave_memory_running():
            # Coverage completes quickly.
            await asyncio.sleep(0.35)
            mark_subtopic_covered(topic_manager, "1.3", covered=True)
            scribe.coverage_processing_in_progress = False
            # Memory tail remains busy for longer.
            await asyncio.sleep(1.8)
            scribe.processing_in_progress = False

        settle_task = asyncio.create_task(complete_coverage_then_leave_memory_running())

        interviewer.call_engine_async.return_value = (
            "<tool_calls>"
            "<respond_to_user>"
            "<subtopic_id>1.3</subtopic_id>"
            "<response>Can you walk me through what a typical work week looks like for you?</response>"
            "</respond_to_user>"
            "</tool_calls>"
        )

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime

        user_msg = Message(
            id="u1",
            type=MessageType.CONVERSATION,
            role="User",
            content="I work in information systems with an AI/labor focus.",
            timestamp=datetime.now(),
            metadata={},
        )

        start = asyncio.get_event_loop().time()
        await interviewer._on_message_body(user_msg)
        elapsed = asyncio.get_event_loop().time() - start
        await settle_task

        # Should proceed shortly after coverage completion, without waiting for
        # the full memory-processing tail (~2.15s in this test setup).
        assert elapsed < 1.6
        assert MessageType.PROFILE_CONFIRM_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_non_profile_wait_still_respects_full_scribe_processing(
        self, interviewer, fake_session, topic_manager
    ):
        # Simulate a post-profile-confirm turn, where we still require full
        # scribe completion (including memory tail work) before proceeding.
        fake_session._profile_confirm_widget_sent = True
        mark_first_topic_covered(topic_manager)

        scribe = fake_session.session_scribe
        scribe.processing_in_progress = True
        scribe.coverage_processing_in_progress = False

        async def finish_memory_tail_later():
            await asyncio.sleep(0.95)
            scribe.processing_in_progress = False

        settle_task = asyncio.create_task(finish_memory_tail_later())

        interviewer.call_engine_async.return_value = (
            "<tool_calls>"
            "<respond_to_user>"
            "<subtopic_id>2.1</subtopic_id>"
            "<response>Can you walk me through what a typical work week looks like for you?</response>"
            "</respond_to_user>"
            "</tool_calls>"
        )

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime

        user_msg = Message(
            id="u2",
            type=MessageType.CONVERSATION,
            role="User",
            content="I usually split my week across teaching and research.",
            timestamp=datetime.now(),
            metadata={},
        )

        start = asyncio.get_event_loop().time()
        await interviewer._on_message_body(user_msg)
        elapsed = asyncio.get_event_loop().time() - start
        await settle_task

        # Should not short-circuit off coverage-only signal for this path.
        assert elapsed >= 0.85


class TestSpeculativeReuseLatency:

    @pytest.mark.asyncio
    async def test_keeps_speculative_when_coverage_changed_but_target_still_uncovered(
        self, interviewer, fake_session, topic_manager
    ):
        # Post-profile-confirm path so we run full_scribe wait logic.
        fake_session._profile_confirm_widget_sent = True

        scribe = fake_session.session_scribe
        scribe.processing_in_progress = True
        scribe.coverage_processing_in_progress = False

        async def update_unrelated_coverage_then_release():
            await asyncio.sleep(0.2)
            # Coverage changes, but not for the speculative target subtopic.
            mark_subtopic_covered(topic_manager, "1.1", covered=True)
            scribe.processing_in_progress = False

        settle_task = asyncio.create_task(update_unrelated_coverage_then_release())

        interviewer.call_engine_async.return_value = (
            "<tool_calls>"
            "<respond_to_user>"
            "<subtopic_id>2.1</subtopic_id>"
            "<response>Can you walk me through what a typical week looks like for you?</response>"
            "</respond_to_user>"
            "</tool_calls>"
        )

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime

        user_msg = Message(
            id="u-spec-keep",
            type=MessageType.CONVERSATION,
            role="User",
            content="I usually split my time between experiments and writing.",
            timestamp=datetime.now(),
            metadata={},
        )

        await interviewer._on_message_body(user_msg)
        await settle_task

        # Regression: previously any coverage change forced a second LLM call.
        assert interviewer.call_engine_async.await_count == 1

    @pytest.mark.asyncio
    async def test_discards_speculative_when_target_subtopic_becomes_covered(
        self, interviewer, fake_session, topic_manager
    ):
        fake_session._profile_confirm_widget_sent = True

        scribe = fake_session.session_scribe
        scribe.processing_in_progress = True
        scribe.coverage_processing_in_progress = False

        async def cover_target_then_release():
            await asyncio.sleep(0.2)
            mark_subtopic_covered(topic_manager, "1.1", covered=True)
            scribe.processing_in_progress = False

        settle_task = asyncio.create_task(cover_target_then_release())

        interviewer.call_engine_async.side_effect = [
            (
                "<tool_calls>"
                "<respond_to_user>"
                "<subtopic_id>1.1</subtopic_id>"
                "<response>How would you describe your role?</response>"
                "</respond_to_user>"
                "</tool_calls>"
            ),
            (
                "<tool_calls>"
                "<respond_to_user>"
                "<subtopic_id>2.1</subtopic_id>"
                "<response>Can you walk me through what a typical week looks like for you?</response>"
                "</respond_to_user>"
                "</tool_calls>"
            ),
        ]

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime

        user_msg = Message(
            id="u-spec-drop",
            type=MessageType.CONVERSATION,
            role="User",
            content="I'm a postdoc in the digital economy lab.",
            timestamp=datetime.now(),
            metadata={},
        )

        await interviewer._on_message_body(user_msg)
        await settle_task

        assert interviewer.call_engine_async.await_count == 2


# ---------------------------------------------------------------------------
# Integration: time_split_widget fires for time-allocation subtopic
# ---------------------------------------------------------------------------

class TestTimeSplitWidget:

    @pytest.mark.asyncio
    async def test_fires_for_time_allocation_subtopic(self, interviewer, fake_session, topic_manager):
        subtopic_id = find_time_allocation_subtopic_id(topic_manager)
        await interviewer._handle_response(
            "Here's a quick widget to fill in your time allocation.", subtopic_id=subtopic_id
        )
        assert MessageType.TIME_SPLIT_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_first_time_allocation_question_explicitly_mentions_widget(
        self, interviewer, fake_session, topic_manager
    ):
        subtopic_id = find_time_allocation_subtopic_id(topic_manager)
        await interviewer._handle_response(
            "Since your job market paper is the main thing right now, how would you roughly "
            "break down your work time across everything you listed as percentages that add "
            "up to about 100%?",
            subtopic_id=subtopic_id,
        )
        interviewer_msgs = [
            m for m in fake_session.chat_history
            if m["message_type"] == MessageType.CONVERSATION and m["role"] == "Interviewer"
        ]
        assert interviewer_msgs, "expected an interviewer conversation message"
        assert "widget" in interviewer_msgs[-1]["content"].lower()

    @pytest.mark.asyncio
    async def test_fires_only_once(self, interviewer, fake_session, topic_manager):
        subtopic_id = find_time_allocation_subtopic_id(topic_manager)
        await interviewer._handle_response("Time widget.", subtopic_id=subtopic_id)
        await interviewer._handle_response("Follow-up question.", subtopic_id=subtopic_id)
        count = widget_types(fake_session).count(MessageType.TIME_SPLIT_WIDGET)
        assert count == 1

    @pytest.mark.asyncio
    async def test_does_not_fire_for_other_subtopics(self, interviewer, fake_session, topic_manager):
        subtopic_id = find_non_time_allocation_subtopic_id(topic_manager)
        await interviewer._handle_response("What tasks do you do?", subtopic_id=subtopic_id)
        assert MessageType.TIME_SPLIT_WIDGET not in widget_types(fake_session)


# ---------------------------------------------------------------------------
# Integration: feedback_widget safety net (_on_message_body)
# ---------------------------------------------------------------------------

class TestFeedbackWidgetSafetyNet:
    """
    When all topics are covered but the LLM uses respond_to_user (not
    end_conversation) for the goodbye, the safety net at the end of
    _on_message_body must fire the feedback widget.
    """

    GOODBYE_VIA_RESPOND = (
        '<tool_calls>'
        '<respond_to_user>'
        '<subtopic_id>2.3</subtopic_id>'
        '<response>Thanks so much — we\'re all set, appreciate you walking through that!</response>'
        '</respond_to_user>'
        '</tool_calls>'
    )

    @pytest.mark.asyncio
    async def test_safety_net_fires_when_all_covered_no_end_conversation(
        self, interviewer, fake_session, topic_manager
    ):
        mark_all_topics_covered(topic_manager)
        # Scribe is not running (processing_in_progress=False already)
        interviewer.call_engine_async.return_value = self.GOODBYE_VIA_RESPOND

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="That covers about 30%.", timestamp=datetime.now(), metadata={}
        )
        await interviewer._on_message_body(user_msg)
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_safety_net_does_not_fire_when_topics_incomplete(
        self, interviewer, fake_session, topic_manager
    ):
        # Only first topic covered — Task Inventory still open
        mark_first_topic_covered(topic_manager)
        interviewer.call_engine_async.return_value = (
            '<tool_calls>'
            '<respond_to_user>'
            '<subtopic_id>2.1</subtopic_id>'
            '<response>Walk me through a typical week.</response>'
            '</respond_to_user>'
            '</tool_calls>'
        )
        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="I'm a PhD student.", timestamp=datetime.now(), metadata={}
        )
        await interviewer._on_message_body(user_msg)
        assert MessageType.FEEDBACK_WIDGET not in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_safety_net_fires_on_goodbye_intent_even_if_coverage_lags(
        self, interviewer, fake_session, topic_manager
    ):
        # Regression: when coverage flags lag, the model can send a closing
        # message through respond_to_user before all topics are marked covered.
        # The safety net should still emit feedback_widget.
        mark_first_topic_covered(topic_manager)
        interviewer.call_engine_async.return_value = (
            '<tool_calls>'
            '<respond_to_user>'
            '<subtopic_id>2.4</subtopic_id>'
            '<response>'
            "Thanks for laying that out so clearly. That's everything I needed on my side, so we're all set here."
            '</response>'
            '</respond_to_user>'
            '</tool_calls>'
        )
        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="30% meetings, 30% analysis, 40% writing.", timestamp=datetime.now(), metadata={}
        )
        await interviewer._on_message_body(user_msg)
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_safety_net_does_not_fire_after_follow_up_question(
        self, interviewer, fake_session, topic_manager
    ):
        mark_all_topics_covered(topic_manager)
        interviewer.call_engine_async.return_value = (
            '<tool_calls>'
            '<respond_to_user>'
            '<subtopic_id>2.3</subtopic_id>'
            '<response>'
            'When you think about those four buckets - meetings, experiments, reading, and writeups - '
            'does that breakdown feel typical, or does anything major shift those percentages around?'
            '</response>'
            '</respond_to_user>'
            '</tool_calls>'
        )
        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="That seems about right.", timestamp=datetime.now(), metadata={}
        )
        await interviewer._on_message_body(user_msg)
        assert MessageType.FEEDBACK_WIDGET not in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_safety_net_does_not_fire_when_session_already_ending(
        self, interviewer, fake_session, topic_manager
    ):
        mark_all_topics_covered(topic_manager)
        fake_session._session_ending = True
        interviewer.call_engine_async.return_value = self.GOODBYE_VIA_RESPOND

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="Thanks!", timestamp=datetime.now(), metadata={}
        )
        await interviewer._on_message_body(user_msg)
        assert MessageType.FEEDBACK_WIDGET not in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_no_active_topics_ends_without_extra_followup(
        self, interviewer, fake_session, topic_manager
    ):
        # Regression: if active_topic_id_list is empty, prompt rendering can
        # produce an empty <topics_list>. In that state the interviewer should
        # close deterministically, not invent another question.
        mark_all_topics_covered(topic_manager)
        topic_manager.active_topic_id_list = []
        interviewer.call_engine_async.return_value = (
            '<tool_calls>'
            '<respond_to_user>'
            '<subtopic_id>2.4</subtopic_id>'
            '<response>Does that feel representative of most weeks?</response>'
            '</respond_to_user>'
            '</tool_calls>'
        )

        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="yes", timestamp=datetime.now(), metadata={}
        )

        await interviewer._on_message_body(user_msg)

        interviewer_conversations = [
            m for m in fake_session.chat_history
            if m["role"] == "Interviewer" and m["message_type"] == MessageType.CONVERSATION
        ]
        assert interviewer_conversations, "expected an interviewer closing message"
        assert "?" not in interviewer_conversations[-1]["content"]
        assert "covers what I needed for this intake" in interviewer_conversations[-1]["content"]
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)


# ---------------------------------------------------------------------------
# Integration: end_conversation tool triggers feedback_widget
# ---------------------------------------------------------------------------

class TestEndConversationTool:

    @pytest.mark.asyncio
    async def test_end_conversation_triggers_feedback_widget(
        self, interviewer, fake_session
    ):
        interviewer.call_engine_async.return_value = (
            '<tool_calls>'
            '<end_conversation>'
            '<goodbye>Thanks so much for your time — we\'re all wrapped up!</goodbye>'
            '</end_conversation>'
            '</tool_calls>'
        )
        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="Sure, that covers it.", timestamp=datetime.now(), metadata={}
        )
        await interviewer._on_message_body(user_msg)
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)


# ---------------------------------------------------------------------------
# Regression: mixed free-text + JSON tool payload should not leak raw JSON
# ---------------------------------------------------------------------------

class TestToolPayloadLeakRegression:

    @pytest.mark.asyncio
    async def test_mixed_text_plus_goodbye_json_is_sanitized(
        self, interviewer, fake_session, topic_manager
    ):
        mark_all_topics_covered(topic_manager)
        goodbye = (
            "Those percentages you shared give a really clear picture of how your "
            "PhD workweek actually looks."
        )
        interviewer.call_engine_async.return_value = (
            'Thanks for breaking that down. That actually covers everything I needed, so we are all set."}\n'
            f'{{"goodbye":"{goodbye}"}}'
        )
        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="sounds good", timestamp=datetime.now(), metadata={}
        )

        await interviewer._on_message_body(user_msg)

        interviewer_conversations = [
            m for m in fake_session.chat_history
            if m["role"] == "Interviewer" and m["message_type"] == MessageType.CONVERSATION
        ]
        assert interviewer_conversations, "expected at least one interviewer conversation message"
        assert interviewer_conversations[-1]["content"] == goodbye
        assert not any(
            '"goodbye":' in m["content"] or '{"goodbye"' in m["content"]
            for m in interviewer_conversations
        )
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)

    @pytest.mark.asyncio
    async def test_line_delimited_response_then_goodbye_prefers_goodbye(
        self, interviewer, fake_session
    ):
        goodbye = (
            "Those percentages you just shared give a clear picture of how your "
            "week is distributed. I will stop here and thank you for your time."
        )
        interviewer.call_engine_async.return_value = (
            '{"subtopic_id":"2.4","response":"Thanks for laying that out clearly."}\n'
            f'{{"goodbye":"{goodbye}"}}'
        )
        from src.interview_session.session_models import Message, MessageType
        from datetime import datetime
        user_msg = Message(
            id="u1", type=MessageType.CONVERSATION, role="User",
            content="sounds good", timestamp=datetime.now(), metadata={}
        )

        await interviewer._on_message_body(user_msg)

        interviewer_conversations = [
            m for m in fake_session.chat_history
            if m["role"] == "Interviewer" and m["message_type"] == MessageType.CONVERSATION
        ]
        assert interviewer_conversations, "expected at least one interviewer conversation message"
        assert interviewer_conversations[-1]["content"] == goodbye
        assert MessageType.FEEDBACK_WIDGET in widget_types(fake_session)


# ---------------------------------------------------------------------------
# Regression: real add_message_to_chat_history must deliver profile widget
# ---------------------------------------------------------------------------

class TestProfileWidgetDeliveryRegression:
    def test_profile_widget_is_appended_and_notified_by_real_method(self):
        from datetime import datetime
        from src.interview_session.interview_session import InterviewSession

        session = InterviewSession.__new__(InterviewSession)
        session.session_in_progress = True
        session.chat_history = []
        session.user_id = "test_user"
        session.session_id = 1
        session._last_user_message = None
        session._last_message_time = datetime.now()
        session._profile_confirm_widget_sent = False
        session._notify_participants = MagicMock(return_value=None)

        with patch(
            "src.interview_session.interview_session.save_feedback_to_csv"
        ) as mock_save_feedback, patch(
            "src.interview_session.interview_session.SessionLogger.log_to_file",
            return_value=None,
        ), patch(
            "src.interview_session.interview_session.asyncio.create_task"
        ) as mock_create_task:
            session.trigger_profile_confirm_widget()

        assert session._profile_confirm_widget_sent is True
        assert len(session.chat_history) == 1
        assert session.chat_history[0].type == MessageType.PROFILE_CONFIRM_WIDGET
        mock_create_task.assert_called_once()
        mock_save_feedback.assert_not_called()
