import asyncio
import os
import re
from typing import TYPE_CHECKING, TypedDict, Tuple

import json

from src.agents.base_agent import BaseAgent
from src.agents.interviewer.prompts import get_prompt
from src.agents.interviewer.tools import EndConversation, RespondToUser
from src.agents.shared.memory_tools import Recall
from src.utils.llm.prompt_utils import format_prompt
from src.interview_session.session_models import Participant, Message, MessageType
from src.utils.llm.xml_formatter import parse_rubric_call
from src.utils.logger.session_logger import SessionLogger
from src.utils.constants.colors import GREEN, RESET
from src.content.question_bank.question import Rubric

if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession



class TTSConfig(TypedDict, total=False):
    """Configuration for text-to-speech."""
    enabled: bool
    provider: str  # e.g. 'openai'
    voice: str     # e.g. 'alloy'


class InterviewerConfig(TypedDict, total=False):
    """Configuration for the Interviewer agent."""
    user_id: str
    tts: TTSConfig
    interview_description: str


class Interviewer(BaseAgent, Participant):
    '''Inherits from BaseAgent and Participant. Participant is a class that all agents in the interview session inherit from.'''

    def __init__(self, config: InterviewerConfig, interview_session: 'InterviewSession'):
        BaseAgent.__init__(
            self, name="Interviewer",
            description="The agent that holds the interview and asks questions.",
            config=config)
        Participant.__init__(
            self, title="Interviewer",
            interview_session=interview_session)

        self.interview_description = config.get("interview_description")
        self._turn_to_respond = False
        self._message_sent_this_turn = False
        self._time_split_widget_shown = False
        self._response_lock = asyncio.Lock()
        self._turn_start: float | None = None

        self.tools = {
            "recall": Recall(memory_bank=self.interview_session.memory_bank),
            "respond_to_user": RespondToUser(
                tts_config=config.get("tts", {}),
                base_path= \
                    f"{os.getenv('DATA_DIR', 'data')}/{config.get('user_id')}/",
                on_response=self._guarded_handle_response,
                on_turn_complete=lambda: setattr(
                    self, '_turn_to_respond', False)
            ),
            "end_conversation": EndConversation(
                on_goodbye=self._guarded_send_goodbye,
                on_end=lambda: (
                    setattr(self, '_turn_to_respond', False),
                    self.interview_session.trigger_feedback_widget()
                )
            )
        }

    def _handle_quantify_response(self, quantified_response: str,
                                  original_response: str) -> Tuple[str, Rubric]:
        # 2. Parse the <tool_calls> block from the response
        final_question_text = original_response
        final_rubric = None
        try:
            parsed_calls = parse_rubric_call(quantified_response)
            for parsed_call in parsed_calls:
                is_quantifiable = str(parsed_call.get('quantifiable', 'false')).lower() == 'true'

                if is_quantifiable:
                    final_question_text = parsed_call.get('question', original_response)
                    rubric_data_str = parsed_call.get('rubric')
                    if rubric_data_str and isinstance(rubric_data_str, str):
                        # The rubric might be a string representation of JSON
                        try:
                            rubric_data = json.loads(rubric_data_str)
                            final_rubric = Rubric(**rubric_data) # Need to be in string
                        except json.JSONDecodeError:
                            SessionLogger.log_to_file(
                                "execution_log",
                                f"Could not parse rubric JSON string: {rubric_data_str}",
                                log_level="warning"
                            )
                    elif isinstance(rubric_data_str, dict): # Sometimes it's already a dict
                        final_rubric = Rubric(**rubric_data_str) # Need to be in string

        except Exception as e:
            # If parsing fails for any reason, log it but fall back gracefully
            SessionLogger.log_to_file(
                "execution_log",
                f"Could not parse rubric response: {e}. Using original question.",
                log_level="warning"
            )
            final_question_text = original_response
            final_rubric = None

        return final_question_text, final_rubric

    async def _guarded_handle_response(self, response: str, subtopic_id: str = "") -> str:
        """Gate: only the first message per turn reaches the user."""
        if self._message_sent_this_turn:
            SessionLogger.log_to_file(
                "execution_log",
                f"(Interviewer) Blocked duplicate respond_to_user this turn: "
                f"{response[:150]}",
                log_level="warning"
            )
            return response
        self._message_sent_this_turn = True
        return await self._handle_response(response, subtopic_id)

    def _guarded_send_goodbye(self, goodbye: str) -> None:
        """Gate: only the first message per turn reaches the user (for end_conversation)."""
        if self._message_sent_this_turn:
            SessionLogger.log_to_file(
                "execution_log",
                f"(Interviewer) Blocked duplicate end_conversation this turn: "
                f"{goodbye[:150]}",
                log_level="warning"
            )
            return
        self._message_sent_this_turn = True
        # Mark session ending immediately so concurrent on_message tasks don't
        # start new turns during the 1-second sleep in EndConversation._run.
        self.interview_session._session_ending = True
        self.add_event(sender=self.name, tag="goodbye", content=goodbye)
        self.interview_session.add_message_to_chat_history(
            role=self.title, content=goodbye)

    async def _handle_response(self, response: str, subtopic_id: str = "") -> str:
        """Handle responses from the RespondToUser tool by quantifying it and adding them to chat history.
        
        Args:
            response: The response text to add to chat history
            topic_id: The topic ID of the response
            subtopic_id: The subtopic ID of the response
        """
        # # Quantify question even further
        # quantify_prompt = format_prompt(get_prompt("quantify_question"), {"question_text": response})
        # self.add_event(sender=self.name, tag="llm_prompt", content=quantify_prompt)
        # quantified_response = await self.call_engine_async(quantify_prompt)
        # print(f"{GREEN}Interviewer Quantified:\n{quantified_response}{RESET}")
        # quantified_question, rubric = self._handle_quantify_response(quantified_response=quantified_response,
        #                                                              original_response=response)
        
        # If we disable quantification
        quantified_question = response
        rubric = None

        # Don't send a question if the session is already ending gracefully
        if getattr(self.interview_session, '_session_ending', False):
            self._turn_to_respond = False
            return quantified_question

        turn_start = getattr(self, '_turn_start', None)
        if turn_start is not None:
            total_wait_s = asyncio.get_event_loop().time() - turn_start
            SessionLogger.log_to_file(
                "execution_log",
                f"[LATENCY] total_wait={total_wait_s:.2f}s"
                f" response_preview={quantified_question[:60].replace(chr(10), ' ')!r}"
            )

        self.interview_session.add_message_to_chat_history(
            role=self.title,
            content=quantified_question,
            metadata={'subtopic_id': str(subtopic_id), "rubric": rubric},
        )
        self.add_event(sender=self.name, tag="message",
                       content=quantified_question)

        if (subtopic_id
                and self._is_time_allocation_subtopic(subtopic_id)
                and not self._time_split_widget_shown):
            self._time_split_widget_shown = True
            # Signal to the scribe that the next user message is a widget
            # submission so it can auto-mark coverage and validate task names.
            self.interview_session._widget_pending_subtopic_id = str(subtopic_id)
            # Emit the widget message immediately so the loading placeholder
            # appears as soon as the question does. Portrait refresh runs async;
            # the frontend polls /api/session-state until tasks are ready.
            self.interview_session.add_message_to_chat_history(
                role=self.title,
                content="",
                message_type=MessageType.TIME_SPLIT_WIDGET,
            )
            asyncio.create_task(self._refresh_portrait_before_widget())

        return quantified_question

    async def _refresh_portrait_before_widget(self) -> None:
        """Refresh portrait with latest memories, then emit the time-split widget.
        Emitting AFTER portrait is ready means the frontend's /api/session-state
        call on render always sees a populated Task Inventory."""
        scribe = getattr(self.interview_session, 'session_scribe', None)
        _scribe_wait_start = asyncio.get_event_loop().time()
        if scribe is not None:
            start = asyncio.get_event_loop().time()
            while getattr(scribe, 'processing_in_progress', False):
                await asyncio.sleep(0.1)
                if asyncio.get_event_loop().time() - start > 30:
                    SessionLogger.log_to_file(
                        "execution_log",
                        "[WIDGET] Timed out waiting for scribe before portrait refresh"
                    )
                    break
        _scribe_wait_s = asyncio.get_event_loop().time() - _scribe_wait_start
        _portrait_start = asyncio.get_event_loop().time()
        try:
            # Acquire the lock (wait, don't skip) so we run AFTER any concurrent
            # portrait update finishes, then force a fresh extraction with all
            # current memories — the widget needs a populated Task Inventory.
            async with self.interview_session._portrait_update_lock:
                await self.interview_session._generate_and_save_user_portrait_inner()
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[WIDGET] Portrait refresh before time-split widget failed: {e}"
            )
        _portrait_s = asyncio.get_event_loop().time() - _portrait_start
        SessionLogger.log_to_file(
            "execution_log",
            f"[LATENCY] widget_portrait_refresh: scribe_wait={_scribe_wait_s:.2f}s"
            f" portrait_llm={_portrait_s:.2f}s"
        )
        # Pre-warm the task-tree cache so the widget's first /api/organize-tasks
        # request is served instantly from cache rather than triggering a fresh
        # GROUP-pass LLM call at render time.
        asyncio.create_task(self._prewarm_task_tree_cache())

    async def _prewarm_task_tree_cache(self) -> None:
        """Run organize-tasks server-side and populate the backend cache so the
        widget's first GET /api/organize-tasks is a cache hit, not a 7-second LLM call."""
        try:
            from src.utils.task_hierarchy import organize_tasks as _organize_tasks
            from src.main_flask import _TASK_TREE_CACHE, _TASK_TREE_CACHE_LOCK, _task_tree_signature, _TASK_TREE_CACHE_MAX
            portrait = self.interview_session.session_agenda.user_portrait
            tasks = portrait.get("Task Inventory", []) if isinstance(portrait, dict) else []
            if len(tasks) < 2:
                return
            sig = _task_tree_signature(tasks)
            if not sig:
                return
            with _TASK_TREE_CACHE_LOCK:
                if sig in _TASK_TREE_CACHE:
                    return  # already cached
            tree = await asyncio.to_thread(_organize_tasks, [str(t) for t in tasks], "gpt-4.1-mini", True)
            with _TASK_TREE_CACHE_LOCK:
                _TASK_TREE_CACHE[sig] = tree
                _TASK_TREE_CACHE.move_to_end(sig)
                while len(_TASK_TREE_CACHE) > _TASK_TREE_CACHE_MAX:
                    _TASK_TREE_CACHE.popitem(last=False)
            SessionLogger.log_to_file("execution_log", "[WIDGET] Task tree pre-warmed in cache")
        except Exception as e:
            SessionLogger.log_to_file("execution_log", f"[WIDGET] Task tree pre-warm failed: {e}")

    def _is_time_allocation_subtopic(self, subtopic_id: str) -> bool:
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return False
        for topic in agenda.interview_topic_manager:
            for st in topic.required_subtopics.values():
                if str(st.subtopic_id) == str(subtopic_id) and 'time allocation' in st.description.lower():
                    return True
        return False


    async def on_message(self, message: Message):

        if getattr(self.interview_session, '_session_ending', False):
            return

        if self._response_lock.locked():
            SessionLogger.log_to_file(
                "execution_log",
                "(Interviewer) Dropping duplicate on_message — turn already in progress",
                log_level="warning"
            )
            return

        async with self._response_lock:
            await self._on_message_body(message)

    async def _on_message_body(self, message: Message):

        self._message_sent_this_turn = False
        self._turn_start = asyncio.get_event_loop().time()

        if message:
            # Ignore notifications about the Interviewer's own messages.
            if message.role == self.title:
                return
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] Interviewer received message from {message.role}"
            )
            self.add_event(sender=message.role, tag="message",
                           content=message.content)
        # Opening turn of a fresh session with no prior summary: let LLM generate first question
        is_weekly = getattr(self.interview_session, "session_type", "intake") == "weekly"

        # If session is paused, wait until resumed or a step is requested
        await self.interview_session.wait_if_paused()

        # Speculative execution: start the LLM call immediately so it runs
        # concurrently with the scribe's processing.  The prompt may use
        # slightly stale coverage (previous turn), but this cuts latency from
        # (scribe_time + LLM_time) down to max(scribe_time, LLM_time).
        speculative_prompt: str | None = None
        speculative_task = None
        pre_scribe_coverage: frozenset | None = None
        _speculative_start = None
        if message is not None:
            pre_scribe_coverage = self._get_coverage_snapshot()
            speculative_prompt = self._get_prompt()
            _speculative_start = asyncio.get_event_loop().time()
            speculative_task = asyncio.create_task(
                self.call_engine_async(speculative_prompt)
            )

        # Wait for the scribe to finish processing the current turn so coverage is up to date.
        # Early-exit once coverage has been stable for 1s AND the speculative LLM call is done
        # — further scribe work (memory writes, portrait) doesn't affect topic selection.
        _scribe_wait_start = asyncio.get_event_loop().time()
        _scribe_waited = False
        if message is not None:
            scribe = getattr(self.interview_session, 'session_scribe', None)
            if scribe is not None:
                _last_seen_coverage = pre_scribe_coverage
                _coverage_stable_ticks = 0
                for _ in range(100):  # max ~10s wait
                    if not getattr(scribe, 'processing_in_progress', False):
                        break
                    _scribe_waited = True
                    await asyncio.sleep(0.1)
                    # Every 5 ticks (0.5s), check if coverage has changed.
                    # If stable for 2 consecutive checks (1s) and the speculative
                    # result is ready, we can proceed without waiting for the rest
                    # of the scribe's work (memory writes, portrait update).
                    if _ % 5 == 4 and speculative_task is not None:
                        curr = self._get_coverage_snapshot()
                        if curr == _last_seen_coverage:
                            _coverage_stable_ticks += 1
                            if _coverage_stable_ticks >= 2 and speculative_task.done():
                                SessionLogger.log_to_file(
                                    "execution_log",
                                    "[LATENCY] early-exit scribe wait: coverage stable 1s"
                                )
                                break
                        else:
                            _coverage_stable_ticks = 0
                            _last_seen_coverage = curr
        _scribe_wait_s = asyncio.get_event_loop().time() - _scribe_wait_start

        # If the scribe updated coverage, the speculative prompt is stale — discard it
        # and regenerate with the fresh state to avoid acting on stale coverage.
        _speculative_discarded = False
        if message is not None and speculative_task is not None:
            post_scribe_coverage = self._get_coverage_snapshot()
            if post_scribe_coverage != pre_scribe_coverage:
                speculative_task.cancel()
                speculative_task = None
                speculative_prompt = None
                _speculative_discarded = True

        if message is not None:
            SessionLogger.log_to_file(
                "execution_log",
                f"[LATENCY] scribe_wait={_scribe_wait_s:.2f}s"
                f" speculative_discarded={_speculative_discarded}"
            )

        self._turn_to_respond = True
        iterations = 0

        while (self._turn_to_respond
               and iterations < self._max_consideration_iterations
               and not getattr(self.interview_session, '_session_ending', False)):
            # First iteration: reuse the speculative LLM call started before
            # the scribe wait.  Subsequent iterations (e.g. recall tool)
            # always build a fresh prompt with up-to-date session state.
            if iterations == 0 and speculative_task is not None:
                prompt = speculative_prompt
                response = await speculative_task
                speculative_task = None  # consumed
            else:
                prompt = self._get_prompt()
                response = await self.call_engine_async(prompt)
            self.add_event(sender=self.name, tag="llm_prompt", content=prompt)
            print(f"{GREEN}Interviewer:\n{response}{RESET}")
            try:
                # gpt-5.x may return JSON instead of XML — convert it to the
                # expected XML tool-call format so the rest of the pipeline works.
                if "<tool_calls>" not in response:
                    parsed = None
                    raw = response.strip()
                    try:
                        parsed = json.loads(raw)
                    except (ValueError, Exception):
                        # LLM may return JSON with unescaped quotes inside string values.
                        # Fall back to regex extraction: subtopic_id is simple, response
                        # is everything between "response":" and the last " before "}"
                        sid_m = re.search(r'"subtopic_id"\s*:\s*"([^"]*)"', raw)
                        resp_m = re.search(r'"response"\s*:\s*"(.*)"', raw, re.DOTALL)
                        if sid_m and resp_m:
                            parsed = {
                                "subtopic_id": sid_m.group(1),
                                "response": resp_m.group(1).rstrip("}").rstrip(),
                            }
                    if isinstance(parsed, dict) and "response" in parsed:
                        subtopic_id = str(parsed.get("subtopic_id", ""))
                        resp_text = parsed["response"]
                        # Escape XML special chars in the values
                        for ch, esc in [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")]:
                            resp_text = resp_text.replace(ch, esc)
                            subtopic_id = subtopic_id.replace(ch, esc)
                        response = (
                            f"<tool_calls><respond_to_user>"
                            f"<subtopic_id>{subtopic_id}</subtopic_id>"
                            f"<response>{resp_text}</response>"
                            f"</respond_to_user></tool_calls>"
                        )
                if "<tool_calls>" not in response and any(
                    f"<{tool_name}>" in response for tool_name in self.tools
                ):
                    response = f"<tool_calls>{response}</tool_calls>"
                await self.handle_tool_calls_async(response)
            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"(Interviewer) Error handling tool call: {e}",
                    log_level="error"
                )
                print(f"Error calling tool: {e}. Attempting to extract response text.")
                # If the LLM was trying to end but the XML was malformed, trigger
                # the feedback widget directly rather than sending raw XML to the user.
                if 'end_conversation' in response and not self._message_sent_this_turn:
                    self._turn_to_respond = False
                    if not getattr(self.interview_session, '_feedback_widget_sent', False):
                        SessionLogger.log_to_file(
                            "execution_log",
                            "(Interviewer) Malformed end_conversation XML — triggering feedback widget directly.",
                            log_level="warning"
                        )
                        self.interview_session.trigger_feedback_widget()
                elif not self._message_sent_this_turn and response.strip():
                    # Extract human-readable text rather than sending raw XML/JSON
                    fallback = response
                    if "<response>" in response:
                        m = re.search(r'<response>(.*?)</response>', response, re.DOTALL)
                        if m:
                            fallback = m.group(1).strip()
                    elif response.strip().startswith('{'):
                        try:
                            parsed = json.loads(response.strip())
                            if isinstance(parsed, dict) and "response" in parsed:
                                fallback = parsed["response"]
                        except Exception:
                            pass
                    await self._guarded_handle_response(fallback)

            iterations += 1
            if iterations >= self._max_consideration_iterations:
                self.add_event(
                    sender="system",
                    tag="error",
                    content=f"Exceeded maximum number of consideration "
                    f"iterations ({self._max_consideration_iterations})"
                )

    def _get_coverage_snapshot(self) -> frozenset:
        """Return a frozenset of subtopic_ids that are currently marked as covered.

        Used to detect whether the scribe updated coverage while the speculative
        LLM call was in flight, so we can discard a stale speculative response.
        """
        covered = set()
        topic_manager = self.interview_session.session_agenda.interview_topic_manager
        for core_topic in topic_manager.core_topic_dict.values():
            for subtopic in core_topic:
                if subtopic.is_covered:
                    covered.add(subtopic.subtopic_id)
        return frozenset(covered)

    def _get_prompt(self):
        '''Gets the prompt for the interviewer. '''
        # Get user portrait and last meeting summary from session agenda
        user_portrait_str = self.interview_session.session_agenda \
            .get_user_portrait_str()
        last_meeting_summary_str = (
            self.interview_session.session_agenda
            .get_last_meeting_summary_str()
        )

        # Get chat history from event stream where these are the senders
        chat_history_events = self.get_event_stream_str(
            [
                {"sender": "Interviewer", "tag": "message"},
                {"sender": "User", "tag": "message"},
                {"sender": "system", "tag": "recall"},
            ],
            as_list=True
        )

        recent_events = chat_history_events[-self._max_events_len:] if \
            len(chat_history_events) > self._max_events_len else chat_history_events
        current_events = recent_events[-2:] if len(recent_events) >= 2 else recent_events

        all_interviewer_messages = self.get_event_stream_str(
            [{"sender": "Interviewer", "tag": "message"}],
            as_list=True
        )
        recent_interviewer_messages = all_interviewer_messages[-15:] if \
            len(all_interviewer_messages) >= 15 else all_interviewer_messages

        # Start with all available tools
        tools_set = set(self.tools.keys())
        
        # if self.interview_session.api_participant:
        #     # Don't end_conversation directly if API participant is present
        #     tools_set.discard("end_conversation")
        
        if self.use_baseline:
            # For baseline mode, remove recall tool
            tools_set.discard("recall")


        # For weekly sessions, provide the snapshot for prompt formatting
        last_week_snapshot_str = ""
        if getattr(self.interview_session, "session_type", "intake") == "weekly":
            last_week_snapshot_str = (
                self.interview_session.session_agenda
                .get_last_week_snapshot_str()
            )

        # Provide dynamic remaining time so the LLM can pace topic selection.
        # Time checks are handled by the session layer — the LLM must never
        # mention time or ask the user about extending.
        available_time = self.interview_session.session_agenda.available_time_minutes
        session_start = self.interview_session._session_start_time
        if available_time and available_time > 0 and session_start:
            from datetime import datetime
            elapsed_minutes = (datetime.now() - session_start).total_seconds() / 60.0
            remaining_minutes = max(0, available_time - elapsed_minutes)
            available_time_context = (
                f"Time remaining: approximately {remaining_minutes:.0f} minutes out of {available_time} total. "
                f"Use this to pace your topic selection: prioritize higher-weight topics first, "
                f"skip or abbreviate lower-priority ones if time is short."
            )
        elif available_time and available_time > 0:
            available_time_context = (
                f"The session has {available_time} minutes total. "
                f"Prioritize higher-weight topics first."
            )
        else:
            available_time_context = ""

        # Create format parameters based on prompt type
        format_params = {
            "user_portrait": user_portrait_str,
            "interview_description": self.interview_description,
            "available_time_context": available_time_context,
            "last_meeting_summary": last_meeting_summary_str,
            "last_week_snapshot": last_week_snapshot_str,
            "chat_history": '\n'.join(recent_events),
            "current_events": '\n'.join(current_events),
            "recent_interviewer_messages": '\n'.join(
                [msg for msg in recent_interviewer_messages]),
            "tool_descriptions": self.get_tools_description(list(tools_set))
        }

        # Only add questions_and_notes for normal mode
        if not self.use_baseline:
            questions_and_notes_str = self.interview_session.session_agenda \
                .get_questions_and_notes_str(
                    hide_answered="all", active_topics_only=True
                )
            format_params["questions_and_notes"] = questions_and_notes_str

            # Get strategic question suggestions from StrategicPlanner (only if not stale)
            # Staleness is checked before formatting to avoid unnecessary work
            if self._should_include_strategic_questions():
                strategic_questions_str = self._format_strategic_questions()
                format_params["strategic_questions"] = strategic_questions_str

        # Select prompt variant based on session type and conversation state
        is_weekly = getattr(self.interview_session, "session_type", "intake") == "weekly"

        if len(all_interviewer_messages) == 0 and len(last_meeting_summary_str) == 0:
            main_prompt = get_prompt("weekly_introduction" if is_weekly else "introduction")
        elif len(all_interviewer_messages) == 0:
            main_prompt = get_prompt("weekly_introduction" if is_weekly else "introduction_continue_session")
        elif self.use_baseline:
            main_prompt = get_prompt("baseline")
        elif is_weekly:
            main_prompt = get_prompt("weekly_normal")
        else:
            main_prompt = get_prompt("normal")

        # Remove strategic questions section if not included to avoid unfilled placeholders
        if not self.use_baseline and not self._should_include_strategic_questions():
            # Remove the entire <strategic_questions>...</strategic_questions> block
            main_prompt = re.sub(
                r'\s*<strategic_questions>.*?</strategic_questions>\s*',
                '\n',
                main_prompt,
                flags=re.DOTALL
            )
            main_prompt = main_prompt.replace("{strategic_questions}", "")

        # Strip the emergent-insight probing block when no active topic opts
        # into emergent exploration (e.g. structured intake where the agenda
        # is intentionally fixed). Otherwise the interviewer would free-
        # associate beyond the configured subtopics.
        topic_manager = self.interview_session.session_agenda.interview_topic_manager
        if not topic_manager.any_active_topic_allows_emergent():
            main_prompt = re.sub(
                r'[ \t]*<emergent_insights_block>.*?</emergent_insights_block>[ \t]*\n?',
                '',
                main_prompt,
                flags=re.DOTALL,
            )
        else:
            # Keep contents, just drop the marker tags.
            main_prompt = main_prompt.replace('<emergent_insights_block>', '')
            main_prompt = main_prompt.replace('</emergent_insights_block>', '')

        # Include strict mode banner only when ALL active topics disable both emergent
        # exploration and strategic planning — i.e. the interviewer must stay exactly
        # on the configured subtopics.
        strict_mode = (
            not topic_manager.any_active_topic_allows_emergent()
            and not topic_manager.any_active_topic_allows_strategic_planner()
        )
        if strict_mode:
            main_prompt = main_prompt.replace('<strict_mode_block>', '').replace('</strict_mode_block>', '')
        else:
            main_prompt = re.sub(
                r'[ \t]*<strict_mode_block>.*?</strict_mode_block>[ \t]*\n?',
                '',
                main_prompt,
                flags=re.DOTALL,
            )

        return format_prompt(main_prompt, format_params)

    def _format_strategic_questions(self) -> str:
        """
        Format strategic question suggestions from StrategicPlanner.

        Returns formatted string with strategic questions or empty state message.
        Handles case where suggestions may be stale (from 3-5 turns ago).
        """
        # Access strategic state from StrategicPlanner
        strategic_state = self.interview_session.strategic_planner.strategic_state
        suggestions = strategic_state.strategic_question_suggestions

        if not suggestions:
            return "No strategic question suggestions available yet. Use coverage-based heuristics to select questions from the topics list."

        # Get top rollout if available
        top_rollout = None
        if strategic_state.rollout_predictions:
            top_rollout = strategic_state.rollout_predictions[0]

        formatted_lines = []

        # Add top rollout context if available
        if top_rollout:
            formatted_lines.append("**Highest-Utility Conversation Path Predicted:**")
            formatted_lines.append(f"Utility Score: {top_rollout.utility_score:.3f} (Higher is better)")
            formatted_lines.append(f"- Expected new subtopics covered: {top_rollout.expected_coverage_delta}")
            formatted_lines.append(f"- Emergence potential: {top_rollout.emergence_potential:.2f}")
            formatted_lines.append(f"- Cost (turns): {top_rollout.cost_estimate}")
            formatted_lines.append("")
            formatted_lines.append("The questions below are optimized to align with this high-utility path.")
            formatted_lines.append("")

        # Format suggestions by priority (high to low)
        sorted_suggestions = sorted(suggestions, key=lambda x: x.get('priority', 0), reverse=True)

        formatted_lines.append("**Strategic Question Suggestions (sorted by priority):**")
        formatted_lines.append("")
        for i, suggestion in enumerate(sorted_suggestions, 1):
            formatted_lines.append(f"{i}. **{suggestion['content']}**")
            formatted_lines.append(f"   - Target: Subtopic {suggestion['subtopic_id']}")
            formatted_lines.append(f"   - Strategy: {suggestion['strategy_type']}")
            formatted_lines.append(f"   - Priority: {suggestion['priority']}/10")
            formatted_lines.append(f"   - Reasoning: {suggestion['reasoning']}")
            formatted_lines.append("")  # Blank line between suggestions

        return "\n".join(formatted_lines)

    def _should_include_strategic_questions(self) -> bool:
        """
        Determine if strategic questions should be included in the prompt.

        Strategic questions become stale after exceeding the rollout horizon.
        Only include them if they are fresh (within horizon + buffer).

        Returns:
            bool: True if strategic questions should be included, False if stale
        """
        strategic_state = self.interview_session.strategic_planner.strategic_state

        # If no suggestions exist, don't include
        if not strategic_state.strategic_question_suggestions:
            return False

        # Calculate current turn (count User messages)
        current_turn = len([
            m for m in self.interview_session.chat_history
            if m.role == "User"
        ])

        # Get last planning turn from strategic state
        last_planning_turn = strategic_state.last_planning_turn

        # If planning hasn't run yet (turn 0), don't include
        if last_planning_turn == 0:
            return False

        # Get rollout horizon from strategic planner
        rollout_horizon = self.interview_session.strategic_planner.rollout_horizon

        # Calculate staleness: questions are stale if we're beyond horizon + buffer
        # Buffer of 2 turns accounts for: 1) planning completes after trigger, 2) grace period
        staleness_threshold = last_planning_turn + rollout_horizon + 2

        # Include questions only if NOT stale
        is_fresh = current_turn <= staleness_threshold

        if not is_fresh:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] (Interviewer) Strategic questions are stale "
                f"(last_planning_turn={last_planning_turn}, current_turn={current_turn}, "
                f"threshold={staleness_threshold}). Excluding from prompt."
            )

        return is_fresh
