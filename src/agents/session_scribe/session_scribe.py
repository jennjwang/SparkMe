from typing import List, TYPE_CHECKING, TypedDict, Optional
import asyncio
import re
import time
import os


from src.agents.base_agent import BaseAgent
from src.agents.session_scribe.prompts import get_prompt
from src.agents.session_scribe.tools import UpdateSessionNote, UpdateSubtopicNotes, UpdateSubtopicCoverage, FeedbackSubtopicCoverage, \
    UpdateMemoryBankAndSession, UpdateExistingMemory, AddHistoricalQuestion, IdentifyEmergentInsights, AddSnapshotSubtopic, \
    UpdateCriteriaCoverage, AddTaskDeepDiveTopic
from src.agents.shared.memory_tools import Recall
from src.agents.shared.note_tools import AddInterviewQuestion
from src.agents.shared.feedback_prompts import SIMILAR_QUESTIONS_WARNING, QUESTION_WARNING_OUTPUT_FORMAT
from src.content.question_bank.question import QuestionSearchResult, SimilarQuestionsGroup
from src.content.weekly_snapshot.snapshot_manager import SnapshotManager
from src.utils.data_process import read_from_pdf
from src.utils.llm.prompt_utils import format_prompt
from src.utils.llm.xml_formatter import extract_tool_arguments, extract_tool_calls_xml
from src.utils.logger.session_logger import SessionLogger
from src.utils.text_formatter import format_similar_questions
from src.interview_session.session_models import Participant, Message
from src.content.memory_bank.memory import Memory

if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession



class SessionScribeConfig(TypedDict, total=False):
    """Configuration for the SessionScribe agent."""
    user_id: str


class SessionScribe(BaseAgent, Participant):
    def __init__(self, config: SessionScribeConfig, interview_session: 'InterviewSession'):
        BaseAgent.__init__(
            self, name="SessionScribe",
            description="Agent that takes notes and manages the user's memory bank",
            config=config
        )
        Participant.__init__(self, title="SessionScribe",
                             interview_session=interview_session)
        
        # Current unprocessed memories
        self._new_memories: List[Memory] = []
        # All memories from this session
        self._all_session_memories: List[Memory] = []
        # Mapping from temporary memory IDs to real IDs
        self._memory_id_map = {}

        # Track last interviewer message
        self._last_interviewer_message = None
        # Set by _locked_update_subtopic_coverage when a widget submission is detected;
        # consumed by _get_formatted_prompt and _update_subtopic_coverage.
        self._pending_widget_subtopic_id: str | None = None
        self._pending_widget_task_names: list[str] | None = None

        # Locks and processing flags
        self.processing_in_progress = False # If processing is in progress
        self._pending_tasks = 0             # Track number of pending tasks
        # Coverage-specific progress flag: interviewer topic selection only
        # depends on this, not on memory-writing tail work.
        self.coverage_processing_in_progress = False
        self._coverage_pending_tasks = 0
        self._notes_lock = asyncio.Lock()   # Lock for _write_notes_and_questions
        self._session_agenda_lock = asyncio.Lock()  # Lock for session agenda
        self._snapshot_lock = asyncio.Lock()  # Lock for snapshot comparison findings
        self._tasks_lock = asyncio.Lock()   # Lock for updating task counter

        # Tools agent can use
        self.tools = {
            "update_memory_bank_and_session": UpdateMemoryBankAndSession(
                memory_bank=self.interview_session.memory_bank,
                on_memory_added=self._add_new_memory,
                update_memory_map=self._update_memory_map,
                get_current_qa=self._get_recent_qa,
                session_agenda=self.interview_session.session_agenda
            ),
            "update_existing_memory": UpdateExistingMemory(
                memory_bank=self.interview_session.memory_bank,
                on_memory_added=self._add_new_memory,
                get_current_qa=self._get_recent_qa,
                session_agenda=self.interview_session.session_agenda
            ),
            # "add_historical_question": AddHistoricalQuestion(
            #     question_bank=self.interview_session.historical_question_bank,
            #     memory_bank=self.interview_session.memory_bank,
            #     get_real_memory_ids=self._get_real_memory_ids
            # ),
            "update_session_agenda": UpdateSessionNote(
                session_agenda=self.interview_session.session_agenda
            ),
            "update_subtopic_coverage": UpdateSubtopicCoverage(
                session_agenda=self.interview_session.session_agenda
            ),
            "update_criteria_coverage": UpdateCriteriaCoverage(
                session_agenda=self.interview_session.session_agenda
            ),
            "feedback_subtopic_coverage": FeedbackSubtopicCoverage(
                session_agenda=self.interview_session.session_agenda
            ),
            "update_subtopic_notes": UpdateSubtopicNotes(
                session_agenda=self.interview_session.session_agenda
            ),
            "identify_emergent_insights": IdentifyEmergentInsights(
                session_agenda=self.interview_session.session_agenda,
                min_novelty_score=3
            ),
            "add_interview_question": AddInterviewQuestion(
                session_agenda=self.interview_session.session_agenda,
                historical_question_bank= \
                    self.interview_session.historical_question_bank,
                proposed_question_bank=self.interview_session.proposed_question_bank,
                proposer="SessionScribe",
                llm_engine=self.engine
            ),
            "recall": Recall(
                memory_bank=self.interview_session.memory_bank
            ),
            "add_snapshot_subtopic": AddSnapshotSubtopic(
                session_agenda=self.interview_session.session_agenda
            ),
            "add_task_deep_dive_topic": AddTaskDeepDiveTopic(
                session_agenda=self.interview_session.session_agenda
            ),
        }

    async def on_message(self, message: Message):
        '''Handle incoming messages'''
        SessionLogger.log_to_file(
            "execution_log",
            f"[NOTIFY] Session scribe received message from {message.role}"
        )

        if message.role == "Interviewer":
            self._last_interviewer_message = message
            # Add question to session agenda. The embedding HTTP call inside
            # this method is blocking, so run it as a background task instead
            # of awaiting it here — that keeps the session event loop free to
            # deliver the interviewer's message to the frontend immediately,
            # rather than waiting for the embedding round-trip to complete.
            asyncio.create_task(self._add_question_to_session_agenda_async())
        elif message.role == "User":
            if self._last_interviewer_message:
                interviewer_message = self._last_interviewer_message
                self._last_interviewer_message = None
                # Set the processing flag before scheduling the background
                # task so other subscribers (e.g., Interviewer) can observe
                # that scribe work is pending on this user turn.
                await self._increment_pending_tasks()
                await self._increment_pending_coverage_tasks()
                try:
                    asyncio.create_task(self._process_qa_pair(
                        interviewer_message=interviewer_message,
                        user_message=message
                    ))
                except Exception:
                    # If scheduling fails, release the pending-task counter.
                    await self._decrement_pending_coverage_tasks()
                    await self._decrement_pending_tasks()
                    raise
     
    async def augment_session_agenda(self, additional_context_path: Optional[str] = None):
        # If there is existing user profile, we load them
        if additional_context_path and os.path.exists(additional_context_path):
            if additional_context_path.endswith('.txt') or additional_context_path.endswith('.md'):
                with open(additional_context_path, 'r', encoding='utf-8') as f:
                    additional_context = f.read()
            elif additional_context_path.endswith('.pdf'):
                additional_context = read_from_pdf(additional_context_path)
            else:
                SessionLogger.log_to_file(
                    "execution_log", f"[INIT] Existing user profile is IGNORED, currently only supports .txt, .md, and .pdf files"
                )
            
            # Found initial context to be initialized with
            SessionLogger.log_to_file(
                "execution_log", f"[RUN] Found initial context to be initialized with, preparing an optimized session!"
            )

            # Get user portrait and last meeting summary
            await asyncio.gather(
                self.interview_session.session_scribe._update_user_portrait(
                    additional_context=additional_context
                ),
                self.interview_session.session_scribe._update_last_meeting_summary(
                    additional_context=additional_context
                ),
                self.interview_session.session_scribe._update_subtopic_notes(
                    additional_context=additional_context
                )
            )

            # TODO update memory?
            # Update session agenda notes and eventually coverage
            await self._update_list_of_subtopics(additional_context=additional_context)
            await self._update_subtopic_coverage()

        # For weekly check-ins: load the previous snapshot for turn-by-turn comparison
        if self.interview_session.session_type == "weekly":
            await self._load_last_week_snapshot()

    async def _load_last_week_snapshot(self):
        """Load the previous weekly snapshot onto the session agenda and
        inject each task as a subtopic so the interviewer covers them."""
        user_id = self.interview_session.user_id
        manager = SnapshotManager(user_id)
        prev_snapshot = manager.load_latest_snapshot()

        if prev_snapshot is None:
            SessionLogger.log_to_file(
                "execution_log",
                "[SNAPSHOT] No previous snapshot found — skipping."
            )
            return

        self.interview_session.session_agenda.last_week_snapshot = prev_snapshot.to_dict()

        injected_count = 0

        # Topic 1: Tasks and Deliverables
        for task in prev_snapshot.tasks:
            time_pct = f" (~{task.time_share:.0%})" if task.time_share else ""
            ai_str = f", used {task.ai_tool}" if task.ai_involved and task.ai_tool else ""
            description = (
                f"Last week: {task.description}{time_pct}{ai_str} "
                f"— is this still part of your week? What changed?"
            )
            if self.interview_session.session_agenda.add_emergent_subtopic(
                topic_id="1", subtopic_description=description
            ):
                injected_count += 1

        # Topic 2: Tools and Methods
        for tool in prev_snapshot.tools:
            description = f"Last week you used {tool} — are you still using it? Any changes in how you use it?"
            if self.interview_session.session_agenda.add_emergent_subtopic(
                topic_id="2", subtopic_description=description
            ):
                injected_count += 1

        for ai_tool in prev_snapshot.ai_tools:
            description = f"Last week you used AI tool {ai_tool} — still using it? How has your trust or usage changed?"
            if self.interview_session.session_agenda.add_emergent_subtopic(
                topic_id="2", subtopic_description=description
            ):
                injected_count += 1

        # Topic 3: Collaboration
        for collaborator in prev_snapshot.collaborators:
            description = f"Last week you worked with {collaborator} — any updates on that collaboration?"
            if self.interview_session.session_agenda.add_emergent_subtopic(
                topic_id="3", subtopic_description=description
            ):
                injected_count += 1

        # Topic 4: Pain Points and Bright Spots
        for pain_point in prev_snapshot.pain_points:
            description = f"Last week you mentioned this frustration: {pain_point} — has it improved or is it still an issue?"
            if self.interview_session.session_agenda.add_emergent_subtopic(
                topic_id="4", subtopic_description=description
            ):
                injected_count += 1

        for change in prev_snapshot.notable_changes:
            description = f"Last week you noted: {change} — any follow-up on that?"
            if self.interview_session.session_agenda.add_emergent_subtopic(
                topic_id="4", subtopic_description=description
            ):
                injected_count += 1

        SessionLogger.log_to_file(
            "execution_log",
            f"[SNAPSHOT] Loaded week {prev_snapshot.week_number} snapshot onto session agenda "
            f"({injected_count} subtopics injected across all topics)."
        )

    def _add_question_to_session_agenda(self):
        """Synchronous variant. Kept for non-async callers and tests; calls a
        blocking embedding HTTP request. Async callers (e.g. `on_message`)
        should use `_add_question_to_session_agenda_async` instead."""
        if self._last_interviewer_message:
            subtopic_id = str(self._last_interviewer_message.metadata.get('subtopic_id', ""))
            question_text = self._last_interviewer_message.content.strip()
            rubric = self._last_interviewer_message.metadata.get('rubric', None)

            adding_status = False
            if self.interview_session.proposed_question_bank:
                question = self.interview_session.proposed_question_bank.add_question(content=question_text,
                                                         subtopic_id=subtopic_id,
                                                         rubric=rubric)

                adding_status = self.interview_session.session_agenda.add_interview_question(question=question)
            else:
                adding_status = self.interview_session.session_agenda.add_interview_question_raw(
                    subtopic_id=subtopic_id,
                    question=question_text,
                    rubric=rubric
                )

            if not adding_status:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[NOTIFY] SessionAgenda failed/skipped to add question to session agenda.",
                )

    async def _add_question_to_session_agenda_async(self):
        """Async variant that offloads the blocking embedding API call to a
        worker thread so the session event loop stays responsive while the
        question is registered."""
        if not self._last_interviewer_message:
            return

        subtopic_id = str(self._last_interviewer_message.metadata.get('subtopic_id', ""))
        question_text = self._last_interviewer_message.content.strip()
        rubric = self._last_interviewer_message.metadata.get('rubric', None)

        try:
            adding_status = False
            if self.interview_session.proposed_question_bank:
                bank = self.interview_session.proposed_question_bank
                add_async = getattr(bank, 'add_question_async', None)
                if add_async is not None:
                    question = await add_async(
                        content=question_text,
                        subtopic_id=subtopic_id,
                        rubric=rubric,
                    )
                else:
                    # Fallback for question-bank implementations that don't
                    # expose an async API: run the blocking add in a thread.
                    question = await asyncio.to_thread(
                        bank.add_question,
                        content=question_text,
                        subtopic_id=subtopic_id,
                        rubric=rubric,
                    )

                adding_status = self.interview_session.session_agenda.add_interview_question(
                    question=question
                )
            else:
                adding_status = self.interview_session.session_agenda.add_interview_question_raw(
                    subtopic_id=subtopic_id,
                    question=question_text,
                    rubric=rubric,
                )

            if not adding_status:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[NOTIFY] SessionAgenda failed/skipped to add question to session agenda.",
                )
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[NOTIFY] Error adding question to session agenda: {e}",
                log_level="error",
            )

    async def _process_qa_pair(self, interviewer_message: Message, user_message: Message):
        """Process a Q&A pair with task tracking"""
        try:
            # 1. Update notes and questions given the repsonse to ALL Subtopic
            # 2. Update to memory bank as well, but this has already been done since it's only tied to question id
            # 3. Update session agenda
            # 4.  a. See if current subtopic is well done or not (LLM-as-a-judge); if it does, mark as covered and update the summary of the subtopic (TODO)
            #     b. Brainstorm possible emergent insights from the recent response.
            # await asyncio.gather(
            #     self._locked_write_memory_notes_and_question_bank(
            #         interviewer_message, user_message), # Technically this should have lock for session agenda but ... we'll see
            #     # self._locked_identify_emergent_insights(interviewer_message, user_message) # Quick analysis for emergent insights
            # )
            # Run memory/notes and coverage in parallel — they use separate locks
            # (_notes_lock vs _session_agenda_lock) and write independent state
            parallel_tasks = [
                self._locked_write_memory_notes_and_question_bank(interviewer_message, user_message),
                self._locked_update_subtopic_coverage(interviewer_message, user_message),
            ]
            if (getattr(self.interview_session, "session_type", "intake") == "weekly"
                    and self.interview_session.session_agenda.last_week_snapshot):
                parallel_tasks.append(
                    self._locked_compare_against_snapshot(interviewer_message, user_message)
                )
            await asyncio.gather(*parallel_tasks)

        finally:
            await self._decrement_pending_tasks()

        # Fire portrait update after releasing processing_in_progress so it
        # doesn't block the Interviewer from asking the next question.
        # Screening runs as a separate follow-on task so it never competes
        # with the interviewer's LLM call.
        if getattr(self.interview_session, "session_type", "intake") == "intake":
            async def _portrait_task():
                # Skip portrait re-extraction when no new memories were added since
                # the last update — the LLM would produce a near-identical portrait
                # (modulo phrasing noise).
                current_count = len(self.interview_session.memory_bank.memories)
                if current_count <= self.interview_session._portrait_update_memory_count:
                    return
                try:
                    await self.interview_session._generate_and_save_user_portrait()
                except Exception as e:
                    SessionLogger.log_to_file(
                        "execution_log", f"[PORTRAIT] Error updating user portrait: {e}"
                    )

            asyncio.create_task(_portrait_task())

    async def _locked_write_memory_notes_and_question_bank(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle update_memory_bank_and_session with lock"""
        async with self._notes_lock:
            self.add_event(sender=interviewer_message.role,
                        tag="memory_lock_message", 
                        content=interviewer_message.content)
            self.add_event(sender=user_message.role,
                        tag="memory_lock_message", 
                        content=user_message.content)
            await self._write_memory_notes_and_question_bank()
            
    async def _locked_update_subtopic_coverage(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle update_subtopic_coverage with lock"""
        try:
            async with self._session_agenda_lock:
                self.add_event(sender=interviewer_message.role,
                            tag="agenda_lock_message",
                            content=interviewer_message.content)
                self.add_event(sender=user_message.role,
                            tag="agenda_lock_message",
                            content=user_message.content)

                # Detect widget submission: interviewer set a pending subtopic_id when the
                # time-split widget was shown. Consume the flag here and extract task names
                # from the user's formatted allocation string for downstream validation.
                widget_subtopic_id = getattr(
                    self.interview_session, '_widget_pending_subtopic_id', None
                )
                if widget_subtopic_id:
                    self.interview_session._widget_pending_subtopic_id = None
                    self._pending_widget_subtopic_id = widget_subtopic_id
                    # Extract leaf task names: "Task Name: XX%" or "Parent › Child: XX%"
                    # Take only the part after the last "›" to get the leaf name.
                    raw_names = re.findall(
                        r'([^,\n]+?):\s*\d+(?:\.\d+)?%',
                        user_message.content
                    )
                    self._pending_widget_task_names = [
                        name.split('›')[-1].strip()
                        for name in raw_names
                        if name.strip()
                    ]
                else:
                    self._pending_widget_subtopic_id = None
                    self._pending_widget_task_names = None

                await self._update_subtopic_coverage()
        finally:
            await self._decrement_pending_coverage_tasks()
            
    async def _locked_update_list_of_subtopics(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle update_subtopic_coverage with lock"""
        async with self._session_agenda_lock:
            self.add_event(sender=interviewer_message.role,
                        tag="agenda_lock_message", 
                        content=interviewer_message.content)
            self.add_event(sender=user_message.role,
                        tag="agenda_lock_message", 
                        content=user_message.content)
            await self._update_list_of_subtopics()

    async def _locked_identify_emergent_insights(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle emergent insight identification with lock"""
        async with self._session_agenda_lock:
            self.add_event(sender=interviewer_message.role,
                        tag="agenda_lock_message",
                        content=interviewer_message.content)
            self.add_event(sender=user_message.role,
                        tag="agenda_lock_message",
                        content=user_message.content)
            await self._identify_emergent_insights()

    async def _locked_compare_against_snapshot(self, interviewer_message: Message, user_message: Message) -> None:
        """Wrapper to handle snapshot comparison with its own lock (separate from session agenda)"""
        async with self._snapshot_lock:
            self.add_event(sender=interviewer_message.role,
                        tag="snapshot_lock_message",
                        content=interviewer_message.content)
            self.add_event(sender=user_message.role,
                        tag="snapshot_lock_message",
                        content=user_message.content)
            await self._compare_against_snapshot()

    async def _compare_against_snapshot(self) -> None:
        """Compare the latest user response against last week's snapshot.
        Uses an LLM call to detect inconsistencies, confirmations, and unmentioned items."""
        prompt = self._get_formatted_prompt("compare_against_snapshot")
        self.add_event(
            sender=self.name,
            tag="compare_against_snapshot_prompt",
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name,
            tag="compare_against_snapshot_response",
            content=response
        )

        # Handle tool calls (LLM decides whether to call add_snapshot_finding or not)
        self.handle_tool_calls(response)

    async def _identify_emergent_insights(self) -> None:
        """
        Identify emergent insights from the recent Q&A pair.

        Quick analysis to detect counter-intuitive findings that contradict
        conventional wisdom or reveal unexpected patterns.
        """
        prompt = self._get_formatted_prompt("identify_emergent_insights")
        self.add_event(
            sender=self.name,
            tag="identify_emergent_insights_prompt",
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name,
            tag="identify_emergent_insights_response",
            content=response
        )

        # Handle tool calls (LLM decides whether to call tool or not)
        self.handle_tool_calls(response)

    async def _write_notes_and_questions(self) -> None:
        """
        Process user's response by updating session agenda 
        and considering follow-up questions.
        """
        if self.use_baseline:
            return
        
        # First update the direct response in session agenda
        await self._update_session_agenda()

        # Then consider and propose follow-up questions if appropriate
        # await self._propose_followups()
        
    async def _update_last_meeting_summary(self, additional_context: str):
        prompt = self._get_formatted_prompt("update_last_meeting_summary",
                                            additional_context=additional_context)
        self.add_event(
            sender=self.name, 
            tag="update_last_meeting_summary_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_last_meeting_summary_response", 
            content=response
        )
        
        async with self._session_agenda_lock:
            self.interview_session.session_agenda.update_last_meeting_summary_str(response)
    
    async def _update_user_portrait(self, additional_context: str):
        prompt = self._get_formatted_prompt("update_user_portrait",
                                            additional_context=additional_context)
        self.add_event(
            sender=self.name,
            tag="update_user_portrait_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_user_portrait_response", 
            content=response
        )
        
        async with self._session_agenda_lock:
            self.interview_session.session_agenda.update_user_portrait_str(response)
            
    async def _update_subtopic_notes(self, additional_context: str) -> None:
        """Process the latest conversation and update both memory and question banks."""
        prompt = self._get_formatted_prompt("update_subtopic_notes", additional_context=additional_context)
        self.add_event(
            sender=self.name, 
            tag="update_subtopic_notes_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_subtopic_notes_response", 
            content=response
        )
        
        async with self._session_agenda_lock:
            self.handle_tool_calls(response)

    async def _propose_followups(self) -> None:
        """
        Determine if follow-up questions should be proposed 
        and propose them if appropriate.
        """
        iterations = 0
        previous_tool_call = None
        similar_questions: List[SimilarQuestionsGroup] = []
        
        while iterations < self._max_consideration_iterations:
            prompt = self._get_formatted_prompt(
                "consider_and_propose_followups",
                previous_tool_call=previous_tool_call,
                similar_questions=similar_questions
            )
            
            self.add_event(
                sender=self.name,
                tag=f"consider_and_propose_followups_prompt_{iterations}",
                content=prompt
            )

            response = await self.call_engine_async(prompt)
            self.add_event(
                sender=self.name,
                tag=f"consider_and_propose_followups_response_{iterations}",
                content=response
            )

            # Check if agent wants to proceed with similar questions
            if "<proceed>true</proceed>" in response.lower():
                self.add_event(
                    sender=self.name,
                    tag=f"feedback_loop_{iterations}",
                    content="Agent chose to proceed with similar questions"
                )
                # Handle the tool calls to add questions
                await self.handle_tool_calls_async(response)
                break
            
            try:
                # Extract proposed questions from add_interview_question tool calls
                proposed_questions = extract_tool_arguments(
                    response, "add_interview_question", "question"
                )
            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[ERROR] Error extracting tool arguments: {e}"
                    f"Set proposed questions to empty list"
                )
                proposed_questions = []
            
            if not proposed_questions:
                if "recall" in response:
                    # Handle recall response
                    result = await self.handle_tool_calls_async(response)
                    self.add_event(
                        sender=self.name, 
                        tag="recall_response", 
                        content=result
                    )
                else:
                    # No questions proposed and no recall needed
                    # TODO
                    self._claim_core_topic_completion()
                    break
            else:
                # Search for similar questions. Each `search_questions` call
                # makes a blocking embedding API request; offload to a worker
                # thread (via the async variant when available) so the session
                # event loop stays responsive while we look up similarity.
                similar_questions: List[SimilarQuestionsGroup] = []
                for question in proposed_questions:
                    historical_bank = self.interview_session.historical_question_bank
                    proposed_bank = self.interview_session.proposed_question_bank
                    historical_search = getattr(
                        historical_bank, 'search_questions_async', None
                    )
                    proposed_search = getattr(
                        proposed_bank, 'search_questions_async', None
                    )
                    if historical_search is not None:
                        historical_results = await historical_search(
                            query=question, k=3
                        )
                    else:
                        historical_results = await asyncio.to_thread(
                            historical_bank.search_questions, query=question, k=3
                        )
                    if proposed_search is not None:
                        proposed_results = await proposed_search(
                            query=question, k=3
                        )
                    else:
                        proposed_results = await asyncio.to_thread(
                            proposed_bank.search_questions, query=question, k=3
                        )
                    
                    # Combine results and remove duplicates
                    all_results: List[QuestionSearchResult] = []
                    seen_questions = set()
                    
                    # Process all results and keep track of seen questions
                    for result_list in [historical_results, proposed_results]:
                        if result_list:
                            for result in result_list:
                                # Only add if we haven't seen before
                                if result.content not in seen_questions:
                                    all_results.append(result)
                                    seen_questions.add(result.content)
                    
                    # Sort by similarity score (higher score = more similar)
                    all_results.sort(key=lambda x: x.similarity_score, reverse=True)
                    
                    # Take top 3 unique results if available
                    top_results = all_results[:3] if all_results else []
                    
                    if top_results:
                        similar_questions.append(SimilarQuestionsGroup(
                            proposed=question,
                            similar=top_results
                        ))
                
                if not similar_questions:
                    # No similar questions found, proceed with adding
                    await self.handle_tool_calls_async(response)
                    break
                else:
                    # Save tool calls for next iteration
                    previous_tool_call = extract_tool_calls_xml(response)
            
            iterations += 1

        if iterations >= self._max_consideration_iterations:
            self.add_event(
                sender="system",
                tag="error",
                content=(
                    f"Exceeded maximum number of consideration iterations "
                    f"({self._max_consideration_iterations})"
                )
            )

    async def _write_memory_notes_and_question_bank(self) -> None:
        """Process the latest conversation and update both memory and question banks."""
        prompt = self._get_formatted_prompt("update_memory_and_session")
        self.add_event(
            sender=self.name, 
            tag="update_memory_question_bank_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_memory_question_bank_response", 
            content=response
        )
        self.handle_tool_calls(response)
        
    async def _update_subtopic_coverage(self, active_topics_only: bool = False) -> None:
        """Process the latest conversation and update subtopic coverage and possibly move to the next topic."""
        prompt = self._get_formatted_prompt("update_subtopic_coverage", active_topics_only=active_topics_only)
        self.add_event(
            sender=self.name,
            tag="update_subtopic_coverage_prompt",
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name,
            tag="update_subtopic_coverage_response",
            content=response
        )
        self.handle_tool_calls(response)

        # Guarantee the time-allocation subtopic is marked covered after a widget
        # submission, even if the LLM failed to call update_subtopic_coverage.
        if self._pending_widget_subtopic_id:
            agenda = self.interview_session.session_agenda
            subtopic_id = str(self._pending_widget_subtopic_id)
            self._pending_widget_subtopic_id = None
            self._pending_widget_task_names = None
            topic_id = subtopic_id.split(".")[0]
            core_topic = agenda.interview_topic_manager.get_core_topic(topic_id)
            subtopic = core_topic.get_subtopic(subtopic_id) if core_topic else None
            if subtopic is not None and not subtopic.is_covered:
                agenda.update_subtopic_coverage(
                    subtopic_id=subtopic_id,
                    aggregated_notes="Time allocation captured via widget submission."
                )

        # Decide if need to proceed
        self.interview_session.session_agenda.revise_agenda_after_update()
        
    async def _update_list_of_subtopics(self, active_topics_only: bool = False, additional_context: Optional[str] = None) -> None:
        """Process the latest conversation and update list of subtopics."""
        prompt = self._get_formatted_prompt("update_list_of_subtopics", active_topics_only=active_topics_only,
                                            additional_context=additional_context)
        self.add_event(
            sender=self.name, 
            tag="update_list_of_subtopics_prompt", 
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name, 
            tag="update_list_of_subtopics_response", 
            content=response
        )
        self.handle_tool_calls(response)

    async def _update_session_agenda(self) -> None:
        """Update session agenda with user's response"""
        prompt = self._get_formatted_prompt("update_session_agenda")
        self.add_event(
            sender=self.name,
            tag="update_session_agenda_prompt",
            content=prompt
        )
        response = await self.call_engine_async(prompt)
        self.add_event(
            sender=self.name,
            tag="update_session_agenda_response",
            content=response
        )
        self.handle_tool_calls(response)

    def _get_formatted_prompt(self, prompt_type: str, **kwargs) -> str:
        '''Gets the formatted prompt for the SessionScribe agent.'''
        prompt = get_prompt(prompt_type)
        if prompt_type == "consider_and_propose_followups":
            # Get all message events
            events = self.get_event_stream_str(filter=[
                {"tag": "notes_lock_message"},
                {"sender": self.name, "tag": "recall_response"},
                *[{"tag": f"consider_and_propose_followups_response_{i}"} \
                   for i in range(self._max_consideration_iterations)]
            ], as_list=True)

            recent_events = events[-self._max_events_len:] if len(
                events) > self._max_events_len else events

            # Format warning if needed
            similar_questions = kwargs.get('similar_questions', [])
            previous_tool_call = kwargs.get('previous_tool_call')
            warning = (
                SIMILAR_QUESTIONS_WARNING.format(
                    previous_tool_call=previous_tool_call,
                    similar_questions= \
                        format_similar_questions(similar_questions)
                ) if similar_questions and previous_tool_call 
                else ""
            )

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda \
                    .get_user_portrait_str(),
                "event_stream": "\n".join(recent_events),
                "questions_and_notes": (
                    self.interview_session.session_agenda \
                        .get_questions_and_notes_str()
                ),
                "similar_questions_warning": warning,
                "warning_output_format": QUESTION_WARNING_OUTPUT_FORMAT \
                                         if similar_questions else "",
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["recall", "add_interview_question"]
                )
            })
        elif prompt_type == "update_memory_and_session":
            events = self.get_event_stream_str(filter=[
                {"tag": "memory_lock_message"},
            ], as_list=True)
            # Only pass the user's response — memories should only come from
            # what the user said, not the interviewer's question
            current_response = events[-1:] if events else []
            previous_events = events[:-1] if events else []

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            # Show only the top semantically similar existing memories to avoid
            # the LLM over-merging unrelated memories
            memory_bank = self.interview_session.memory_bank
            if memory_bank.memories:
                query = "\n".join(current_response)
                similar = memory_bank.search_memories(query, k=5)
                existing_memories_str = ""
                for result in similar:
                    existing_memories_str += (
                        f"<memory id=\"{result.id}\">\n"
                        f"  <title>{result.title}</title>\n"
                        f"  <text>{result.text}</text>\n"
                        f"</memory>\n"
                    )
            else:
                existing_memories_str = "(no existing memories yet)"

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_response),
                "topics_list": self.interview_session.session_agenda.get_all_topics_and_subtopics(),
                "existing_memories": existing_memories_str,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_memory_bank_and_session", "update_existing_memory"]
                )
            })
        elif prompt_type == "update_list_of_subtopics":
            events = self.get_event_stream_str(filter=[
                {"tag": "agenda_lock_message"},
            ], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            active_topics_only = kwargs.get("active_topics_only", True)
            topics_list = self.interview_session.session_agenda.get_questions_and_notes_str(hide_answered="all",
                                                                                            active_topics_only=active_topics_only)

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "interview_description": self.interview_session.session_agenda.interview_description,
                "previous_events": "\n".join(previous_events),
                "additional_context": kwargs.get("additional_context", None),
                "current_qa": "\n".join(current_qa),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": topics_list,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_subtopic_coverage"]
                )
            })
        elif prompt_type == "update_session_agenda":
            events = self.get_event_stream_str(
                filter=[{"tag": "notes_lock_message"}], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_qa),
                "questions_and_notes": (
                    self.interview_session.session_agenda \
                        .get_questions_and_notes_str(
                            hide_answered="qa"
                        )
                ),
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_session_agenda"]
                )
            })
        elif prompt_type == "update_subtopic_coverage":
            events = self.get_event_stream_str(
                filter=[{"tag": "agenda_lock_message"}], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            active_topics_only = kwargs.get("active_topics_only", True)
            topics_list = self.interview_session.session_agenda.get_questions_and_notes_str(hide_answered="all",
                                                                                            active_topics_only=active_topics_only)

            task_deep_dive_enabled = (
                os.getenv("ENABLE_TASK_DEEP_DIVE", "false").lower() == "true"
                and getattr(self.interview_session, "session_type", "intake") != "weekly"
            )

            # Build widget task context when processing a widget submission.
            # _pending_widget_task_names is set by _locked_update_subtopic_coverage.
            widget_task_names = self._pending_widget_task_names
            widget_subtopic_id = self._pending_widget_subtopic_id
            if widget_task_names:
                names_list = "\n".join(f"  - {n}" for n in widget_task_names)
                widget_task_context = (
                    f"\n<widget_task_context>\n"
                    f"The participant just submitted the time-allocation widget. "
                    f"The following task names were included in the submission — "
                    f"check each one for clear action+object format (objective optional) "
                    f"and call `add_snapshot_subtopic` for any that are bare verbs, "
                    f"bare nouns, or missing a clear object:\n"
                    f"{names_list}\n"
                    f"\n⚠️ SCOPE RESTRICTION: In this evaluation round, ONLY assess "
                    f"subtopic {widget_subtopic_id} (time allocation). "
                    f"Do NOT mark any other subtopic as covered based on widget data — "
                    f"time percentages do NOT constitute answers to questions the "
                    f"interviewer has not yet asked (e.g. priority tasks).\n"
                    f"</widget_task_context>"
                )
                # Narrow the topics_list to just the time-allocation subtopic so the
                # LLM cannot see — and accidentally mark — other subtopics as covered.
                if widget_subtopic_id:
                    topic_id = widget_subtopic_id.split(".")[0]
                    agenda = self.interview_session.session_agenda
                    core_topic = agenda.interview_topic_manager.get_core_topic(topic_id)
                    st = core_topic.get_subtopic(widget_subtopic_id) if core_topic else None
                    if st is not None:
                        lines = [
                            "=== TOPIC ===",
                            f"Topic ID: {topic_id}",
                            f"Topic Description: {core_topic.description}",
                            "",
                            "    --- SUBTOPIC ---",
                            f"    Subtopic ID: {st.subtopic_id}",
                            f"    Subtopic Description: {st.description}",
                        ]
                        if st.coverage_criteria:
                            lines.append("    Coverage Criteria:")
                            for c in st.coverage_criteria:
                                lines.append(f"        - {c}")
                        lines.append(
                            f"    Subtopic Status: "
                            f"{'COVERED' if st.is_covered else 'NOT COVERED'}"
                        )
                        if not st.is_covered and st.notes:
                            lines.append(
                                f"    [Subtopic NOTES]: "
                                + "\n         -".join(st.notes)
                            )
                        topics_list = "\n".join(lines)
            else:
                widget_task_context = ""

            selected_tools = (
                ["update_criteria_coverage", "update_subtopic_coverage",
                 "add_task_deep_dive_topic"]
                if task_deep_dive_enabled
                else ["update_criteria_coverage", "update_subtopic_coverage"]
            )
            if widget_task_names:
                selected_tools = selected_tools + ["add_snapshot_subtopic"]

            formatted_prompt = format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_qa),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "widget_task_context": widget_task_context,
                "topics_list": topics_list,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=selected_tools
                )
            })
            if not task_deep_dive_enabled:
                formatted_prompt = re.sub(
                    r'[ \t]*<task_deep_dive_block>.*?</task_deep_dive_block>[ \t]*\n?',
                    '',
                    formatted_prompt,
                    flags=re.DOTALL,
                )
            else:
                formatted_prompt = formatted_prompt.replace('<task_deep_dive_block>', '').replace('</task_deep_dive_block>', '')
            return formatted_prompt
        elif prompt_type == "update_subtopic_notes":
            return format_prompt(prompt, {
                "additional_context": kwargs.get("additional_context"),
                "topics_list": (
                    self.interview_session.session_agenda.get_all_topics_and_subtopics(active_topics_only=False)
                ),
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["update_subtopic_notes"] # Note disable feedback as tools
                )
            })
        elif prompt_type == "update_last_meeting_summary":
            return format_prompt(prompt, {
                "additional_context": kwargs.get("additional_context"),
                "interview_description": self.interview_session.session_agenda.interview_description,
            })
        elif prompt_type == "update_user_portrait":
            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "additional_context": kwargs.get("additional_context"),
                "interview_description": self.interview_session.session_agenda.interview_description,
            })
        elif prompt_type == "compare_against_snapshot":
            events = self.get_event_stream_str(
                filter=[{"tag": "snapshot_lock_message"}], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []

            snapshot_str = self.interview_session.session_agenda.get_last_week_snapshot_str()
            coverage_str = self.interview_session.session_agenda.get_questions_and_notes_str(
                hide_answered="all", active_topics_only=True
            )

            return format_prompt(prompt, {
                "current_qa": "\n".join(current_qa),
                "last_week_snapshot": snapshot_str,
                "topic_coverage": coverage_str,
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["add_snapshot_subtopic", "recall"]
                )
            })
        elif prompt_type == "identify_emergent_insights":
            events = self.get_event_stream_str(
                filter=[{"tag": "agenda_lock_message"}], as_list=True)
            current_qa = events[-2:] if len(events) >= 2 else []
            previous_events = events[:-2] if len(events) >= 2 else events

            if len(previous_events) > self._max_events_len:
                previous_events = previous_events[-self._max_events_len:]

            return format_prompt(prompt, {
                "user_portrait": self.interview_session.session_agenda.user_portrait,
                "interview_description": self.interview_session.session_agenda.interview_description,
                "previous_events": "\n".join(previous_events),
                "current_qa": "\n".join(current_qa),
                "last_meeting_summary": self.interview_session.session_agenda.get_last_meeting_summary_str(),
                "topics_list": self.interview_session.session_agenda.get_all_topics_and_subtopics(active_topics_only=True),
                "tool_descriptions": self.get_tools_description(
                    selected_tools=["identify_emergent_insights"]
                )
            })

    async def get_session_memories(self, clear_processed=False, wait_for_processing=True, include_processed=False) -> List[Memory]:
        """Get memories added by session scribe during current session.
        
        Args:
            clear_processed: 
                - If True, clears the list of unprocessed memories after returning
            wait_for_processing: 
                - If True, waits for all pending memory updates to complete
            include_processed: 
                - If True, returns all memories from the session
                - If False, returns only the currently unprocessed memories
        
        Returns:
            List of Memory objects based on the include_processed parameter
        """
        if wait_for_processing:
            start_time = time.time()

            SessionLogger.log_to_file(
                "execution_log",
                f"[MEMORY] Waiting for memory updates to complete..."
            )
            
            while self.processing_in_progress:
                await asyncio.sleep(0.1)
                if time.time() - start_time > 300:  # 5 minutes timeout
                    SessionLogger.log_to_file(
                        "execution_log",
                        f"[MEMORY] Timeout waiting for memory updates"
                    )
                    break
        elif self.processing_in_progress:
            SessionLogger.log_to_file(
                "execution_log",
                f"[MEMORY] Retrieving memories..."
            )

        if include_processed:
            memories = self._all_session_memories.copy()
            memory_source = "all session"
        else:
            memories = self._new_memories.copy()
            memory_source = "unprocessed"
        
        if clear_processed:
            SessionLogger.log_to_file(
                "execution_log",
                f"[MEMORY] Clearing {len(self._new_memories)} unprocessed memories"
            )
            self._new_memories = []
            
        SessionLogger.log_to_file(
            "execution_log",
            (
                f"[MEMORY] Collected {len(memories)} {memory_source} memories "
                f"from current session"
            )
        )
        return memories

    def _add_new_memory(self, memory: Memory):
        """Callback to track newly added memory in the session"""
        self._new_memories.append(memory)
        self._all_session_memories.append(memory)  # Also add to all memories list

    def _update_memory_map(self, temp_id: str, real_id: str) -> None:
        """Callback to update the memory ID mapping"""
        self._memory_id_map[temp_id] = real_id
        SessionLogger.log_to_file("execution_log",
                                  f"[MEMORY] Write a new memory with {real_id}")

    def _get_real_memory_ids(self, temp_ids: List[str]) -> List[str]:
        """Callback to get real memory IDs from temporary IDs"""
        real_ids = [
            self._memory_id_map[temp_id]
            for temp_id in temp_ids
            if temp_id in self._memory_id_map
        ]
        return real_ids

    async def _increment_pending_tasks(self):
        """Increment the pending tasks counter"""
        async with self._tasks_lock:
            self._pending_tasks += 1
            self.processing_in_progress = True

    async def _increment_pending_coverage_tasks(self):
        """Increment the pending coverage-task counter."""
        async with self._tasks_lock:
            self._coverage_pending_tasks += 1
            self.coverage_processing_in_progress = True

    async def _decrement_pending_tasks(self):
        """Decrement the pending tasks counter"""
        async with self._tasks_lock:
            self._pending_tasks -= 1
            if self._pending_tasks <= 0:
                self._pending_tasks = 0
                self.processing_in_progress = False

    async def _decrement_pending_coverage_tasks(self):
        """Decrement the pending coverage-task counter."""
        async with self._tasks_lock:
            self._coverage_pending_tasks -= 1
            if self._coverage_pending_tasks <= 0:
                self._coverage_pending_tasks = 0
                self.coverage_processing_in_progress = False

    def _get_recent_qa(self) -> str:
        """Safely get the current user response, with error handling."""
        interviewer_question = "No interviewer question available"
        user_response = "No user response available"
        
        try:
            messages = self.get_event_stream_str(filter=[
                {"tag": "memory_lock_message", "sender": "Interviewer"}
            ], as_list=True)
                        
            if messages:
                last_message = messages[-1]
                interviewer_question = last_message. \
                    removeprefix("<Interviewer>\n"). \
                    removesuffix("\n</Interviewer>")
        except Exception as e:
            return "Error retrieving interviewer's question"
        
        try:
            messages = self.get_event_stream_str(filter=[
                {"tag": "memory_lock_message", "sender": "User"}
            ], as_list=True)
                        
            if messages:
                last_message = messages[-1]
                user_response = last_message. \
                    removeprefix("<User>\n"). \
                    removesuffix("\n</User>")
        except Exception as e:
            return "Error retrieving user response"
        
        return interviewer_question, user_response
