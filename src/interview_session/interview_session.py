import asyncio
import json
import os
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, TypedDict
import signal
import contextlib

import time
from tiktoken import get_encoding

from src.agents.base_agent import BaseAgent
from src.interview_session.session_models import Message, MessageType, Participant
from src.agents.interviewer.interviewer import Interviewer, InterviewerConfig, TTSConfig
from src.agents.session_scribe.session_scribe import SessionScribe, SessionScribeConfig
from src.agents.strategic_planner.strategic_planner import StrategicPlanner, StrategicPlannerConfig
from src.agents.user.user_agent import UserAgent
from src.content.session_agenda.session_agenda import SessionAgenda, normalize_user_portrait
from src.utils.data_process import save_feedback_to_csv
from src.utils.logger.session_logger import SessionLogger, setup_logger
from src.utils.logger.evaluation_logger import EvaluationLogger
from src.interview_session.user.user import User
from src.interview_session.user.dummy_participant import UserDummyParticipant
from src.agents.report_team.orchestrator import ReportOrchestrator
from src.agents.report_team.base_report_agent import ReportConfig
from src.content.memory_bank.memory_bank_vector_db import VectorMemoryBank
from src.content.memory_bank.memory import Memory
from src.content.question_bank.question_bank_vector_db import QuestionBankVectorDB
from src.interview_session.prompts.conversation_summarize import summarize_conversation
from src.utils.token_tracker import TokenUsageTracker
from src.content.weekly_snapshot.snapshot_manager import SnapshotManager
from src.content.weekly_snapshot.weekly_snapshot import WeeklySnapshot, TaskEntry




class UserConfig(TypedDict, total=False):
    """Configuration for user settings.
    """
    user_id: str
    enable_voice: bool
    report_style: str


class InterviewConfig(TypedDict, total=False):
    """Configuration for interview settings."""
    enable_voice: bool
    interview_description: str
    interview_plan_path: str
    interview_evaluation: str
    additional_context_path: str
    initial_user_portrait_path: str
    session_type: str  # "intake" (default) or "weekly"

class BankConfig(TypedDict, total=False):
    """Configuration for memory and question banks."""
    memory_bank_type: str  # "vector_db", "graph_rag", etc.
    historical_question_bank_type: str  # "vector_db", "graph", "semantic", etc.


class InterviewSession:

    def __init__(self, interaction_mode: str = 'terminal', user_config: UserConfig = {},
                 interview_config: InterviewConfig = {}, bank_config: BankConfig = {},
                 use_baseline: Optional[bool] = None, max_turns: Optional[int] = None):
        """Initialize the interview session.

        Args:
            interaction_mode: How to interact with user 
                Options: 'terminal', 'agent', or 'api'
            user_config: User configuration dictionary
                user_id: User identifier (default: 'default_user')
                enable_voice: Enable voice input (default: False)
            interview_config: Interview configuration dictionary
                enable_voice: Enable voice output (default: False)
            bank_config: Bank configuration dictionary
                memory_bank_type: Type of memory bank 
                    Options: "vector_db", etc.
                historical_question_bank_type: Type of question bank 
                    Options: "vector_db", etc.
            use_baseline: Whether to use baseline prompt (default: read from .env)
            max_turns: Optional maximum number of turns before ending session
                      If None, session continues until manually ended
        """

        # Set the baseline mode for all agents
        if use_baseline is not None:
            # Set the class variable directly to affect all agent instances
            BaseAgent.use_baseline = use_baseline
        else:
            BaseAgent.use_baseline = \
                os.getenv("USE_BASELINE_PROMPT", "false").lower() == "true"
        
        # User setup
        self.user_id = user_config.get("user_id", "default_user")
        self._initial_additional_context_path = interview_config.get("additional_context_path", None)
        self._interview_description = interview_config.get("interview_description", "any topic")
        self._available_time = interview_config.get("available_time")  # minutes
        self.session_type = interview_config.get("session_type", "intake")

        # Session agenda setup
        self.session_agenda = SessionAgenda.get_last_session_agenda(self.user_id,
                                                                    initial_user_portrait_path=interview_config.get('initial_user_portrait_path'),
                                                                    interview_plan_path=interview_config.get('interview_plan_path'),
                                                                    interview_description=self._interview_description,
                                                                    interview_evaluation=interview_config.get('interview_evaluation'))
        self.session_id = self.session_agenda.session_id + 1
        self.session_agenda.available_time_minutes = self._available_time

        # Memory bank setup
        memory_bank_type = bank_config.get("memory_bank_type", "vector_db")
        if memory_bank_type == "vector_db":
            self.memory_bank = VectorMemoryBank.load_from_file(self.user_id)
            self.memory_bank.set_session_id(self.session_id)
        else:
            raise ValueError(f"Unknown memory bank type: {memory_bank_type}")

        # Question bank setup
        historical_question_bank_type = \
            bank_config.get("historical_question_bank_type", "vector_db")
        if historical_question_bank_type == "vector_db":
            self.historical_question_bank = \
                QuestionBankVectorDB.load_from_file(
                    self.user_id)
            self.historical_question_bank.set_session_id(self.session_id)
            self.proposed_question_bank = QuestionBankVectorDB()
        else:
            raise ValueError(
                f"Unknown question bank type: {historical_question_bank_type}")

        # Logger setup
        setup_logger(self.user_id, self.session_id,
                     console_output_files=["execution_log"])
        EvaluationLogger.setup_logger(self.user_id, self.session_id)

        # Token usage tracking setup
        self.token_tracker = TokenUsageTracker(
            session_id=str(self.session_id),
            user_id=self.user_id
        )
        # Set the class variable so all agents can access it
        BaseAgent.token_tracker = self.token_tracker

        # Chat history
        self.chat_history: list[Message] = []

        # Session states signals
        self.interaction_mode = interaction_mode
        self.session_in_progress = True
        self.session_completed = False
        self._session_ending = False  # Set when graceful end is in flight
        self._session_timeout = False
        self.paused = False           # session-level pause (for visualizer)
        self.step_requested = False   # single-step control
        self.max_turns = max_turns

        # End-of-session feedback collection state
        self._feedback_widget_sent = False
        self._feedback_submitted = False

        # Report auto-update states
        self.auto_report_update_in_progress = False
        self.memory_threshold = int(
            os.getenv("MEMORY_THRESHOLD_FOR_UPDATE", 10))
        
        # Conversation summary for auto-updates
        self.conversation_summary = ""
        
        # Counter for user messages to trigger auto-updates check
        self._user_message_count = 0
        self._check_interval = max(1, self.memory_threshold // 5)
        self._accumulated_auto_update_time = 0

        # Last message timestamp tracking for session timeout
        self._last_message_time = datetime.now()
        self._session_start_time = datetime.now()
        self._last_user_message = None
        self.timeout_minutes = int(os.getenv("SESSION_TIMEOUT_MINUTES", 10))

        # Time-limit extension state
        self._time_limit_triggered = False   # True once the time check fires
        self._awaiting_time_extension = False  # True while waiting for user's yes/no

        # User in the interview session
        if interaction_mode == 'agent':
            hesitancy = float(os.getenv("USER_AGENT_HESITANCY", "0.0"))
            self.user: User = UserAgent(
                user_id=self.user_id, interview_session=self,
                config=user_config, hesitancy=hesitancy)
        elif interaction_mode == 'terminal':
            self.user: User = User(user_id=self.user_id, interview_session=self,
                                   enable_voice_input=user_config \
                                   .get("enable_voice", False))
        elif interaction_mode == 'api':
            self.user: User = UserDummyParticipant(user_id=self.user_id, interview_session=self) # No direct user interface for API mode
        else:
            raise ValueError(f"Invalid interaction_mode: {interaction_mode}")

        # Agents in the interview session
        self._interviewer: Interviewer = Interviewer(
            config=InterviewerConfig(
                user_id=self.user_id,
                tts=TTSConfig(enabled=interview_config.get(
                    "enable_voice", False)),
                interview_description=self._interview_description,
            ),
            interview_session=self
        )
        # SessionScribe config with optional dedicated model
        scribe_config = SessionScribeConfig(user_id=self.user_id)

        # Use dedicated scribe model if configured
        scribe_model = os.getenv("SESSION_SCRIBE_MODEL_NAME")
        if scribe_model:
            scribe_config["model_name"] = scribe_model
            # Pass base_url if configured (for vLLM)
            scribe_base_url = os.getenv("SESSION_SCRIBE_VLLM_BASE_URL")
            if scribe_base_url:
                scribe_config["base_url"] = scribe_base_url

        self.session_scribe = SessionScribe(
            config=scribe_config,
            interview_session=self
        )
        
        # StrategicPlanner config
        # TODO: Tune strategic planner parameters
        planner_config = StrategicPlannerConfig(
                user_id=self.user_id,
                turn_trigger=int(os.getenv("STRATEGIC_PLANNER_TURN_TRIGGER", "3")),
                num_rollouts=int(os.getenv("STRATEGIC_PLANNER_NUM_ROLLOUTS", "3")),
                rollout_horizon=int(os.getenv("STRATEGIC_PLANNER_ROLLOUT_HORIZON", "3")),
                max_strategic_questions=int(os.getenv("STRATEGIC_PLANNER_MAX_QUESTIONS", "5")),
                alpha=float(os.getenv("STRATEGIC_PLANNER_ALPHA", "0.5")),  # Coverage weight
                beta=float(os.getenv("STRATEGIC_PLANNER_BETA", "0.3")),   # Cost penalty
                gamma=float(os.getenv("STRATEGIC_PLANNER_GAMMA", "0.2"))   # Emergence reward
        )
        
        # Use dedicated planner model if configured
        planner_model = os.getenv("STRATEGIC_PLANNER_MODEL_NAME")
        if planner_model:
            planner_config["model_name"] = planner_model
            # Pass base_url if configured (for vLLM)
            planner_base_url = os.getenv("STRATEGIC_PLANNER_VLLM_BASE_URL")
            if planner_base_url:
                planner_config["base_url"] = planner_base_url
        
        self.strategic_planner: StrategicPlanner = StrategicPlanner(
            config=planner_config,
            interview_session=self
        )
        self.report_orchestrator = ReportOrchestrator(
            config=ReportConfig(
                user_id=self.user_id,
                report_style=user_config.get(
                    "report_style", "chronological")
            ),
            interview_session=self
        )

        # Subscriptions of participants to each other
        self._subscriptions: Dict[str, List[Participant]] = {
            # Subscribers of Interviewer: Note-taker and User (in following code)
            "Interviewer": [self.session_scribe],
            # Subscribers of User: Interviewer, SessionScribe, and StrategicPlanner
            "User": [self._interviewer, self.session_scribe, self.strategic_planner]
        }

        # User participant for terminal interaction.
        # IMPORTANT: insert at the front so the user (which only buffers the
        # message for the frontend in API mode) is notified BEFORE the heavier
        # subscribers (e.g. SessionScribe, which performs synchronous OpenAI
        # embedding calls inside `_add_question_to_session_agenda` and can
        # block the session's event loop for hundreds of ms). Putting the user
        # first guarantees the typing bubble clears as soon as the message is
        # available, instead of waiting for scribe-side work to finish.
        if self.user:
            self._subscriptions["Interviewer"].insert(0, self.user)

        # User API participant for backend API interaction
        # self.api_participant = None
        # if interaction_mode == 'api':
        #     self.api_participant = UserDummyParticipant(interview_session=self)
        #     self._subscriptions["Interviewer"].append(self.api_participant)
        #     self._subscriptions["User"].append(self.api_participant)

        # Shutdown signal handler - only for agent mode
        if interaction_mode == 'agent':
            try:
                self._setup_signal_handlers()
            except Exception:
                pass  # Signal handlers unavailable in threaded/Flask contexts
        
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Interview session initialized")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] User ID: {self.user_id}")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Session ID: {self.session_id}")
        SessionLogger.log_to_file(
            "execution_log", f"[INIT] Use baseline: {BaseAgent.use_baseline}")
        
        self.tokenizer = get_encoding("cl100k_base")

    async def wait_if_paused(self):
        """Block the calling coroutine while the session is paused."""
        while self.paused and not self.step_requested:
            await asyncio.sleep(0.2)
        self.step_requested = False  # consume step token

    async def _notify_participants(self, message: Message):
        """Notify subscribers asynchronously"""
        # Gets subscribers for the user that sent the message.
        subscribers = self._subscriptions.get(message.role, [])
        SessionLogger.log_to_file(
            "execution_log", 
            (
                f"[NOTIFY] Notifying {len(subscribers)} subscribers "
                f"for message from {message.role}"
            )
        )

        # Pre-flight checks for user messages — run BEFORE creating subscriber
        # tasks so that flags like `paused` are visible to subscribers immediately.
        time_extension_intercept = False
        if message.role == "User":
            self._last_user_message = message
            self._user_message_count += 1
            BaseAgent.current_turn = self._user_message_count

            snapshot_path = self.token_tracker.save_snapshot()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved token usage snapshot to {snapshot_path}",
                log_level="info"
            )

            if self.max_turns is not None and \
                    self._user_message_count >= self.max_turns:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TURNS] Maximum turns ({self.max_turns}) reached. "
                    f"Ending session."
                )
                self.session_in_progress = False
                final_summary_path = self.token_tracker.save_final_summary()
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                    log_level="info"
                )

            elif self._awaiting_time_extension and not self._session_ending:
                time_extension_intercept = True
                self._awaiting_time_extension = False

            elif (not self._time_limit_triggered
                  and not self._awaiting_time_extension
                  and not self._session_ending
                  and self.session_agenda.available_time_minutes):
                elapsed = (datetime.now() - self._session_start_time).total_seconds() / 60
                if elapsed >= self.session_agenda.available_time_minutes:
                    self._time_limit_triggered = True
                    self._awaiting_time_extension = True
                    self.paused = True  # block interviewer before its task starts

        # Create independent tasks for each subscriber
        tasks = []
        for sub in subscribers:
            if self.session_in_progress:
                task = asyncio.create_task(sub.on_message(message))
                tasks.append(task)
        
        # Allow tasks to run concurrently without waiting for each other
        await asyncio.sleep(0)  # Explicitly yield control

        # Post-subscriber triggers (run as tasks so they don't block notification)
        if message.role == "User":
            if (self._user_message_count % self._check_interval == 0 and
                not self.auto_report_update_in_progress):
                asyncio.create_task(self._check_and_trigger_report_update())

            if time_extension_intercept:
                asyncio.create_task(self._handle_time_extension_response(message.content))

            elif self.paused and self._awaiting_time_extension:
                asyncio.create_task(self._offer_time_extension())

    def add_message_to_chat_history(self, role: str, content: str = "", 
                                    message_type: str = MessageType.CONVERSATION,
                                    metadata: dict = {}):
        """Add a message to the chat history"""

        # Reject messages if session is not in progress
        if not self.session_in_progress:
            return

        # Set fixed content for skip and like messages
        if message_type == MessageType.SKIP:
            content = "Skip the question"
        elif message_type == MessageType.LIKE:
            content = "Like the question"

        # Create message object
        message = Message(
            id=str(uuid.uuid4()),
            type=message_type,
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata,
        )

        if role == "User":
            self._last_message_time = message.timestamp
        elif role == "Interviewer" and self._last_user_message is not None:
            self._last_user_message = None
        
        # Log feedback (not for widget trigger messages)
        if message_type not in (MessageType.CONVERSATION, MessageType.TIME_SPLIT_WIDGET, MessageType.FEEDBACK_WIDGET):
            save_feedback_to_csv(
                self.chat_history[-1], message, self.user_id, self.session_id)

        # Notify participants if message is a skip, conversation, or UI widget trigger
        if message_type in (MessageType.SKIP, MessageType.CONVERSATION, MessageType.TIME_SPLIT_WIDGET, MessageType.FEEDBACK_WIDGET):

            # Add message to chat history
            self.chat_history.append(message)
            if message_type not in (MessageType.TIME_SPLIT_WIDGET, MessageType.FEEDBACK_WIDGET):
                SessionLogger.log_to_file(
                    "chat_history", f"{message.role}: {message.content}")

            # Notify participants
            asyncio.create_task(self._notify_participants(message))


        SessionLogger.log_to_file(
            "execution_log", 
            (
                f"[CHAT_HISTORY] {message.role}'s message has been added "
                f"to chat history."
            )
        )

    async def run(self):
        """Run the interview session"""
        # Start the opening question LLM call concurrently with agenda augmentation so
        # the first question is ready as soon as possible after the session is created.
        # For fresh sessions augmentation is a no-op, so there is no trade-off.
        # For sessions with prior context the opening question uses pre-augmentation
        # state (generic introduction), which is acceptable to avoid added latency.
        opening_task = asyncio.ensure_future(self._interviewer.on_message(None))

        await self.session_scribe.augment_session_agenda(additional_context_path=self._initial_additional_context_path)

        SessionLogger.log_to_file(
            "execution_log", f"[RUN] Starting interview session")
        self.session_in_progress = True

        # In-interview Processing
        try:
            await opening_task

            # Monitor the session for completion and timeout
            while self.session_in_progress or \
                self.session_scribe.processing_in_progress:
                await asyncio.sleep(0.1)

                # Check for timeout
                if datetime.now() - self._last_message_time \
                        > timedelta(minutes=self.timeout_minutes):
                    SessionLogger.log_to_file(
                        "execution_log", 
                        (
                            f"[TIMEOUT] Session timed out after "
                            f"{self.timeout_minutes} minutes of inactivity"
                        )
                    )
                    self.session_in_progress = False
                    self._session_timeout = True
                    break

        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[RUN] Unexpected error: {str(e)}")
            raise e

        # Post-interview Processing
        finally:
            try:
                self.session_in_progress = False

                # Update report and user portrait at session end
                with contextlib.suppress(KeyboardInterrupt):
                    SessionLogger.log_to_file(
                        "execution_log",
                        (
                            f"[REPORT] Trigger final report update. "
                            f"Waiting for session scribe to finish processing..."
                        )
                    )
                    await self.final_update_report_and_agenda(
                        selected_topics=[])

                # Wait for report update to complete if it's in progress
                start_time = time.time()
                while (self.report_orchestrator.report_update_in_progress or 
                       self.report_orchestrator.session_agenda_update_in_progress):
                    await asyncio.sleep(0.1)
                    if time.time() - start_time > 600:  # 10 minutes timeout
                        SessionLogger.log_to_file(
                            "execution_log", 
                            (
                                f"[REPORT] Timeout waiting for report update"
                            )
                        )
                        break

            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log", f"[RUN] Error during report update: \
                          {str(e)}")
            finally:
                # Save memory bank
                self.memory_bank.save_to_file(self.user_id)
                SessionLogger.log_to_file(
                    "execution_log", f"[COMPLETED] Memory bank saved")
                
                # Save historical question bank
                self.historical_question_bank.save_to_file(self.user_id)
                SessionLogger.log_to_file(
                    "execution_log", f"[COMPLETED] Question bank saved")

                # Generate and persist weekly snapshot for weekly sessions
                if self.session_type == "weekly":
                    try:
                        await self._generate_and_save_weekly_snapshot()
                    except Exception as snap_err:
                        SessionLogger.log_to_file(
                            "execution_log",
                            f"[SNAPSHOT] Error generating weekly snapshot: {snap_err}"
                        )

                # Generate and save user portrait from memories for intake sessions
                if self.session_type == "intake":
                    try:
                        await self._generate_and_save_user_portrait()
                    except Exception as portrait_err:
                        SessionLogger.log_to_file(
                            "execution_log",
                            f"[PORTRAIT] Error generating user portrait: {portrait_err}"
                        )

                self.session_completed = True
                SessionLogger.log_to_file(
                    "execution_log", f"[COMPLETED] Session completed")

    async def _generate_and_save_weekly_snapshot(self):
        """Extract a structured WeeklySnapshot from session memories and save it."""
        from src.agents.session_scribe.prompts import get_prompt as scribe_get_prompt
        from src.utils.llm.engines import get_engine, invoke_engine

        memories = await self.get_session_memories(include_processed=True)
        if not memories:
            SessionLogger.log_to_file(
                "execution_log", "[SNAPSHOT] No memories to extract snapshot from."
            )
            return

        memories_text = "\n".join(
            f"- [{m.title}] {m.text}" for m in memories
        )
        portrait = self.session_agenda.user_portrait

        prompt_template = scribe_get_prompt("extract_weekly_snapshot")
        prompt = prompt_template.format(
            memories=memories_text,
            user_portrait=str(portrait),
        )

        engine = get_engine(os.getenv("MODEL_NAME", "gpt-4.1-mini"))
        response = invoke_engine(engine, prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]

        try:
            import json
            data = json.loads(text.strip())
        except Exception:
            SessionLogger.log_to_file(
                "execution_log", "[SNAPSHOT] Failed to parse snapshot JSON."
            )
            return

        tasks = [TaskEntry(**t) for t in data.get("tasks", [])]
        snapshot = WeeklySnapshot(
            user_id=self.user_id,
            session_id=self.session_id,
            week_number=SnapshotManager.current_week_number(),
            tasks=tasks,
            tools=data.get("tools", []),
            ai_tools=data.get("ai_tools", []),
            collaborators=data.get("collaborators", []),
            time_allocation=data.get("time_allocation", {}),
            pain_points=data.get("pain_points", []),
            notable_changes=data.get("notable_changes", []),
            session_summary=data.get("session_summary", ""),
        )

        manager = SnapshotManager(self.user_id)
        path = manager.save_snapshot(snapshot)
        SessionLogger.log_to_file(
            "execution_log", f"[SNAPSHOT] Weekly snapshot saved to {path}"
        )

    async def _generate_and_save_user_portrait(self):
        """Synthesize all session memories into the user portrait schema and save to file."""
        from src.agents.session_scribe.prompts import get_prompt as scribe_get_prompt
        from src.utils.llm.engines import get_engine, invoke_engine

        memories = self.memory_bank.memories
        if not memories:
            return

        memories_text = "\n".join(
            f"- [{m.title}] {m.text}" for m in memories
        )
        current_portrait = self.session_agenda.user_portrait

        prompt_template = scribe_get_prompt("extract_user_portrait")
        prompt = prompt_template.format(
            memories=memories_text,
            user_portrait=str(current_portrait),
        )

        # Run the synchronous engine call in a thread to avoid blocking the event loop
        engine = get_engine(os.getenv("MODEL_NAME", "gpt-4.1-mini"))
        response = await asyncio.to_thread(invoke_engine, engine, prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences and find the JSON object
        text = text.strip()
        # Remove ```json ... ``` or ``` ... ``` fences
        if "```" in text:
            parts = text.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    text = part
                    break
        # Find the first { ... } block if there's surrounding text
        start = text.find("{")
        end   = text.rfind("}")
        if start != -1 and end != -1:
            text = text[start:end + 1]

        try:
            portrait_data = json.loads(text)
        except json.JSONDecodeError as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[PORTRAIT] JSON parse error: {e}. Raw text (first 300 chars): {text[:300]}"
            )
            return

        portrait_data = normalize_user_portrait(portrait_data)

        # Update the in-memory portrait
        self.session_agenda.user_portrait = portrait_data

        # Save to user directory so it becomes input for the next session
        portrait_path = os.path.join(
            os.getenv("LOGS_DIR"), self.user_id, "user_portrait.json"
        )
        os.makedirs(os.path.dirname(portrait_path), exist_ok=True)
        with open(portrait_path, "w") as f:
            json.dump(portrait_data, f, indent=2)

        SessionLogger.log_to_file(
            "execution_log", f"[PORTRAIT] User portrait updated ({len(memories)} memories)"
        )

    async def get_session_memories(self, include_processed=True) -> List[Memory]:
        """Get memories added during this session
        
        Args:
            include_processed: If True, returns all memories from the session
                              If False, returns only the unprocessed memories
        """
        return await self.session_scribe.get_session_memories(
            clear_processed=False, 
            wait_for_processing=True,
            include_processed=include_processed
        )

    async def _check_and_trigger_report_update(self):
        """Check if we have enough memories to trigger a report update"""
        # Skip if report update already in progress or session not in progress
        if self.auto_report_update_in_progress or \
           not self.session_in_progress or \
           self.report_orchestrator.report_update_in_progress:
            return
            
        # Get current memory count without clearing or waiting
        memories = await self.session_scribe \
            .get_session_memories(clear_processed=False,
                                   wait_for_processing=False)
        
        # Check if we've reached the threshold
        if len(memories) >= self.memory_threshold:
            SessionLogger.log_to_file(
                "execution_log",
                f"[AUTO-UPDATE] Triggering report update "
                f"with {len(memories)} memories"
            )
            
            try:
                self.auto_report_update_in_progress = True
                
                # Generate a summary of recent conversation
                await self._update_conversation_summary()
                
                # Get memories and clear them from the session scribe
                memories_to_process = \
                    await self.session_scribe.get_session_memories(
                        clear_processed=True, wait_for_processing=False)
                
                # Measure the time auto-update would take
                start_time = time.time()
                
                # Update report with these memories and the conversation summary
                # await self.report_orchestrator.update_report_with_memories(
                #     memories_to_process,
                #     is_auto_update=True
                # )
                
                # Record the time it took
                update_time = time.time() - start_time
                self._accumulated_auto_update_time += update_time
                
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[AUTO-UPDATE] Report update completed "
                    f"for {len(memories_to_process)} memories"
                )
                
            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log", 
                    f"[AUTO-UPDATE] Error during report update: {str(e)}"
                )
            finally:
                self.auto_report_update_in_progress = False
    
    async def _update_conversation_summary(self):
        """Generate a summary of recent conversation messages"""
        
        # Extract recent messages from chat history
        recent_messages: List[Message] = []
        for msg in self.chat_history[-self.session_scribe._max_events_len:]:
            if msg.type == MessageType.CONVERSATION:
                recent_messages.append(msg)
        
        # Generate summary if we have messages
        if recent_messages:
            self.conversation_summary = \
                summarize_conversation(recent_messages)
    
    async def final_update_report_and_agenda(self, selected_topics: Optional[List[str]] = None):
        """Trigger final report update"""
        # Record start time
        start_time = time.time()
        
        try:
            # Proceed with the final update
            await self.report_orchestrator.final_update_report_and_agenda(
                selected_topics=selected_topics,
                wait_time=self._accumulated_auto_update_time if \
                    (BaseAgent.use_baseline and 
                    self.interaction_mode == "api") else None
                # Simulate baseline mode without auto-updates for web user testing
            )
        finally:
            # Calculate and log duration
            duration = time.time() - start_time
            eval_logger = EvaluationLogger.setup_logger(
                self.user_id, self.session_id)
            eval_logger.log_report_update_time(
                update_type="final",
                duration=duration if not BaseAgent.use_baseline \
                    else (duration + self._accumulated_auto_update_time),
                accumulated_auto_time=self._accumulated_auto_update_time
                # Simulate baseline mode without auto-updates
            )

    async def _offer_time_extension(self):
        """Ask the participant if they want to continue for 5 more minutes.

        If the configured agenda is already fully covered and no active topic
        allows emergent exploration, skip the offer entirely and end the
        session gracefully — there are no more questions to ask, so soliciting
        extra time would only invite the interviewer to free-associate.
        """
        # Pause the interviewer so it doesn't generate a question while waiting
        self.paused = True

        topic_manager = self.session_agenda.interview_topic_manager
        incomplete_topics = topic_manager.get_all_incomplete_core_topic()
        allows_emergent = topic_manager.any_active_topic_allows_emergent()
        if not incomplete_topics and not allows_emergent:
            SessionLogger.log_to_file(
                "execution_log",
                "[TIME] Time limit reached but agenda is fully covered and no "
                "active topic allows emergent exploration — ending without "
                "offering an extension."
            )
            self._awaiting_time_extension = False
            self._session_ending = True
            await self._graceful_session_end()
            return

        available = self.session_agenda.available_time_minutes
        msg = (
            f"We've reached the {available}-minute mark — thank you for your time so far! "
            f"Do you have another 5 minutes to continue, or would you prefer to wrap up here?"
        )
        self.add_message_to_chat_history(role="Interviewer", content=msg)
        SessionLogger.log_to_file(
            "execution_log",
            f"[TIME] Offered {available}-min extension to participant"
        )

    async def _handle_time_extension_response(self, user_reply: str):
        """Interpret the participant's yes/no and either extend or end gracefully."""
        from src.utils.llm.engines import get_engine, invoke_engine
        try:
            classify_prompt = (
                f"The interviewer asked the participant if they have 5 more minutes. "
                f"The participant replied: \"{user_reply}\"\n"
                f"Reply with only one word: YES if they want to continue, NO if they want to stop."
            )
            engine = get_engine(os.getenv("MODEL_NAME", "gpt-4.1-mini"))
            result = await asyncio.to_thread(invoke_engine, engine, classify_prompt)
            verdict = (result.content if hasattr(result, "content") else str(result)).strip().upper()
        except Exception:
            verdict = "NO"

        if "YES" in verdict:
            # Even if the participant wants to continue, only extend when there
            # is still uncovered ground in the configured agenda. Otherwise the
            # interviewer would free-associate beyond its script (e.g. drilling
            # into emergent threads that the topic config explicitly opted out
            # of). Wrap up gracefully instead.
            topic_manager = self.session_agenda.interview_topic_manager
            incomplete_topics = topic_manager.get_all_incomplete_core_topic()
            allows_emergent = topic_manager.any_active_topic_allows_emergent()
            if not incomplete_topics and not allows_emergent:
                SessionLogger.log_to_file(
                    "execution_log",
                    "[TIME] Participant said yes, but agenda is fully covered "
                    "and no active topic allows emergent exploration — ending gracefully."
                )
                self._awaiting_time_extension = False
                self._session_ending = True
                await self._graceful_session_end()
                return

            # Extend by 5 minutes and let the interview continue
            self.session_agenda.available_time_minutes += 5
            self._time_limit_triggered = False  # allow another check in 5 min
            self._awaiting_time_extension = False
            self.paused = False  # resume the interviewer
            self.add_message_to_chat_history(
                role="Interviewer",
                content="Great, let's keep going for a few more minutes!"
            )
            SessionLogger.log_to_file("execution_log", "[TIME] Participant extended — adding 5 minutes")
        else:
            # End gracefully
            SessionLogger.log_to_file("execution_log", "[TIME] Participant chose to end — sending goodbye")
            self._session_ending = True
            await self._graceful_session_end()

    async def _graceful_session_end(self):
        """Send a closing message from the interviewer, then end the session."""
        try:
            from src.utils.llm.engines import get_engine, invoke_engine
            # Include recent chat so the closing message reflects what was actually discussed.
            recent = self.chat_history[-10:] if len(self.chat_history) >= 10 else self.chat_history
            history_str = "\n".join(
                f"{m.role}: {m.content}" for m in recent if m.type == "conversation"
            )
            closing_prompt = (
                "You are an interviewer closing out a session. Here is the recent conversation:\n\n"
                f"{history_str}\n\n"
                "Write a 2-3 sentence closing message. "
                "Start with a brief, natural acknowledgment of the participant's last response "
                "(one sentence that reflects something specific they just said — not generic filler). "
                "Then signal that the session is complete and thank them warmly. "
                "Only reference things that actually came up in the conversation. "
                "Do not ask any questions."
            )
            engine = get_engine(os.getenv("MODEL_NAME", "gpt-4.1-mini"))
            response = await asyncio.to_thread(invoke_engine, engine, closing_prompt)
            goodbye_msg = response.content if hasattr(response, "content") else str(response)
            # Send via chat history so the user sees it
            self.add_message_to_chat_history(role="Interviewer", content=goodbye_msg)
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GOODBYE] Error generating closing message: {e}"
            )
            self.add_message_to_chat_history(
                role="Interviewer",
                content="Thank you so much for your time today — I really appreciate you sharing all of that. We're all wrapped up!"
            )

        # Yield to the event loop so _notify_participants can run and populate
        # the user's message buffer before we emit the feedback widget.
        await asyncio.sleep(1)

        # Show the feedback widget instead of ending immediately — the session
        # will truly end once the participant submits the form.
        self.trigger_feedback_widget()

    def trigger_feedback_widget(self):
        """Send the end-of-session feedback widget and defer true session end
        until the user submits feedback (via /api/submit-feedback) or
        `complete_session_after_feedback` is called.

        Safe to call multiple times — only the first call emits the widget.
        """
        if self._feedback_widget_sent:
            return
        self._feedback_widget_sent = True
        # Mark as ending so the interviewer stops producing new messages,
        # but keep session_in_progress True so the frontend can still poll
        # and submit feedback.
        self._session_ending = True
        # Reset the inactivity clock so the idle-timeout doesn't kill the
        # session while the participant is filling in the widget.
        self._last_message_time = datetime.now()
        self.add_message_to_chat_history(
            role="Interviewer",
            content="",
            message_type=MessageType.FEEDBACK_WIDGET,
        )
        SessionLogger.log_to_file(
            "execution_log",
            "[FEEDBACK] Feedback widget emitted; awaiting submission."
        )

    def submit_feedback(self, feedback: dict):
        """Persist end-of-session feedback and finalize session completion.

        Called by the /api/submit-feedback endpoint after the participant
        fills the feedback widget.
        """
        self._feedback_submitted = True
        try:
            logs_dir = os.getenv("LOGS_DIR", "logs")
            feedback_dir = os.path.join(
                logs_dir, self.user_id, "execution_logs", f"session_{self.session_id}"
            )
            os.makedirs(feedback_dir, exist_ok=True)
            feedback_path = os.path.join(feedback_dir, "feedback.json")
            with open(feedback_path, "w") as f:
                json.dump(feedback, f, indent=2)
            SessionLogger.log_to_file(
                "execution_log", f"[FEEDBACK] Saved feedback to {feedback_path}"
            )
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log", f"[FEEDBACK] Error saving feedback: {e}"
            )

        # Now truly end the session — run()'s finally block will handle
        # the report/portrait/snapshot finalization.
        self.session_in_progress = False
        try:
            final_summary_path = self.token_tracker.save_final_summary()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                log_level="info",
            )
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Error saving final summary: {e}",
                log_level="warning",
            )

    def end_session(self):
        """End the session without triggering report update"""
        self.session_in_progress = False

        # Save final token usage summary
        if hasattr(self, 'token_tracker'):
            final_summary_path = self.token_tracker.save_final_summary()
            SessionLogger.log_to_file(
                "execution_log",
                f"[TOKEN_TRACKING] Saved final token usage summary to {final_summary_path}",
                log_level="info"
            )

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._signal_handler)

    def _signal_handler(self):
        """Handle shutdown signals"""
        self.session_in_progress = False
        SessionLogger.log_to_file(
            "execution_log", f"[SIGNAL] Shutdown signal received")
        SessionLogger.log_to_file(
            "execution_log", f"[SIGNAL] Waiting for interview session to finish...")
    
    def set_db_session_id(self, db_session_id: int):
        """Set the database session ID. Used for server mode"""
        self.db_session_id = db_session_id

    def get_db_session_id(self) -> int:
        """Get the database session ID. Used for server mode"""
        return self.db_session_id
        