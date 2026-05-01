import asyncio
import os
import re
import time
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
from src.utils.llm.engines import get_engine, invoke_engine

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
        self._ai_question_asked = False
        self._response_lock = asyncio.Lock()
        self._turn_start: float | None = None
        self._last_sent_message_text: str = ""
        self._guard_llm_mode = str(
            os.getenv("INTERVIEWER_GUARD_LLM_MODE", "off")
        ).strip().lower()
        self._inferability_mode = str(
            os.getenv("INTERVIEWER_INFERABILITY_MODE", "auto")
        ).strip().lower()
        try:
            self._inferability_threshold = float(
                os.getenv("INTERVIEWER_INFERABILITY_THRESHOLD", "0.80")
            )
        except (TypeError, ValueError):
            self._inferability_threshold = 0.80
        self._inferability_threshold = max(0.50, min(0.99, self._inferability_threshold))
        # Temporarily hard-disable rewrite path to reduce latency and avoid
        # unintended prompt drift in production interviews.
        self._rewrite_mode = "off"
        try:
            self._prompt_chat_events_limit = int(
                os.getenv("INTERVIEWER_PROMPT_CHAT_EVENTS", "16")
            )
        except (TypeError, ValueError):
            self._prompt_chat_events_limit = 16
        self._prompt_chat_events_limit = max(8, self._prompt_chat_events_limit)
        try:
            self._prompt_recent_interviewer_limit = int(
                os.getenv("INTERVIEWER_PROMPT_RECENT_INTERVIEWER", "8")
            )
        except (TypeError, ValueError):
            self._prompt_recent_interviewer_limit = 8
        self._prompt_recent_interviewer_limit = max(4, self._prompt_recent_interviewer_limit)

        try:
            self._prompt_recent_user_answers_limit = int(
                os.getenv("INTERVIEWER_PROMPT_RECENT_USER_ANSWERS", "8")
            )
        except (TypeError, ValueError):
            self._prompt_recent_user_answers_limit = 8
        self._prompt_recent_user_answers_limit = max(4, self._prompt_recent_user_answers_limit)
        self._judge_engine = None

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

    def _guard_llm_checks_enabled(self) -> bool:
        """Return whether post-generation guard LLM calls may run."""
        mode = str(
            os.getenv("INTERVIEWER_GUARD_LLM_MODE", self._guard_llm_mode)
        ).strip().lower()
        return mode in {"on", "true", "1", "yes", "auto"}

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
        delivered = await self._handle_response(response, subtopic_id)
        self._last_sent_message_text = delivered
        return delivered

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
        self._last_sent_message_text = goodbye
        self.add_event(sender=self.name, tag="goodbye", content=goodbye)
        self.interview_session.add_message_to_chat_history(
            role=self.title, content=goodbye)

    def _last_sent_message_is_question(self) -> bool:
        """Return True when the interviewer's last delivered message is a question."""
        return "?" in str(self._last_sent_message_text or "")

    def _get_judge_engine(self):
        """Lazily construct a small LLM used for lightweight classification calls."""
        if self._judge_engine is None:
            model_name = os.getenv("INTERVIEWER_JUDGE_MODEL", "gpt-4.1-mini")
            self._judge_engine = get_engine(model_name=model_name)
        return self._judge_engine

    async def _looks_like_closing_goodbye(self, text: str) -> bool:
        """Classify whether a `respond_to_user` message is actually a goodbye/closing.

        Uses a small LLM instead of keyword matching so natural-sounding closings
        ("thanks for the chat, that's all I needed for today") are recognized.
        """
        t = self._normalize_text(text)
        if not t or "?" in t:
            return False

        judge_prompt = (
            "You decide whether an interviewer's message is intended as a closing/goodbye "
            "at the end of a session, versus an in-progress conversational turn that just "
            "happens to lack a question mark (e.g. a pure acknowledgment).\n\n"
            "A CLOSING signals the session is over: thanks + wrap-up intent, "
            "'we're all set', 'that covers what I needed', 'we'll wrap here', etc.\n"
            "An IN-PROGRESS message acknowledges what the user said but does not signal "
            "the session is ending (e.g. 'thanks for laying that out, that's a clear "
            "picture of your week' with no wrap-up).\n\n"
            f"Message:\n\"\"\"\n{t}\n\"\"\"\n\n"
            "Return JSON only:\n"
            "{\n"
            "  \"is_closing\": true or false,\n"
            "  \"reason\": \"short reason\"\n"
            "}\n"
        )

        try:
            response = await asyncio.to_thread(
                invoke_engine, self._get_judge_engine(), judge_prompt
            )
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] closing_goodbye_check LLM call failed: {e}",
                log_level="warning",
            )
            return False

        parsed = self._extract_json_dict(raw)
        if not isinstance(parsed, dict):
            return False
        return bool(parsed.get("is_closing"))

    async def _classify_completeness_probe_reply(
        self,
        latest_question: str,
        participant_reply: str,
    ) -> dict:
        """Semantically classify replies to the final missed-task probe."""
        prompt = (
            "You are classifying a participant reply in a work-task inventory interview.\n\n"
            "The interviewer has just asked whether any work tasks are still missing.\n"
            "Classify the participant reply by meaning, not by exact wording or length.\n\n"
            f"Latest interviewer question:\n{latest_question or '(unknown)'}\n\n"
            f"Participant reply:\n{participant_reply or '(empty)'}\n\n"
            "Decisions:\n"
            "- closure: the participant is saying there are no additional tasks to add.\n"
            "- new_task: the participant mentions additional work, a task, an activity, or a responsibility that may need to be added.\n"
            "- unclear_response: the reply is noise, gibberish, accidental text, or does not clearly answer whether there are more tasks.\n\n"
            "Rules:\n"
            "- Do not treat a short reply as closure unless its meaning clearly indicates closure.\n"
            "- If the participant names work they do, choose new_task even if the wording is brief.\n"
            "- If the reply is unclear, provide one short clarification question asking whether they want to add another task or are done with the task list.\n\n"
            "Return ONLY valid JSON:\n"
            '{"decision": "closure|new_task|unclear_response", "clarification_question": "", "reason": "short reason"}'
        )

        try:
            raw = await self.call_engine_async(prompt)
            parsed = self._extract_json_dict(raw)
            if not isinstance(parsed, dict):
                return {"decision": "unclear_response"}
            decision = str(parsed.get("decision", "")).strip().lower()
            if decision not in {"closure", "new_task", "unclear_response"}:
                decision = "unclear_response"
            parsed["decision"] = decision
            return parsed
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[PROBE-GUARD] semantic classifier failed: {e}",
                log_level="warning",
            )
            return {"decision": "unclear_response"}

    def _no_active_topics_remaining(self) -> bool:
        """Return True when agenda has no active or incomplete topics.

        This corresponds to the edge case where prompt topic blocks become empty.
        """
        topic_manager = self.interview_session.session_agenda.interview_topic_manager
        return (
            len(topic_manager.active_topic_id_list) == 0
            and len(topic_manager.get_all_incomplete_core_topic()) == 0
        )


    def _default_completion_goodbye(self) -> str:
        """Deterministic close used when agenda is complete before an LLM turn."""
        if getattr(self.interview_session, "session_type", "intake") == "weekly":
            return "That’s everything I needed for this week — we’ll wrap here."
        return "That’s everything I needed — we’ll wrap here."

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    def _is_breadth_probe_question(self, text: str) -> bool:
        """Detect broad catch-all questions that often become repetitive."""
        t = self._normalize_text(text).lower()
        if "?" not in t:
            return False
        if "anything else" in t or "what else" in t:
            return True
        if "anything not captured" in t or "anything not mentioned" in t:
            return True
        if ("besides" in t or "outside of" in t) and "anything" in t:
            return True
        if "what other tasks" in t or ("what other" in t and "work" in t):
            return True
        return False

    def _extract_trailing_question(self, text: str) -> str:
        """Extract the last question sentence from a message."""
        t = self._normalize_text(text)
        if "?" not in t:
            return ""
        parts = re.split(r'(?<=[.!?])\s+', t)
        for part in reversed(parts):
            if "?" in part:
                return part.strip()
        return ""

    def _is_repetition_signal(self, text: str) -> bool:
        t = self._normalize_text(text).lower()
        markers = (
            "you keep asking",
            "already answered",
            "am i not providing enough",
            "i don't really understand",
            "dont really understand",
            "same question",
        )
        return any(m in t for m in markers)

    def _content_tokens(self, text: str) -> set[str]:
        """Lightweight tokenization for overlap-based duplicate-risk gating."""
        raw = re.findall(r"[a-z0-9]+", self._normalize_text(text).lower())
        stop = {
            "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "with",
            "is", "are", "do", "did", "does", "you", "your", "it", "that", "this",
            "what", "which", "how", "when", "where", "why", "can", "could", "would",
            "should", "about", "from", "into", "over", "under", "at", "by", "as",
            "have", "has", "had", "been", "be", "i", "we", "they", "he", "she",
        }
        out: set[str] = set()
        for tok in raw:
            if tok in stop:
                continue
            if tok.endswith("ies") and len(tok) > 4:
                tok = tok[:-3] + "y"
            elif tok.endswith("ing") and len(tok) > 5:
                tok = tok[:-3]
            elif tok.endswith("ed") and len(tok) > 4:
                tok = tok[:-2]
            elif tok.endswith("s") and len(tok) > 3:
                tok = tok[:-1]
            if tok and tok not in stop:
                out.add(tok)
        return out

    def _question_overlap_score(self, a: str, b: str) -> float:
        ta = self._content_tokens(a)
        tb = self._content_tokens(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        denom = min(len(ta), len(tb))
        return inter / denom if denom > 0 else 0.0

    def _extract_json_dict(self, raw: str) -> dict | None:
        """Best-effort parse for generic JSON dictionary payloads."""
        text = str(raw or "").strip()
        if not text:
            return None

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue

        decoder = json.JSONDecoder()
        start = text.find("{")
        while start != -1:
            try:
                parsed, _ = decoder.raw_decode(text[start:])
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                start = text.find("{", start + 1)
        return None

    def _get_subtopic(self, subtopic_id: str):
        sid = str(subtopic_id or "").strip()
        if not sid:
            return None
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return None
        topic_id = sid.split(".", 1)[0]
        core_topic = agenda.interview_topic_manager.get_core_topic(topic_id)
        if core_topic is None:
            return None
        return core_topic.get_subtopic(sid)

    def _get_subtopic_inference_context(self, subtopic_id: str) -> str:
        """Return compact subtopic context for inferability checks."""
        subtopic = self._get_subtopic(subtopic_id)
        if subtopic is None:
            return "(unknown subtopic)"

        criteria = list(getattr(subtopic, "coverage_criteria", []) or [])
        statuses = list(getattr(subtopic, "criteria_coverage", []) or [])
        if len(criteria) == len(statuses) and criteria:
            unmet = [c for c, covered in zip(criteria, statuses) if not covered]
        else:
            unmet = criteria
        if not unmet:
            unmet = criteria

        notes = list(getattr(subtopic, "notes", []) or [])
        note_block = "\n".join(f"- {n}" for n in notes[-6:]) if notes else "(none)"
        unmet_block = "\n".join(f"- {c}" for c in unmet[:6]) if unmet else "(none)"
        return (
            f"Subtopic description: {subtopic.description}\n"
            f"Unmet coverage criteria:\n{unmet_block}\n"
            f"Recent subtopic notes:\n{note_block}"
        )

    def _should_run_inferability_llm(self, proposed_question: str, subtopic_id: str = "") -> bool:
        """Gate inferability checks to avoid asking answerable questions."""
        if not self._guard_llm_checks_enabled():
            return False

        mode = self._inferability_mode
        if mode == "off":
            return False

        q = self._normalize_text(proposed_question)
        if not q or "?" not in q:
            return False

        recent_answers = self._get_recent_user_answers(limit=8)
        if not recent_answers:
            return False

        if mode == "always":
            return True

        if self._is_task_list_collection_subtopic(subtopic_id):
            return True
        if self._is_breadth_probe_question(q):
            return True

        q_l = q.lower()
        inferable_markers = (
            "main thing",
            "actually producing",
            "working on most",
            "what are you actually",
            "main outcome",
            "what's the output",
        )
        if any(marker in q_l for marker in inferable_markers):
            return True

        for ans in recent_answers[-3:]:
            if self._question_overlap_score(q, ans) >= 0.75:
                return True
        return False

    async def _check_inferable_question(
        self,
        proposed_question: str,
        subtopic_id: str = "",
    ) -> tuple[bool, str]:
        """LLM judge: determine if question answer is already inferable from context."""
        check_start = time.perf_counter()
        recent_answers = self._get_recent_user_answers(limit=10)
        recent_qs = self._get_recent_interviewer_questions(limit=8)
        subtopic_context = self._get_subtopic_inference_context(subtopic_id)

        judge_prompt = (
            "You are a strict interview quality gate.\n"
            "Policy: only ask questions whose answer cannot be reasonably and confidently inferred from known context.\n\n"
            f"Candidate question:\n{proposed_question}\n\n"
            f"Current subtopic id: {subtopic_id or '(none)'}\n"
            f"{subtopic_context}\n\n"
            "Recent interviewer questions (oldest -> newest):\n"
            + "\n".join(f"- {q}" for q in recent_qs)
            + "\n\nRecent user answers (oldest -> newest):\n"
            + "\n".join(f"- {a}" for a in recent_answers)
            + "\n\nDecision rule:\n"
              "- inferable=true when a reasonable interviewer can answer the question from existing context with high confidence.\n"
              "- If inferable=true, provide ONE replacement question that targets genuinely missing information.\n"
              "- Replacement must be one sentence, non-leading, no examples/options, and exactly one question mark.\n\n"
              "Return JSON only:\n"
              "{\n"
              "  \"inferable\": true or false,\n"
              "  \"confidence\": 0.0 to 1.0,\n"
              "  \"reason\": \"short reason\",\n"
              "  \"replacement_question\": \"question text or empty\"\n"
              "}\n"
        )

        llm_s = 0.0
        try:
            llm_start = time.perf_counter()
            response = await asyncio.to_thread(invoke_engine, self.engine, judge_prompt)
            llm_s = time.perf_counter() - llm_start
            raw = response.content if hasattr(response, "content") else str(response)
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] inferability_check LLM call failed: {e}",
                log_level="warning",
            )
            return False, ""

        parse_start = time.perf_counter()
        parsed = self._extract_json_dict(raw)
        parse_s = time.perf_counter() - parse_start
        total_s = time.perf_counter() - check_start
        SessionLogger.log_to_file(
            "execution_log",
            f"[LATENCY] inferability_check: llm={llm_s:.2f}s parse={parse_s:.2f}s total={total_s:.2f}s "
            f"subtopic_id={subtopic_id or '(none)'}"
        )
        if not isinstance(parsed, dict):
            return False, ""

        inferable = bool(parsed.get("inferable"))
        replacement = self._normalize_text(parsed.get("replacement_question", ""))
        confidence_raw = parsed.get("confidence", 0.0)
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.0

        triggered = inferable or confidence >= self._inferability_threshold
        if triggered and replacement and "?" in replacement:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] inferability flagged: confidence={confidence:.2f} "
                f"original={proposed_question[:120]!r} "
                f"replacement={replacement[:120]!r}"
            )
            return True, replacement
        if triggered:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] inferability flagged without usable rewrite: confidence={confidence:.2f} "
                f"original={proposed_question[:120]!r}",
                log_level="warning",
            )
        return False, ""

    def _is_task_list_collection_subtopic(self, subtopic_id: str) -> bool:
        """Return True for Task Inventory list-collection subtopics (not priority/time)."""
        sid = str(subtopic_id or "").strip()
        if not sid:
            return False
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return False

        for topic in agenda.interview_topic_manager:
            topic_desc = str(getattr(topic, "description", "")).lower()
            for st in topic.required_subtopics.values():
                if str(st.subtopic_id) != sid:
                    continue

                desc = str(getattr(st, "description", "")).lower()
                if "task inventory" not in topic_desc and "task inventory" not in desc:
                    return False
                if "time allocation" in desc or "priority task" in desc:
                    return False
                return any(
                    marker in desc for marker in (
                        "typical week",
                        "task list completion",
                        "breadth",
                        "list of tasks",
                        "base task list",
                    )
                )
        return False

    def _is_task_list_completion_subtopic(self, subtopic_id: str) -> bool:
        """Return True when subtopic is the lower-cadence Task List Completion sweep."""
        sid = str(subtopic_id or "").strip()
        if not sid:
            return False
        subtopic = self._get_subtopic(sid)
        if subtopic is None:
            return False
        desc = str(getattr(subtopic, "description", "")).lower()
        return "task list completion" in desc or "lower-cadence" in desc

    def _should_run_semantic_duplicate_llm(self, proposed_question: str, subtopic_id: str = "") -> bool:
        """Gate expensive semantic-duplicate checks behind cheap local risk signals."""
        if not self._guard_llm_checks_enabled():
            return False

        mode = str(os.getenv("INTERVIEWER_SEMANTIC_DUP_MODE", "auto")).strip().lower()
        if mode == "off":
            return False
        if mode == "always":
            return True

        q = self._normalize_text(proposed_question)
        if not q or "?" not in q:
            return False
        q_l = q.lower()

        # Broad catch-all probes are the most common repeat source.
        if self._is_breadth_probe_question(q):
            return True

        recent_qs = self._get_recent_interviewer_questions(limit=8)
        if len(recent_qs) < 2:
            return False

        # Task List Completion (2.2-like) should only pay semantic LLM cost when
        # cadence-sweep wording is likely duplicative with nearby questions.
        if self._is_task_list_completion_subtopic(subtopic_id):
            cadence_markers = (
                "month or quarter",
                "monthly",
                "quarterly",
                "other recurring",
                "other types of work",
                "anything else",
                "beyond those",
            )
            if any(marker in q_l for marker in cadence_markers):
                for prev in recent_qs[-4:]:
                    if self._question_overlap_score(q, prev) >= 0.70:
                        return True
            return False

        # Only run the LLM judge when lexical overlap is already high.
        for prev in recent_qs[-4:]:
            if self._question_overlap_score(q, prev) >= 0.85:
                return True

        # Catch verbatim or near-verbatim trailing questions even when the full
        # message differs (e.g. different acknowledgment prefix, same closing question).
        trailing = self._extract_trailing_question(q)
        if trailing and len(trailing.split()) >= 4:
            for prev in recent_qs[-6:]:
                prev_trailing = self._extract_trailing_question(prev)
                if prev_trailing and self._question_overlap_score(trailing, prev_trailing) >= 0.75:
                    return True

        return False

    def _rewrite_question_locally(self, question: str, subtopic_id: str = "") -> str:
        """Fast deterministic rewrite for common bad patterns without another LLM call."""
        q = self._normalize_text(question)
        if not q or "?" not in q:
            return q
        q_l = q.lower()

        # Collapse count+duration asks into one qualitative question.
        if ("how many" in q_l and "how long" in q_l) or (
            ("how many" in q_l or "how long" in q_l or "how much" in q_l)
            and " and " in q_l
        ):
            anchor = ""
            recent_answers = self._get_recent_user_answers(limit=4)
            for ans in reversed(recent_answers):
                anchor = self._extract_anchor_clause(ans)
                if anchor:
                    break
            if "meeting" in q_l or "collabor" in q_l:
                if anchor:
                    return f"You mentioned {anchor}. What do those meetings usually focus on?"
                return "What do those meetings usually focus on?"
            if anchor:
                return f"You mentioned {anchor}. What does that usually involve in practice?"
            return "What does that usually involve in practice?"

        # Remove example-list wording in monthly/quarterly sweep questions.
        if self._is_task_list_completion_subtopic(subtopic_id):
            if ("month" in q_l or "quarter" in q_l) and (
                "like " in q_l or "for example" in q_l or "such as" in q_l
            ):
                return (
                    "Over a typical month or quarter, are there any other recurring kinds "
                    "of work you spend meaningful time on?"
                )
        return q

    def _should_run_llm_rewrite(self, proposed_question: str, subtopic_id: str = "") -> bool:
        """Gate LLM rewrite calls to high-risk turns to reduce latency."""
        mode = self._rewrite_mode
        if mode == "off":
            return False
        if mode == "always":
            return True

        q = self._normalize_text(proposed_question)
        if not q or "?" not in q:
            return False
        q_l = q.lower()

        # Known risky patterns.
        if self._is_breadth_probe_question(q):
            return True
        if q.count("?") > 1:
            return True
        if " and " in q_l and ("how many" in q_l or "how long" in q_l or "how much" in q_l):
            return True

        quant_markers = ("how many", "how long", "how much", "number of", "percent", "%", "hours", "minutes")
        quant_hits = sum(1 for marker in quant_markers if marker in q_l)
        if quant_hits >= 2:
            return True

        # For task-list collection subtopics, only rewrite when question shape is
        # likely to drift into count/duration detail that criteria do not require.
        if self._is_task_list_collection_subtopic(subtopic_id):
            if quant_hits >= 1:
                return True
            if q.count("?") > 1:
                return True

        # High lexical overlap with recent interviewer questions is duplicate-prone.
        recent_qs = self._get_recent_interviewer_questions(limit=6)
        for prev in recent_qs[-3:]:
            if self._question_overlap_score(q, prev) >= 0.92:
                return True

        return False

    def _get_recent_interviewer_questions(self, limit: int = 6) -> list[str]:
        questions = [
            str(event.content).strip()
            for event in self.event_stream
            if event.sender == "Interviewer"
            and event.tag == "message"
            and "?" in str(event.content)
        ]
        if limit <= 0:
            return questions
        return questions[-limit:]

    def _extract_anchor_clause(self, text: str, max_words: int = 14) -> str:
        """Extract a concrete clause from a user answer to anchor a specific follow-up."""
        raw = self._normalize_text(text)
        if not raw:
            return ""
        chunks = re.split(r"[.?!;,\n]|(?:\s[—-]\s)", raw)
        for chunk in chunks:
            c = self._normalize_text(chunk)
            if not c or self._is_repetition_signal(c):
                continue
            c = re.sub(r"^(sure|yeah|yea|yes|well|so|and|i mean)\b[ ,:-]*", "", c, flags=re.I)
            c = re.sub(r"^i('m| am)\s+", "", c, flags=re.I)
            c = re.sub(r"^i\s+", "", c, flags=re.I)
            words = c.split()
            if len(words) < 3:
                continue
            if len(words) > max_words:
                c = " ".join(words[:max_words])
            return c.strip(" ,;-")
        return ""

    async def _check_semantic_duplicate(
        self,
        proposed_question: str,
        subtopic_id: str = "",
    ) -> tuple[bool, str]:
        """LLM judge: is proposed_question a semantic duplicate of any recent interviewer question?

        Returns (is_duplicate, replacement_question). If is_duplicate is True and a
        valid replacement was produced, the caller should send the replacement instead.
        """
        check_start = time.perf_counter()
        recent_qs = self._get_recent_interviewer_questions(limit=8)
        if not recent_qs:
            return False, ""
        recent_answers = self._get_recent_user_answers(limit=3)
        last_answer = recent_answers[-1] if recent_answers else ""
        subtopic_context = self._get_subtopic_inference_context(subtopic_id)

        recent_qs_block = "\n".join(f"- {q}" for q in recent_qs)
        judge_prompt = (
            "You are judging whether a proposed interview question is a semantic "
            "duplicate of any recent interviewer question already asked in this session.\n\n"
            f"Current subtopic id: {subtopic_id or '(none)'}\n"
            f"{subtopic_context}\n\n"
            "A question is a DUPLICATE if its core information goal overlaps with a "
            "prior question's — even when:\n"
            "- The wording is different\n"
            "- The time scope is narrower or broader (week vs day, a specific instance "
            "vs a typical one)\n"
            "- It \"zooms in\" or \"zooms out\" from the prior framing\n"
            "- It rephrases the same abstract ask (e.g. \"what are you aiming for\" vs "
            "\"what's the goal in your mind\")\n\n"
            "Numeric policy:\n"
            "- Only ask numeric count/duration/percentage when the active subtopic's "
            "coverage criteria explicitly require numbers.\n"
            "- In Task Inventory list-collection/completion turns, avoid count/duration "
            "questions and prefer qualitative follow-ups that still satisfy cadence/"
            "closure/action+object criteria.\n"
            "- Never introduce two numeric dimensions in one question.\n\n"
            "A question is NOT a duplicate if it targets a genuinely different dimension: "
            "cadence/frequency, a recent specific instance, a named tool, an "
            "assessment/outcome, a collaborator, a coordination detail, or a concrete "
            "artifact.\n\n"
            f"Recent interviewer questions (oldest → newest):\n{recent_qs_block}\n\n"
            f"User's most recent answer:\n{last_answer or '(none)'}\n\n"
            f"Proposed next question:\n{proposed_question}\n\n"
            "Respond with JSON only, no prose:\n"
            "{\n"
            "  \"is_duplicate\": true or false,\n"
            "  \"matches\": \"the prior question it duplicates, verbatim, or empty string\",\n"
            "  \"reason\": \"one short sentence\",\n"
            "  \"replacement\": \"if is_duplicate is true, a specific replacement question that "
            "(a) anchors on a concrete noun from the user's last answer, "
            "(b) targets a DIFFERENT dimension (cadence/frequency, recent instance, "
            "artifact, tool, outcome, collaborator, coordination detail), and (c) is NOT a reworded abstract "
            "question. Otherwise empty string.\"\n"
            "}\n"
        )

        llm_s = 0.0
        try:
            llm_start = time.perf_counter()
            raw = await self.call_engine_async(judge_prompt)
            llm_s = time.perf_counter() - llm_start
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] semantic_duplicate_check LLM call failed: {e}",
                log_level="warning",
            )
            return False, ""

        parse_start = time.perf_counter()
        parsed = self._extract_tool_json_dict(raw)
        if not isinstance(parsed, dict):
            try:
                parsed = json.loads(str(raw).strip())
            except Exception:
                m = re.search(r"\{.*\}", str(raw), re.DOTALL)
                if not m:
                    return False, ""
                try:
                    parsed = json.loads(m.group(0))
                except Exception:
                    return False, ""

        parse_s = time.perf_counter() - parse_start
        total_s = time.perf_counter() - check_start
        SessionLogger.log_to_file(
            "execution_log",
            f"[LATENCY] semantic_duplicate_check: llm={llm_s:.2f}s parse={parse_s:.2f}s total={total_s:.2f}s "
            f"subtopic_id={subtopic_id or '(none)'}"
        )
        if not isinstance(parsed, dict):
            return False, ""

        is_dup = bool(parsed.get("is_duplicate"))
        replacement = str(parsed.get("replacement") or "").strip()
        matches = str(parsed.get("matches") or "").strip()

        # Only act when the judge both flags a duplicate AND produces a usable
        # replacement question. Guards against false positives with no rewrite.
        if is_dup and replacement and "?" in replacement:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] semantic_duplicate flagged: reason={parsed.get('reason', '')!r} "
                f"matches={matches[:120]!r} "
                f"original={proposed_question[:120]!r} "
                f"replacement={replacement[:120]!r}"
            )
            return True, replacement
        return False, ""

    async def _rewrite_question_with_llm(self, question: str, subtopic_id: str = "") -> str:
        """LLM-based question rewrite pass for duplicate-prone or malformed asks."""
        rewrite_start = time.perf_counter()
        q = self._normalize_text(question)
        if not q or "?" not in q:
            return q

        recent_qs = self._get_recent_interviewer_questions(limit=8)
        recent_answers = self._get_recent_user_answers(limit=6)
        last_answer = recent_answers[-1] if recent_answers else ""
        subtopic_context = self._get_subtopic_inference_context(subtopic_id)

        rewrite_prompt = (
            "You are an interview question quality rewriter.\n"
            "Decide whether the candidate question should be rewritten.\n\n"
            "Candidate question:\n"
            f"{q}\n\n"
            f"Current subtopic id: {subtopic_id or '(none)'}\n"
            f"{subtopic_context}\n\n"
            "Recent interviewer questions (oldest -> newest):\n"
            + "\n".join(f"- {x}" for x in recent_qs)
            + "\n\nRecent user answers (oldest -> newest):\n"
            + "\n".join(f"- {x}" for x in recent_answers)
            + "\n\nMost recent user answer:\n"
            + (last_answer or "(none)")
            + "\n\nRules:\n"
              "- Prefer keeping the question unchanged when it is already high-quality.\n"
              "- Preserve the same coverage target as the candidate question's subtopic; do not broaden or narrow coverage scope.\n"
              "- Rewrite when the question is duplicate-prone, catch-all, double-barreled, or asks two numeric dimensions at once.\n"
              "- If the candidate asks for numeric count/duration/percentage but the active subtopic criteria do NOT explicitly require numbers, rewrite it to a qualitative question.\n"
              "- For Task Inventory list-completion turns, do NOT ask headcount/count/duration questions (for example, how many trainees/meetings). Keep focus on cadence/completeness/action+object clarification.\n"
              "- Output question must be exactly one sentence and contain exactly one question mark.\n"
              "- Output must ask for one information target only.\n"
              "- No examples/options lists.\n"
              "- Non-leading.\n"
              "- If rewritten, anchor on a concrete noun from the most recent user answer.\n\n"
              "Return JSON only:\n"
              "{\n"
              "  \"rewrite\": true or false,\n"
              "  \"question\": \"final question text\",\n"
              "  \"reason\": \"short reason\"\n"
              "}\n"
        )

        llm_s = 0.0
        try:
            llm_start = time.perf_counter()
            raw = await self.call_engine_async(rewrite_prompt)
            llm_s = time.perf_counter() - llm_start
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] llm_rewrite_check failed: {e}",
                log_level="warning",
            )
            return q

        parse_start = time.perf_counter()
        parsed = self._extract_json_dict(raw)
        parse_s = time.perf_counter() - parse_start
        total_s = time.perf_counter() - rewrite_start
        SessionLogger.log_to_file(
            "execution_log",
            f"[LATENCY] llm_rewrite_check: llm={llm_s:.2f}s parse={parse_s:.2f}s total={total_s:.2f}s "
            f"subtopic_id={subtopic_id or '(none)'}"
        )
        if not isinstance(parsed, dict):
            return q

        rewrite = bool(parsed.get("rewrite"))
        rewritten = self._normalize_text(parsed.get("question", ""))
        if not rewritten or rewritten.count("?") != 1:
            return q

        if rewrite and rewritten != q:
            SessionLogger.log_to_file(
                "execution_log",
                f"[GUARD] llm_rewrite applied: original={q[:120]!r} rewritten={rewritten[:120]!r}"
            )
        return rewritten

    def _extract_tool_json_dict(self, raw: str) -> dict | None:
        """Best-effort extraction of tool JSON from raw model output.

        Handles:
        - pure JSON object output
        - mixed text + trailing JSON object
        - line-delimited JSON fragments
        - malformed JSON with unescaped quotes in string values
        """
        text = str(raw or "").strip()
        if not text:
            return None

        def _is_tool_payload(obj: object) -> bool:
            return isinstance(obj, dict) and any(
                k in obj for k in ("response", "goodbye", "subtopic_id")
            )

        def _choose_payload(candidates: list[dict]) -> dict | None:
            """Prefer terminal goodbye payloads when multiple tool JSON objects exist."""
            if not candidates:
                return None
            for payload in candidates:
                if "goodbye" in payload:
                    return payload
            return candidates[0]

        candidates: list[dict] = []

        # 1) Full-string parse
        try:
            parsed = json.loads(text)
            if _is_tool_payload(parsed):
                candidates.append(parsed)
        except Exception:
            pass

        # 2) Line-delimited parse (common with leaked JSON on a new line)
        for line in (ln.strip() for ln in text.splitlines() if ln.strip()):
            if not line.startswith("{"):
                continue
            try:
                parsed = json.loads(line)
                if _is_tool_payload(parsed):
                    candidates.append(parsed)
            except Exception:
                continue

        # 3) Scan for any embedded JSON object starting at a '{'
        decoder = json.JSONDecoder()
        for m in re.finditer(r"\{", text):
            snippet = text[m.start():]
            try:
                parsed, _ = decoder.raw_decode(snippet)
            except Exception:
                continue
            if _is_tool_payload(parsed):
                candidates.append(parsed)

        chosen = _choose_payload(candidates)
        if chosen is not None:
            return chosen

        # 4) Regex fallback for malformed JSON values with unescaped quotes
        goodbye_m = re.search(r'"goodbye"\s*:\s*"(.*)"', text, re.DOTALL)
        if goodbye_m:
            return {"goodbye": goodbye_m.group(1).rstrip("}").rstrip()}

        sid_m = re.search(r'"subtopic_id"\s*:\s*"([^"]*)"', text)
        resp_m = re.search(r'"response"\s*:\s*"(.*)"', text, re.DOTALL)
        if resp_m:
            parsed = {"response": resp_m.group(1).rstrip("}").rstrip()}
            if sid_m:
                parsed["subtopic_id"] = sid_m.group(1)
            return parsed

        return None

    def _tool_json_to_xml(self, parsed: dict | None) -> str | None:
        """Convert extracted JSON tool payload into expected XML tool-call format."""
        if not isinstance(parsed, dict):
            return None

        if "goodbye" in parsed:
            goodbye = str(parsed.get("goodbye", ""))
            for ch, esc in [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")]:
                goodbye = goodbye.replace(ch, esc)
            return (
                f"<tool_calls><end_conversation>"
                f"<goodbye>{goodbye}</goodbye>"
                f"</end_conversation></tool_calls>"
            )

        if "response" in parsed:
            subtopic_id = str(parsed.get("subtopic_id", ""))
            resp_text = str(parsed.get("response", ""))
            for ch, esc in [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;")]:
                resp_text = resp_text.replace(ch, esc)
                subtopic_id = subtopic_id.replace(ch, esc)
            return (
                f"<tool_calls><respond_to_user>"
                f"<subtopic_id>{subtopic_id}</subtopic_id>"
                f"<response>{resp_text}</response>"
                f"</respond_to_user></tool_calls>"
            )

        return None

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

        pipeline_start = time.perf_counter()
        sem_gate_start = time.perf_counter()
        should_semantic = self._should_run_semantic_duplicate_llm(
            quantified_question,
            subtopic_id=subtopic_id,
        )
        sem_gate_s = time.perf_counter() - sem_gate_start
        sem_check_s = 0.0

        if should_semantic:
            # Only run the LLM judge when the lightweight risk gate says it is needed.
            try:
                sem_check_start = time.perf_counter()
                is_dup, replacement = await self._check_semantic_duplicate(
                    quantified_question,
                    subtopic_id=subtopic_id,
                )
                sem_check_s = time.perf_counter() - sem_check_start
            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[GUARD] semantic_duplicate_check errored: {e}",
                    log_level="warning",
                )
                is_dup, replacement = False, ""
            if is_dup and replacement:
                quantified_question = replacement
        else:
            SessionLogger.log_to_file(
                "execution_log",
                f"[LATENCY] semantic_duplicate_check skipped (low-risk turn): gate={sem_gate_s:.2f}s"
            )

        infer_gate_start = time.perf_counter()
        should_inferability = self._should_run_inferability_llm(
            quantified_question,
            subtopic_id=subtopic_id,
        )
        infer_gate_s = time.perf_counter() - infer_gate_start
        infer_check_s = 0.0

        if should_inferability:
            try:
                infer_check_start = time.perf_counter()
                inferable, replacement = await self._check_inferable_question(
                    quantified_question,
                    subtopic_id=subtopic_id,
                )
                infer_check_s = time.perf_counter() - infer_check_start
            except Exception as e:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[GUARD] inferability_check errored: {e}",
                    log_level="warning",
                )
                inferable, replacement = False, ""
            if inferable and replacement:
                quantified_question = replacement
        else:
            SessionLogger.log_to_file(
                "execution_log",
                f"[LATENCY] inferability_check skipped (low-risk turn): gate={infer_gate_s:.2f}s"
            )

        local_rewrite_s = 0.0
        rewrite_gate_s = 0.0
        rewrite_s = 0.0
        if self._rewrite_mode != "off":
            local_rewrite_start = time.perf_counter()
            local_rewritten = self._rewrite_question_locally(
                quantified_question,
                subtopic_id=subtopic_id,
            )
            local_rewrite_s = time.perf_counter() - local_rewrite_start
            if local_rewritten != quantified_question:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[GUARD] local_rewrite applied: original={quantified_question[:120]!r} "
                    f"rewritten={local_rewritten[:120]!r}"
                )
                quantified_question = local_rewritten

            rewrite_gate_start = time.perf_counter()
            should_rewrite = self._should_run_llm_rewrite(
                quantified_question,
                subtopic_id=subtopic_id,
            )
            rewrite_gate_s = time.perf_counter() - rewrite_gate_start
            if should_rewrite:
                try:
                    rewrite_start = time.perf_counter()
                    quantified_question = await self._rewrite_question_with_llm(
                        quantified_question,
                        subtopic_id=subtopic_id,
                    )
                    rewrite_s = time.perf_counter() - rewrite_start
                except Exception as e:
                    SessionLogger.log_to_file(
                        "execution_log",
                        f"[GUARD] llm_rewrite_check errored: {e}",
                        log_level="warning",
                    )
            else:
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[LATENCY] llm_rewrite_check skipped (low-risk turn): gate={rewrite_gate_s:.2f}s"
                )
        else:
            SessionLogger.log_to_file(
                "execution_log",
                "[LATENCY] llm_rewrite_check skipped (mode=off)"
            )

        pipeline_s = time.perf_counter() - pipeline_start
        SessionLogger.log_to_file(
            "execution_log",
            f"[LATENCY] response_pipeline: total={pipeline_s:.2f}s "
            f"semantic_gate={sem_gate_s:.2f}s semantic_check={sem_check_s:.2f}s "
            f"infer_gate={infer_gate_s:.2f}s infer_check={infer_check_s:.2f}s "
            f"local_rewrite={local_rewrite_s:.2f}s rewrite_gate={rewrite_gate_s:.2f}s "
            f"rewrite_check={rewrite_s:.2f}s "
            f"subtopic_id={subtopic_id or '(none)'}"
        )

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

        # Emit the profile confirm widget before the first non-first-topic question
        # (intake sessions only; only fires once via the _profile_confirm_widget_sent flag).
        # Coverage can lag briefly while scribe processing finishes, so also trigger
        # when the selected subtopic is already outside the first topic.
        outgoing_topic_index = self._topic_index_for_subtopic(subtopic_id)
        if (getattr(self.interview_session, 'session_type', 'intake') == 'intake'
                and not getattr(self.interview_session, '_profile_confirm_widget_sent', False)
                and (
                    self._is_first_topic_covered()
                    or (outgoing_topic_index is not None and outgoing_topic_index > 0)
                )):
            self.interview_session.trigger_profile_confirm_widget()
            asyncio.create_task(
                self._refresh_portrait_before_widget(
                    widget_context="profile_confirm",
                    max_scribe_wait_s=2.0,
                )
            )

        # Mark that an AI adoption question has been sent so the widget fires
        # after the user's answer (not alongside the question).
        if (
            bool(subtopic_id)
            and not self._ai_question_asked
            and not getattr(self.interview_session, "_ai_usage_widget_sent", False)
            and self._is_ai_adoption_subtopic(subtopic_id)
        ):
            self._ai_question_asked = True

        should_show_time_split_widget = (
            bool(subtopic_id)
            and self._is_time_allocation_subtopic(subtopic_id)
            and not self._time_split_widget_shown
        )
        if should_show_time_split_widget:
            quantified_question = self._ensure_time_split_widget_mention(
                quantified_question
            )

        self.interview_session.add_message_to_chat_history(
            role=self.title,
            content=quantified_question,
            metadata={'subtopic_id': str(subtopic_id), "rubric": rubric},
        )
        self.add_event(sender=self.name, tag="message",
                       content=quantified_question)

        if should_show_time_split_widget:
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
            asyncio.create_task(
                self._refresh_portrait_before_widget(
                    widget_context="time_split",
                    max_scribe_wait_s=30.0,
                )
            )

        return quantified_question

    async def _refresh_portrait_before_widget(
        self,
        widget_context: str = "widget",
        max_scribe_wait_s: float = 30.0,
    ) -> None:
        """Refresh portrait for the widget path without duplicating in-flight work."""
        scribe = getattr(self.interview_session, 'session_scribe', None)
        _scribe_wait_start = asyncio.get_event_loop().time()
        if scribe is not None:
            start = asyncio.get_event_loop().time()
            while getattr(scribe, 'processing_in_progress', False):
                await asyncio.sleep(0.1)
                if asyncio.get_event_loop().time() - start > max(0.0, float(max_scribe_wait_s)):
                    SessionLogger.log_to_file(
                        "execution_log",
                        f"[WIDGET] {widget_context}: timed out waiting for scribe before portrait refresh"
                    )
                    break
        _scribe_wait_s = asyncio.get_event_loop().time() - _scribe_wait_start
        _target_memory_count = len(getattr(self.interview_session.memory_bank, "memories", []))
        _portrait_start = asyncio.get_event_loop().time()
        _portrait_refreshed = False
        try:
            _portrait_refreshed = await self.interview_session.ensure_user_portrait_fresh(
                min_memory_count=_target_memory_count,
                wait_for_inflight=True,
            )
        except Exception as e:
            SessionLogger.log_to_file(
                "execution_log",
                f"[WIDGET] Portrait refresh before time-split widget failed: {e}"
            )
        _portrait_s = asyncio.get_event_loop().time() - _portrait_start
        SessionLogger.log_to_file(
            "execution_log",
            f"[LATENCY] widget_portrait_refresh: context={widget_context}"
            f" scribe_wait={_scribe_wait_s:.2f}s"
            f" portrait_llm={_portrait_s:.2f}s"
            f" refreshed={_portrait_refreshed}"
            f" target_memories={_target_memory_count}"
            f" portrait_memories={getattr(self.interview_session, '_portrait_update_memory_count', 0)}"
        )
    def _is_first_topic_covered(self) -> bool:
        """Return True when every required subtopic in the first topic is covered."""
        topic_manager = self.interview_session.session_agenda.interview_topic_manager
        topics = list(topic_manager)
        if not topics:
            return False
        return all(st.is_covered for st in topics[0].required_subtopics.values())

    def _topic_index_for_subtopic(self, subtopic_id: str) -> int | None:
        """Return 0-based topic index for a subtopic id, or None if unknown."""
        sid = str(subtopic_id or "").strip()
        if not sid:
            return None
        topic_manager = self.interview_session.session_agenda.interview_topic_manager
        for idx, topic in enumerate(topic_manager):
            for st in topic.required_subtopics.values():
                if str(st.subtopic_id) == sid:
                    return idx
        return None

    def _is_subtopic_currently_covered(self, subtopic_id: str) -> bool:
        """Return True if the given subtopic_id is currently marked covered."""
        sid = str(subtopic_id or "").strip()
        if not sid:
            return False
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return False
        for topic in agenda.interview_topic_manager:
            for st in topic.required_subtopics.values():
                if str(st.subtopic_id) == sid:
                    return bool(st.is_covered)
        return False

    def _extract_response_subtopic_id(self, raw_response: str) -> str:
        """Best-effort extraction of target subtopic_id from model output."""
        text = str(raw_response or "")
        if not text:
            return ""

        sid_m = re.search(r"<subtopic_id>\s*(.*?)\s*</subtopic_id>", text, re.DOTALL)
        if sid_m:
            return str(sid_m.group(1) or "").strip()

        parsed = self._extract_tool_json_dict(text)
        if isinstance(parsed, dict):
            sid = str(parsed.get("subtopic_id", "")).strip()
            if sid:
                return sid

        return ""

    def _should_discard_speculative_response(self, raw_response: str) -> bool:
        """Discard speculative response only when it clearly targets covered content."""
        sid = self._extract_response_subtopic_id(raw_response)
        if not sid:
            # If we cannot confidently infer staleness, keep the speculative result
            # to avoid unnecessary second LLM calls.
            return False
        return self._is_subtopic_currently_covered(sid)

    def _is_task_inventory_complete(self) -> bool:
        """Return True when all required subtopics of the Task Inventory topic are covered."""
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return False
        for topic in agenda.interview_topic_manager:
            topic_desc = str(getattr(topic, "description", "")).lower()
            if "task inventory" not in topic_desc:
                continue
            return all(st.is_covered for st in topic.required_subtopics.values())
        return False

    def _is_ai_adoption_subtopic(self, subtopic_id: str) -> bool:
        """Return True when subtopic_id belongs to an AI-related topic."""
        sid = str(subtopic_id or "").strip()
        if not sid:
            return False
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return False
        for topic in agenda.interview_topic_manager:
            topic_desc = str(getattr(topic, "description", "")).lower()
            if "ai" not in topic_desc and "artificial intelligence" not in topic_desc:
                continue
            for st in topic.required_subtopics.values():
                if str(st.subtopic_id) == sid:
                    return True
        return False

    def _is_time_allocation_subtopic(self, subtopic_id: str) -> bool:
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return False
        for topic in agenda.interview_topic_manager:
            for st in topic.required_subtopics.values():
                if str(st.subtopic_id) == str(subtopic_id) and 'time allocation' in st.description.lower():
                    return True
        return False

    def _ensure_time_split_widget_mention(self, question: str) -> str:
        """Ensure first time-allocation prompt explicitly references the widget."""
        q = self._normalize_text(question)
        if not q:
            return (
                "Using the time-allocation widget below, please split your work "
                "time across tasks as percentages that add up to about 100%."
            )
        if "widget" in q.lower():
            return q
        if q.endswith("?"):
            stem = q[:-1].rstrip()
            return f"{stem} using the time-allocation widget below?"
        return f"{q} Please use the time-allocation widget below."

    def _is_fresh_intake_session(self) -> bool:
        """Return True for a brand-new intake session with no prior summary context."""
        if getattr(self.interview_session, "session_type", "intake") != "intake":
            return False
        if self.interview_session.chat_history:
            return False
        summary = (
            self.interview_session.session_agenda.get_last_meeting_summary_str() or ""
        ).strip()
        return len(summary) == 0

    def _get_role_title_subtopic_id(self) -> str:
        """Best-effort lookup for the role/title subtopic so first-turn metadata is preserved."""
        agenda = self.interview_session.session_agenda
        if not agenda or not agenda.interview_topic_manager:
            return ""
        for topic in agenda.interview_topic_manager:
            for st in topic.required_subtopics.values():
                desc = str(getattr(st, "description", "")).lower()
                # Match both legacy wording ("current role or title") and the
                # current config wording ("their role or title").
                if (
                    "current role or title" in desc
                    or "role or title" in desc
                    or ("role" in desc and "title" in desc)
                ):
                    return str(st.subtopic_id)
        return ""


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
        self._last_sent_message_text = ""
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

            # Show the AI task widget as soon as Task Inventory is complete,
            # before the LLM runs, so it fires immediately after the user's "no"
            # to the breadth probe rather than after an LLM goodbye message.
            if (
                not getattr(self.interview_session, "_ai_task_widget_sent", False)
                and not getattr(self.interview_session, "_session_ending", False)
                and self._is_task_inventory_complete()
            ):
                SessionLogger.log_to_file(
                    "execution_log",
                    "[GUARD] Task inventory complete — triggering AI task widget."
                )
                asyncio.create_task(
                    self._refresh_portrait_before_widget(
                        widget_context="ai_task",
                        max_scribe_wait_s=10.0,
                    )
                )
                self.interview_session.trigger_ai_task_widget()
                self._turn_to_respond = False
                return

            # Show the AI usage widget after the user answers the first AI question.
            if (
                self._ai_question_asked
                and not getattr(self.interview_session, "_ai_usage_widget_sent", False)
            ):
                self._ai_question_asked = False
                self.interview_session.trigger_ai_usage_widget()

            # After the completeness probe, either end immediately (closure) or
            # allow one LLM follow-up if the user mentioned a new task.
            # Check flag first; fall back to chat-history inspection (avoids apostrophe issues).
            _probe_active = getattr(self.interview_session, "_completeness_probe_sent", False)
            _chat = getattr(self.interview_session, 'chat_history', [])
            _last_iv = next(
                (m.content for m in reversed(_chat)
                 if getattr(m, 'role', '') == self.title),
                ""
            )
            if not _probe_active:
                # Match on unambiguous substrings that don't contain apostrophes.
                _probe_active = (
                    "any tasks" in str(_last_iv).lower()
                    and "covered" in str(_last_iv).lower()
                )

            SessionLogger.log_to_file(
                "execution_log",
                f"[PROBE-GUARD] probe_active={_probe_active}"
                f" post_pending={getattr(self.interview_session, '_post_probe_followup_pending', False)}"
                f" msg={repr((message.content or '')[:80])}"
            )

            if (
                not getattr(self.interview_session, "_session_ending", False)
                and _probe_active
                and not getattr(self.interview_session, "_post_probe_followup_pending", False)
            ):
                self.interview_session._completeness_probe_sent = False
                _probe_classification = await self._classify_completeness_probe_reply(
                    latest_question=_last_iv,
                    participant_reply=message.content or "",
                )
                _decision = str(
                    _probe_classification.get("decision", "")
                ).strip().lower()
                SessionLogger.log_to_file(
                    "execution_log",
                    f"[PROBE-GUARD] semantic_decision={_decision}"
                    f" reason={_probe_classification.get('reason', '')!r}"
                )
                if _decision == "closure":
                    self._guarded_send_goodbye(self._default_completion_goodbye())
                    if not getattr(self.interview_session, "_feedback_widget_sent", False):
                        self.interview_session.trigger_feedback_widget()
                    self._turn_to_respond = False
                    return
                if _decision == "unclear_response":
                    clarification = str(
                        _probe_classification.get("clarification_question", "")
                    ).strip()
                    if not clarification:
                        clarification = (
                            "Are you adding another task, or are we done with the task list?"
                        )
                    await self._guarded_handle_response(clarification)
                    self._turn_to_respond = False
                    return

                if _decision == "new_task":
                    # User mentioned a new task — allow one LLM turn; suppress the
                    # coverage guard below so it doesn't cut in before the LLM runs.
                    self.interview_session._post_probe_followup_pending = True
                else:
                    self.interview_session._post_probe_followup_pending = True

            elif (
                not getattr(self.interview_session, "_session_ending", False)
                and getattr(self.interview_session, "_post_probe_followup_pending", False)
            ):
                self.interview_session._post_probe_followup_pending = False
                self._guarded_send_goodbye(self._default_completion_goodbye())
                if not getattr(self.interview_session, "_feedback_widget_sent", False):
                    self.interview_session.trigger_feedback_widget()
                self._turn_to_respond = False
                return

        # Opening turn of a fresh session with no prior summary: let LLM generate first question
        is_weekly = getattr(self.interview_session, "session_type", "intake") == "weekly"

        # If session is paused, wait until resumed or a step is requested
        await self.interview_session.wait_if_paused()

        # Fast-path the first turn for fresh intake sessions so the user sees the
        # opening question immediately without waiting for an LLM round-trip.
        if message is None and self._is_fresh_intake_session():
            opening_question = (
                "Hi, thanks for chatting with me. "
                "To start, could you briefly describe your job, what you do for work, and how long you've been in this profession?"
            )
            SessionLogger.log_to_file(
                "execution_log",
                "[LATENCY] Using fast opening question path (no initial LLM call)."
            )
            await self._guarded_handle_response(
                opening_question,
                subtopic_id=self._get_role_title_subtopic_id(),
            )
            self._turn_to_respond = False
            return

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
        _scribe_wait_mode = "none"
        _scribe_wait_exit = "not_needed"
        profile_confirm_pending = (
            getattr(self.interview_session, "session_type", "intake") == "intake"
            and not getattr(self.interview_session, "_profile_confirm_widget_sent", False)
        )

        if message is not None:
            scribe = getattr(self.interview_session, 'session_scribe', None)
            if scribe is None:
                _scribe_wait_mode = "no_scribe"
                _scribe_wait_exit = "no_scribe"
            else:
                # Yield once so SessionScribe can flip its processing flags for this
                # user turn before we inspect them. Without this grace tick, the
                # interviewer can reuse a stale speculative answer.
                await asyncio.sleep(0)
                _scribe_wait_mode = (
                    "coverage_only" if profile_confirm_pending else "full_scribe"
                )
                _last_seen_coverage = pre_scribe_coverage
                _coverage_stable_ticks = 0
                _loop_timed_out = True
                _poll_s = 0.1
                try:
                    _max_wait_s = float(os.getenv("INTERVIEWER_MAX_SCRIBE_WAIT_S", "6.0"))
                except (TypeError, ValueError):
                    _max_wait_s = 6.0
                _max_wait_s = max(0.5, _max_wait_s)
                try:
                    _coverage_stable_required_ticks = int(
                        os.getenv("INTERVIEWER_COVERAGE_STABLE_TICKS", "1")
                    )
                except (TypeError, ValueError):
                    _coverage_stable_required_ticks = 1
                _coverage_stable_required_ticks = max(1, _coverage_stable_required_ticks)
                _max_ticks = max(1, int(_max_wait_s / _poll_s))
                for _ in range(_max_ticks):
                    if profile_confirm_pending:
                        # During the profile-confirm transition we only need
                        # coverage updates to be complete; memory-writing tail
                        # should not block the next interviewer turn.
                        _processing_busy = bool(
                            getattr(scribe, 'processing_in_progress', False)
                        )
                        _coverage_busy = getattr(
                            scribe, 'coverage_processing_in_progress', None
                        )
                        _scribe_busy = (
                            _coverage_busy
                            if isinstance(_coverage_busy, bool)
                            else _processing_busy
                        )
                    else:
                        _scribe_busy = getattr(scribe, 'processing_in_progress', False)
                    if not _scribe_busy:
                        _scribe_wait_exit = "scribe_idle"
                        _loop_timed_out = False
                        break
                    _scribe_waited = True
                    await asyncio.sleep(_poll_s)
                    _elapsed_s = (_ + 1) * _poll_s
                    # Every 5 ticks (0.5s), check if coverage has changed.
                    # If stable for N consecutive checks and the speculative
                    # result is ready, we can proceed without waiting for the rest
                    # of the scribe's work (memory writes, portrait update).
                    if (
                        _ % 5 == 4
                        and speculative_task is not None
                        and not profile_confirm_pending
                    ):
                        curr = self._get_coverage_snapshot()
                        if curr == _last_seen_coverage:
                            _coverage_stable_ticks += 1
                            if (
                                _coverage_stable_ticks >= _coverage_stable_required_ticks
                                and speculative_task.done()
                            ):
                                SessionLogger.log_to_file(
                                    "execution_log",
                                    "[LATENCY] early-exit scribe wait: coverage stable "
                                    f"{0.5 * _coverage_stable_required_ticks:.1f}s"
                                )
                                _scribe_wait_exit = "coverage_stable_early_exit"
                                _loop_timed_out = False
                                break
                        else:
                            _coverage_stable_ticks = 0
                            _last_seen_coverage = curr
                if _loop_timed_out and _scribe_wait_exit == "not_needed":
                    _scribe_wait_exit = "max_wait_reached"
        _scribe_wait_s = asyncio.get_event_loop().time() - _scribe_wait_start

        # If the scribe updated coverage, the speculative prompt is stale — discard it
        # only when it clearly targets already-covered content.
        _speculative_discarded = False
        if message is not None and speculative_task is not None:
            post_scribe_coverage = self._get_coverage_snapshot()
            if post_scribe_coverage != pre_scribe_coverage:
                _discard = False
                if speculative_task.done():
                    try:
                        _discard = self._should_discard_speculative_response(
                            speculative_task.result()
                        )
                    except Exception:
                        # On parsing/runtime errors, keep previous conservative behavior.
                        _discard = True
                if _discard:
                    if not speculative_task.done():
                        speculative_task.cancel()
                    speculative_task = None
                    speculative_prompt = None
                    _speculative_discarded = True

        if message is not None:
            SessionLogger.log_to_file(
                "execution_log",
                f"[LATENCY] scribe_wait={_scribe_wait_s:.2f}s"
                f" gate={_scribe_wait_mode}"
                f" exit={_scribe_wait_exit}"
                f" speculative_discarded={_speculative_discarded}"
            )

        # After scribe has processed, check whether task inventory just became
        # complete — cancel the speculative response and show the AI task widget.
        if (
            message is not None
            and not getattr(self.interview_session, "_session_ending", False)
            and not getattr(self.interview_session, "_ai_task_widget_sent", False)
            and self._is_task_inventory_complete()
        ):
            if speculative_task is not None and not speculative_task.done():
                speculative_task.cancel()
            speculative_task = None
            SessionLogger.log_to_file(
                "execution_log",
                "[GUARD] Task inventory complete after scribe — triggering AI task widget."
            )
            asyncio.create_task(
                self._refresh_portrait_before_widget(
                    widget_context="ai_task",
                    max_scribe_wait_s=10.0,
                )
            )
            self.interview_session.trigger_ai_task_widget()
            self._turn_to_respond = False

        # Deterministic stop: if all configured topics are already covered after
        # scribe processing, do not let the LLM improvise another follow-up.
        # Skip when _post_probe_followup_pending so the LLM can answer the new task.
        if (
            message is not None
            and not getattr(self.interview_session, "_session_ending", False)
            and not getattr(self.interview_session, "_post_probe_followup_pending", False)
            and self._no_active_topics_remaining()
        ):
            if speculative_task is not None and not speculative_task.done():
                speculative_task.cancel()
            # First time all topics are covered: show task validation widget.
            # After widget is submitted the feedback widget is triggered server-side.
            if not getattr(self.interview_session, "_task_validation_widget_sent", False):
                SessionLogger.log_to_file(
                    "execution_log",
                    "[GUARD] No active topics remain — triggering task validation widget."
                )
                self.interview_session.add_message_to_chat_history(
                    role="Interviewer",
                    content=(
                        "Based on what you've told me, here are some tasks I think you might be doing right now. Do these sound right?\n\n<span style='font-size:0.82em;opacity:0.55;font-style:italic;'>Note: these are existing tasks, NOT tasks you might do in the future, or tasks that are just typical for your role in general.</span>"
                    ),
                )
                asyncio.create_task(
                    self._refresh_portrait_before_widget(
                        widget_context="task_validation",
                        max_scribe_wait_s=10.0,
                    )
                )
                self.interview_session.trigger_task_validation_widget()
            else:
                SessionLogger.log_to_file(
                    "execution_log",
                    "[GUARD] No active topics remain after task validation — ending session."
                )
                self._guarded_send_goodbye(self._default_completion_goodbye())
                if not getattr(self.interview_session, "_feedback_widget_sent", False):
                    self.interview_session.trigger_feedback_widget()
            self._turn_to_respond = False
            return

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

            # Re-check after awaiting the LLM: the scribe may have finished
            # committing coverage while the LLM was running.  If all topics are
            # now done, discard the LLM response and go straight to the widget.
            if (
                message is not None
                and not getattr(self.interview_session, "_session_ending", False)
                and self._no_active_topics_remaining()
                and not getattr(self.interview_session, "_task_validation_widget_sent", False)
            ):
                SessionLogger.log_to_file(
                    "execution_log",
                    "[GUARD] Topics completed during LLM wait — discarding response, triggering task widget."
                )
                self.interview_session.add_message_to_chat_history(
                    role="Interviewer",
                    content=(
                        "Okay, here's what I've got so far. Take a look and let me know if anything's off."
                    ),
                )
                asyncio.create_task(
                    self._refresh_portrait_before_widget(
                        widget_context="task_validation",
                        max_scribe_wait_s=10.0,
                    )
                )
                self.interview_session.trigger_task_validation_widget()
                self._turn_to_respond = False
                break

            self.add_event(sender=self.name, tag="llm_prompt", content=prompt)
            print(f"{GREEN}Interviewer:\n{response}{RESET}")
            try:
                # gpt-5.x may return JSON instead of XML — convert it to the
                # expected XML tool-call format so the rest of the pipeline works.
                if "<tool_calls>" not in response:
                    parsed = self._extract_tool_json_dict(response)
                    converted = self._tool_json_to_xml(parsed)
                    if converted:
                        response = converted
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
                    else:
                        parsed = self._extract_tool_json_dict(response.strip())
                        if isinstance(parsed, dict):
                            if "goodbye" in parsed:
                                fallback = str(parsed["goodbye"])
                            elif "response" in parsed:
                                fallback = str(parsed["response"])
                    await self._guarded_handle_response(fallback)

            iterations += 1
            if iterations >= self._max_consideration_iterations:
                self.add_event(
                    sender="system",
                    tag="error",
                    content=f"Exceeded maximum number of consideration "
                    f"iterations ({self._max_consideration_iterations})"
                )

        # Safety net: if all configured topics are covered but end_conversation was
        # not called (e.g. LLM used respond_to_user for the goodbye), trigger the
        # feedback widget so the session ends correctly.
        if (self._message_sent_this_turn
                and not getattr(self.interview_session, '_session_ending', False)
                and not self._last_sent_message_is_question()
                and not getattr(self.interview_session, '_feedback_widget_sent', False)):
            topic_manager = self.interview_session.session_agenda.interview_topic_manager
            incomplete_topics = topic_manager.get_all_incomplete_core_topic()
            allows_emergent = topic_manager.any_active_topic_allows_emergent()
            topics_complete = not incomplete_topics and not allows_emergent
            closing_intent = await self._looks_like_closing_goodbye(
                self._last_sent_message_text
            )
            if topics_complete or closing_intent:
                reason = (
                    "all topics covered"
                    if topics_complete
                    else "closing intent detected in respond_to_user message"
                )
                SessionLogger.log_to_file(
                    "execution_log",
                    "(Interviewer) end_conversation not called — triggering "
                    f"feedback widget as safety net ({reason}).",
                    log_level="warning"
                )
                self.interview_session.trigger_feedback_widget()

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

    def _get_recent_user_answers(self, limit: int = 30) -> list[str]:
        """Return recent non-empty user messages (plain text only)."""
        answers = [
            str(event.content).strip()
            for event in self.event_stream
            if event.sender == "User"
            and event.tag == "message"
            and str(event.content).strip()
        ]
        if limit <= 0:
            return answers
        return answers[-limit:]

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

        event_window = min(self._max_events_len, self._prompt_chat_events_limit)
        recent_events = chat_history_events[-event_window:] if \
            len(chat_history_events) > event_window else chat_history_events
        current_events = recent_events[-2:] if len(recent_events) >= 2 else recent_events

        all_interviewer_messages = self.get_event_stream_str(
            [{"sender": "Interviewer", "tag": "message"}],
            as_list=True
        )
        recent_interviewer_messages = all_interviewer_messages[
            -self._prompt_recent_interviewer_limit:
        ] if len(all_interviewer_messages) >= self._prompt_recent_interviewer_limit else all_interviewer_messages
        recent_user_answers = self._get_recent_user_answers(
            limit=self._prompt_recent_user_answers_limit
        )
        recent_user_answers_str = (
            "\n".join(f"- {msg}" for msg in recent_user_answers)
            if recent_user_answers else "(none yet)"
        )

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
            "recent_user_answers": recent_user_answers_str,
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
