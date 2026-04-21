import os
import re
import json
from collections import deque
from difflib import SequenceMatcher
from typing import Deque, Dict, List, Optional, Tuple
from src.agents.base_agent import BaseAgent
from src.interview_session.user.user import User
from src.interview_session.session_models import Message
from src.interview_session.session_models import MessageType
from src.content.session_agenda.session_agenda import SessionAgenda
from src.utils.constants.colors import ORANGE, RESET
from src.utils.transcript_task_derivation import load_chat_history_messages



class UserAgent(BaseAgent, User):
    def __init__(self, user_id: str, interview_session, config: dict = None,
                 hesitancy: float = 0.0, reuse_similar_answers: bool = False,
                 similar_question_threshold: float = 0.85,
                 similar_answer_cache_size: int = 200,
                 anchor_to_original_transcript: bool = False,
                 transcript_anchor_max_pairs: int = 200):
        config = dict(config or {})
        config["model_name"] = os.getenv("USER_MODEL_NAME", os.getenv("MODEL_NAME", "gpt-4.1-mini"))
        config["base_url"] = os.getenv("USER_VLLM_BASE_URL", None)
        BaseAgent.__init__(
            self, name="UserAgent",
            description="Agent that plays the role of the user", config=config)
        User.__init__(self, user_id=user_id,
                      interview_session=interview_session)
        # hesitancy ∈ [0.0, 1.0]: 0 = fully cooperative, 1 = fully withholding
        self.hesitancy = max(0.0, min(1.0, hesitancy))
        self.reuse_similar_answers = bool(reuse_similar_answers)
        self.similar_question_threshold = max(
            0.0, min(1.0, float(similar_question_threshold))
        )
        cache_size = max(1, int(similar_answer_cache_size))
        self._qa_history: Deque[Dict[str, str]] = deque(maxlen=cache_size)
        self.anchor_to_original_transcript = bool(anchor_to_original_transcript)
        self.transcript_anchor_max_pairs = max(1, int(transcript_anchor_max_pairs))

        # Load profile background (flat bio notes)
        profile_path = os.path.join(
            os.getenv("USER_AGENT_PROFILES_DIR"), f"{user_id}/{user_id}_bio_notes.md")
        if os.path.exists(profile_path):
            with open(profile_path, 'r') as f:
                self.profile_background = f.read()
        else:
            self.profile_background = ""

        # Load structured ground truth (topics_filled.json) if available
        structured_path = os.path.join(
            os.getenv("USER_AGENT_PROFILES_DIR"), f"{user_id}/{user_id}_topics_filled.json")
        if os.path.exists(structured_path):
            with open(structured_path, 'r') as f:
                from src.agents.user.prompts import format_structured_profile
                self.structured_profile = format_structured_profile(json.load(f))
        else:
            self.structured_profile = ""
        
        # Load topics and advance to next topic
        # topics_path = os.path.join(
        #     os.getenv("USER_AGENT_PROFILES_DIR"), f"{user_id}/topics.json")
        # if not os.path.exists(topics_path):
        #     raise ValueError(
        #         f"Topics file not found: {topics_path}\n"
        #         f"Please run: python src/utils/topic_extractor.py --user_id {user_id}"
        #     )
            
        # with open(topics_path, 'r') as f:
        #     topics_data = json.load(f)
        #     topics = topics_data["topics"]
            
        #     # Set the topic for this session
        #     current_topic_index = self.interview_session.session_id - 1
        #     self.current_topic = topics[current_topic_index]

        #     SessionLogger.log_to_file(
        #         "execution_log",
        #         f"Current topic of {user_id}: {self.current_topic}"
        #     )
        
        # Get historical session summaries
        self.session_history = \
            SessionAgenda.get_historical_session_summaries(user_id)

        # Load conversational style
        conv_style_path = os.path.join(
            os.getenv("USER_AGENT_PROFILES_DIR"), f"{user_id}/conversation.md")
        if os.path.exists(conv_style_path):
            with open(conv_style_path, 'r') as f:
                self.conversational_style = f.read()
        else:
            self.conversational_style = ""

        if self.anchor_to_original_transcript:
            seeded = self._seed_from_original_transcript()
            if seeded > 0:
                print(
                    f"{ORANGE}[UserAgent] Seeded {seeded} anchored Q/A pairs from "
                    f"original transcript.{RESET}"
                )

    async def on_message(self, message: Message):
        """Handle incoming messages by generating a response and notifying
        the interview session"""
        if not message:
            return

        # Add the interviewer's message to our event stream
        self.add_event(sender=message.role, tag="message",
                       content=message.content)

        # If session is paused, wait until resumed or a step is requested
        await self.interview_session.wait_if_paused()

        # Score the interviewer's question for potential feedback
        # if os.getenv("EVAL_MODE") == "true":
        #     score_prompt = self._get_prompt(prompt_type="score_question")
        #     self.add_event(sender=self.name,
        #                tag="score_question_prompt", content=score_prompt)

        #     score_response = await self.call_engine_async(score_prompt)
        #     self.add_event(sender=self.name,
        #                  tag="score_question_response", content=score_response)

        #     # Extract the score and reasoning
        #     self.question_score, self.question_score_reasoning = \
        #         self._extract_response(score_response)

        question = str(message.content or "").strip()
        reused = self._find_similar_answer(question)
        if reused is not None:
            response, similarity, matched_question = reused
            self.add_event(
                sender=self.name,
                tag="respond_to_question_reuse",
                content=(
                    f"Matched prior question (similarity={similarity:.3f}):\n"
                    f"{matched_question}"
                ),
            )
            self.add_event(
                sender=self.name,
                tag="respond_to_question_response",
                content=response,
            )
        else:
            prompt = self._get_prompt(prompt_type="respond_to_question")
            self.add_event(sender=self.name,
                           tag="respond_to_question_prompt", content=prompt)

            response = await self.call_engine_async(prompt)
            self.add_event(sender=self.name,
                           tag="respond_to_question_response", content=response)

        if question and response:
            self._qa_history.append({"question": question, "answer": response})

        self.add_event(sender=self.name,
                       tag="message", content=response)

        # No artificial delay — respond immediately
        print(f"{ORANGE}User Agent:\n{response}{RESET}")

        self.interview_session.add_message_to_chat_history(
            role=self.title, content=response, message_type=MessageType.CONVERSATION)

        # # Extract the response content and reasoning
        # response_content, response_reasoning = self._extract_response(response)
        # wants_to_respond = response_content != "SKIP"

        # if wants_to_respond:
        #     # Generate detailed response using LLM

        #     # Extract just the <response> content to send to chat history
        #     self.add_event(sender=self.name, tag="message",
        #                    content=response_content)
        #     self.interview_session.add_message_to_chat_history(
        #         role=self.title, content=response_reasoning, 
        #             message_type=MessageType.FEEDBACK)
        #     self.interview_session.add_message_to_chat_history(
        #         role=self.title, content=response_content, 
        #             message_type=MessageType.CONVERSATION)

        # else:
        #     # We SKIP the response and log a feedback message
        #     self.interview_session.add_message_to_chat_history(
        #         role=self.title, content=response_reasoning, 
        #             message_type=MessageType.FEEDBACK)
        #     self.interview_session.add_message_to_chat_history(
        #         role=self.title, message_type=MessageType.SKIP)

    def _get_prompt(self, prompt_type: str) -> str:
        """Get the formatted prompt for the LLM"""
        from src.agents.user.prompts import get_prompt

        common = dict(
            profile_background=self.profile_background,
            structured_profile=self.structured_profile,
            conversational_style=self.conversational_style,
            session_history=self.session_history,
            hesitancy=self.hesitancy,
        )

        if prompt_type == "score_question":
            return get_prompt(prompt_type, **common).format(
                chat_history=self.get_event_stream_str([{"tag": "message"}])
            )
        elif prompt_type == "respond_to_question":
            chat_history = self.get_event_stream_str([{"tag": "message"}])
            if "<UserAgent>" not in chat_history:  # first turn → introduction
                return get_prompt("introduction", **common).format(
                    chat_history=chat_history
                )
            else:
                return get_prompt(prompt_type, **common).format(
                    chat_history=chat_history
                )

    def _extract_response(self, full_response: str) -> tuple[str, str]:
        """Extract the content between <response_content> and <thinking> tags"""
        response_match = re.search(
            r'<response_content>(.*?)</response_content>', full_response, re.DOTALL)
        thinking_match = re.search(
            r'<thinking>(.*?)</thinking>', full_response, re.DOTALL)

        response = response_match.group(
            1).strip() if response_match else full_response
        thinking = thinking_match.group(1).strip() if thinking_match else ""
        return response, thinking

    @staticmethod
    def _normalize_question(question: str) -> str:
        normalized = question.lower().replace("'", "").replace("’", "")
        normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    @staticmethod
    def _token_jaccard(a: str, b: str) -> float:
        a_tokens = set(a.split())
        b_tokens = set(b.split())
        if not a_tokens or not b_tokens:
            return 0.0
        return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)

    def _question_similarity(self, question_a: str, question_b: str) -> float:
        a = self._normalize_question(question_a)
        b = self._normalize_question(question_b)
        if not a or not b:
            return 0.0
        if a == b:
            return 1.0

        seq_ratio = SequenceMatcher(None, a, b).ratio()
        jaccard = self._token_jaccard(a, b)
        return max(seq_ratio, jaccard)

    def _find_similar_answer(
        self, question: str
    ) -> Optional[Tuple[str, float, str]]:
        if not self.reuse_similar_answers or not question:
            return None

        best_match = None
        best_score = -1.0
        for qa in reversed(self._qa_history):
            prior_question = qa.get("question", "")
            prior_answer = qa.get("answer", "")
            if not prior_question or not prior_answer:
                continue
            score = self._question_similarity(question, prior_question)
            if score >= self.similar_question_threshold and score > best_score:
                best_match = (prior_answer, score, prior_question)
                best_score = score

        return best_match

    @staticmethod
    def _extract_interviewer_user_pairs(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Build interviewer->user Q/A pairs from parsed chat history messages."""
        pairs: List[Dict[str, str]] = []
        current_question = ""
        current_answers: List[str] = []

        def _flush() -> None:
            nonlocal current_question, current_answers
            if not current_question or not current_answers:
                return
            answer_text = "\n".join(a.strip() for a in current_answers if a.strip()).strip()
            if answer_text:
                pairs.append({"question": current_question.strip(), "answer": answer_text})
            current_answers = []

        for msg in messages:
            speaker = str(msg.get("speaker", "")).strip()
            content = str(msg.get("content", "")).strip()
            if not content:
                continue
            if speaker == "Interviewer":
                _flush()
                current_question = content
            elif speaker == "User" and current_question:
                current_answers.append(content)

        _flush()
        return pairs

    def _load_original_transcript_qa_pairs(self) -> List[Dict[str, str]]:
        uid = getattr(self, "_user_id", getattr(self, "user_id", ""))
        if not uid:
            return []
        profiles_dir = os.getenv("USER_AGENT_PROFILES_DIR", "data/sample_user_profiles")
        artifact_path = os.path.join(
            profiles_dir, uid, f"{uid}_derived_tasks_from_transcript.json"
        )
        if not os.path.exists(artifact_path):
            return []

        try:
            with open(artifact_path, "r", encoding="utf-8") as f:
                artifact = json.load(f)
        except Exception:
            return []

        log_path = str(artifact.get("source_chat_history_log") or "").strip()
        if not log_path:
            return []
        if not os.path.exists(log_path):
            return []

        try:
            messages = load_chat_history_messages(log_path)
        except Exception:
            return []

        return self._extract_interviewer_user_pairs(messages)

    def _seed_from_original_transcript(self) -> int:
        """Seed in-memory Q/A cache with pairs from source pilot transcript."""
        pairs = self._load_original_transcript_qa_pairs()
        if not pairs:
            return 0

        seeded = 0
        for qa in pairs[: self.transcript_anchor_max_pairs]:
            question = str(qa.get("question", "")).strip()
            answer = str(qa.get("answer", "")).strip()
            if not question or not answer:
                continue
            self._qa_history.append({"question": question, "answer": answer})
            seeded += 1
        return seeded
