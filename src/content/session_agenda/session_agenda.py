from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
from pydantic import BaseModel
import os
import json

from src.content.question_bank.question import Question, InterviewQuestion, Rubric
from src.content.session_agenda.interview_topic_manager import InterviewTopicManager
from src.utils.logger.session_logger import SessionLogger
from src.utils.data_process import safe_parse_json


LOGS_DIR = os.getenv("LOGS_DIR")


def normalize_user_portrait(portrait: Any) -> dict:
    """Remove portrait keys dropped from the template (e.g. legacy ``Skills``)."""
    if not isinstance(portrait, dict):
        return {}
    if "Skills" not in portrait:
        return dict(portrait)
    return {k: v for k, v in portrait.items() if k != "Skills"}

class SessionAgenda:
    
    def __init__(self, user_id, session_id, data: dict=None):
        self.user_id = user_id
        self.session_id = int(session_id)
        self.user_portrait: dict = normalize_user_portrait(data.get("user_portrait", {}))
        self.last_meeting_summary: str = data.get("last_meeting_summary", "")
        self.interview_description: str = data.get("interview_description", "")
        self.additional_notes: list[str] = data.get("additional_notes", [])
        raw_manager = data.get("interview_topic_manager")
        if raw_manager is None:
            self.interview_topic_manager = InterviewTopicManager()
        elif isinstance(raw_manager, dict):
            self.interview_topic_manager = InterviewTopicManager.from_dict(raw_manager)
        else:
            self.interview_topic_manager = raw_manager
        self.current_snapshot: int = 0

        # Strategic planning data (updated by StrategicPlanner)
        self.strategic_priorities: Dict[str, dict] = data.get("strategic_priorities", {})
        self.emergent_insights: List[dict] = data.get("emergent_insights", [])

        # Participant's available time for this session (set at session start)
        self.available_time_minutes: Optional[int] = data.get("available_time_minutes")

        # How the session ended: "completed", "timeout", "user_ended"
        self.end_reason: str = data.get("end_reason", "completed")

    @classmethod
    def load_from_file(cls, file_path):
        """Loads a SessionAgenda from a JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Extract the core fields from the file
        user_id = data.pop('user_id', '')
        session_id = data.pop('session_id', '')
        
        # Create new SessionAgenda instance
        return cls(user_id, session_id, data)

    @classmethod
    def initialize_session_agenda(cls, user_id: str, initial_user_portrait_path: str,
                                  interview_plan_path: str,
                                  interview_description: Optional[str] = "",
                                  interview_evaluation: Optional[str] = None):
        """Creates a new session agenda for the first session."""
        # Initialize user portrait
        user_portrait = {}
        if initial_user_portrait_path and os.path.exists(initial_user_portrait_path):
            with open(initial_user_portrait_path, 'r', encoding='utf-8') as f:
                user_portrait = normalize_user_portrait(json.load(f))
        
        session_id = 0
        data = {
            "user_portrait": user_portrait,
            "interview_description": interview_description,
            "last_meeting_summary": "",
            "interview_topic_manager": {},
            "additional_notes": []
        }
        
        # Initialize CoreTopic(s) from the template
        interview_topic_manager = InterviewTopicManager()
        if interview_plan_path and os.path.exists(interview_plan_path):
            with open(interview_plan_path, 'r', encoding='utf-8') as f:
                interview_plan = json.load(f)
            
            interview_topic_manager = InterviewTopicManager.init_from_interview_plan(interview_plan,
                                                                                     interview_evaluator=interview_evaluation)
            
            # TODO Refactor
            if float(os.environ.get("STRATEGIC_PLANNER_GAMMA", 0.0)) > 0:
                interview_topic_manager.use_emergent_subtopics()
        
        data["interview_topic_manager"] = interview_topic_manager

        return cls(user_id, session_id, data)
    
    @classmethod
    def get_last_session_agenda(cls, user_id: str, existing_user_profile_path: Optional[str] = None,
                                initial_user_portrait_path: Optional[str] = None,
                                interview_plan_path: Optional[str] = None,
                                interview_description: Optional[str] = None,
                                interview_evaluation: Optional[str] = None):
        """Retrieves the last session agenda for a user."""
        base_path = os.path.join(LOGS_DIR, user_id, "execution_logs")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        # Look for session directories instead of files
        session_dirs = [d for d in os.listdir(base_path) \
                       if d.startswith('session_') and \
                       os.path.isdir(os.path.join(base_path, d))]
        
        # No session dirs found — still load saved portrait if it exists
        if not session_dirs:
            fresh = cls.initialize_session_agenda(user_id=user_id,
                                                  initial_user_portrait_path=initial_user_portrait_path,
                                                  interview_plan_path=interview_plan_path,
                                                  interview_description=interview_description,
                                                  interview_evaluation=interview_evaluation)
            saved_portrait_path = os.path.join(LOGS_DIR, user_id, "user_portrait.json")
            if os.path.exists(saved_portrait_path):
                try:
                    with open(saved_portrait_path, "r", encoding="utf-8") as f:
                        saved_portrait = normalize_user_portrait(json.load(f))
                    if saved_portrait:
                        fresh.user_portrait = saved_portrait
                except Exception:
                    pass
            return fresh
        
        # Sort by session number
        session_dirs.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
        latest_dir = os.path.join(base_path, session_dirs[0])
        latest_file = os.path.join(latest_dir, "session_agenda.json")
        
        if os.path.exists(latest_file):
            prior = cls.load_from_file(latest_file)
            # Always re-initialize topics from the current config so changes take effect.
            # Preserve cross-session state (user portrait, last meeting summary) from the prior agenda.
            fresh = cls.initialize_session_agenda(user_id=user_id,
                                                  initial_user_portrait_path=initial_user_portrait_path,
                                                  interview_plan_path=interview_plan_path,
                                                  interview_description=interview_description,
                                                  interview_evaluation=interview_evaluation)
            fresh.user_portrait = normalize_user_portrait(prior.user_portrait)
            fresh.last_meeting_summary = prior.last_meeting_summary
            fresh.session_id = prior.session_id
            return fresh

        # No session_agenda.json in the latest dir: fall back to fresh agenda,
        # but preserve the session_id inferred from the directory name so the
        # counter keeps advancing instead of resetting to 0 every restart.
        inferred_session_id = int(session_dirs[0].split('_')[1])
        import logging as _logging
        _logging.getLogger(__name__).warning(
            f"[SESSION_AGENDA] No session_agenda.json found in {latest_dir}. "
            f"Inferring session_id={inferred_session_id} from directory name."
        )
        fresh = cls.initialize_session_agenda(user_id=user_id,
                                              initial_user_portrait_path=initial_user_portrait_path,
                                              interview_plan_path=interview_plan_path,
                                              interview_description=interview_description,
                                              interview_evaluation=interview_evaluation)
        fresh.session_id = inferred_session_id
        saved_portrait_path = os.path.join(LOGS_DIR, user_id, "user_portrait.json")
        if os.path.exists(saved_portrait_path):
            try:
                with open(saved_portrait_path, "r", encoding="utf-8") as f:
                    saved_portrait = normalize_user_portrait(json.load(f))
                if saved_portrait:
                    fresh.user_portrait = saved_portrait
            except Exception:
                pass
        return fresh

    def add_interview_question_raw(self, subtopic_id: str, question: str,
                                   rubric: Optional[str] = None) -> bool:
        """Adds a new interview question to the session agenda.
        
        Args:
            subtopic_id: The subtopic identifier
            question: The actual question text
            rubric: Optional rubric for the question
        """
        question_id = Question.generate_question_id()
        
        final_rubric = rubric
        if rubric is not None and rubric != "":
            final_rubric = Rubric(**json.loads(rubric))
        new_question = InterviewQuestion(subtopic_id=subtopic_id,
                                        question_id=question_id, question=question,
                                        rubric=final_rubric)
        status = self.interview_topic_manager.add_question(subtopic_id=subtopic_id,
                                                           new_question=new_question)
        
        return status
        
    def add_interview_question(self, question: Question) -> bool:
        """Adds a new interview question to the session agenda.
        
        Args:
            question: a Question object
        """
        new_question = InterviewQuestion(subtopic_id=question.subtopic_id,
                                        question_id=question.id, question=question.content,
                                        rubric=question.rubric)
        status = self.interview_topic_manager.add_question(subtopic_id=question.subtopic_id,
                                                            new_question=new_question)
        
        return status
    
    # def delete_interview_question(self, topic_id: str, question_id: str):
    #     """Deletes a question by its ID.
        
    #     If the question has sub-questions:
    #     - Clears the question text and notes
    #     - Keeps the question ID and sub-questions
        
    #     If the question has no sub-questions:
    #     - Removes the question completely
        
    #     Args:
    #         question_id: The ID of the question to delete 
    #         (e.g. "1", "1.1", "2.3")
    #     Raises:
    #         ValueError: If question_id or parent is not found
    #     """
    #     # If it's a sub-question, verify parent exists first
        # if '.' in question_id:
        #     parent_id = question_id.rsplit('.', 1)[0]
        #     parent = self.get_question(topic_id, parent_id)
        #     if not parent:
        #         raise ValueError(f"Parent question with id "
        #                          f"{parent_id} not found")
        
        # # Then check if the question exists
        # question = self.get_question(topic_id, question_id)
        # if not question:
        #     raise ValueError(f"Question with id {question_id} not found")
        
        # # If it's a top-level question
        # if '.' not in question_id:
        #     topic_id = question_id.split("-")[0]

        #     if topic_id not in self.interview_topic_manager:
        #         raise ValueError(f"Topic for question {question_id} not found")
            
        #     # If it has sub-questions, clear content but keep structure
        #     if question.sub_questions:
        #         question.question = ""
        #         question.notes = []
        #     else:
        #         # No sub-questions, remove completely
        #         self.core_topic_tracker_dict[topic_id].questions = [
        #             q for q in self.core_topic_tracker_dict[topic_id] if q.question_id != question_id
        #         ]
            
        # # If it's a sub-question
        # else:
        #     # If it has sub-questions, clear content but keep structure
        #     if question.sub_questions:
        #         question.question = ""
        #         question.notes = []
        #     else:
        #         # No sub-questions, remove completely
        #         def remove_question(questions, target_id):
        #             return [q for q in questions if q.question_id != target_id]
                
        #         parent.sub_questions = remove_question(parent.sub_questions, question_id)
        
    def add_note(self, subtopic_id: str="", note: str=""):
        """Adds a note to a question or the additional notes list."""
        if note:
            status = self.interview_topic_manager.add_note_to_subtopic(subtopic_id, note)
            if not status:
                self.additional_notes.append(note)
        
    def get_question(self, topic_id: str, subtopic_id: str, question_id: str):
        """Retrieves an InterviewQuestion object by its ID."""
        return self.interview_topic_manager.get_question(topic_id, subtopic_id, question_id)

    def save(self, save_type: str="original"):
        """Saves the SessionAgenda to a JSON file.
        
        Args:
            save_type: How to save the note:
                - "original": Save as session_agenda.json in session_X directory
                - "updated": Save as session_agenda_updated.json in same directory
                - "next_version": Save as session_agenda.json in next session directory
                - "snapshot": Save as session_agenda_{self.current_snapshot}.json in same directory for bookkeeping/debugging
        """
        # Create path to session directory
        base_path = os.path.join(LOGS_DIR, self.user_id, "execution_logs")
        file_name = "session_agenda.json"
        save_session_id = self.session_id
        
        if save_type == "original":
            pass
        elif save_type == "snapshot":
            file_name = f"session_agenda_snap_{self.current_snapshot}.json"
            self.current_snapshot += 1
        elif save_type == "updated":
            file_name = "session_agenda_updated.json"
        elif save_type == "next_version":
            save_session_id += 1
        else:
            raise ValueError("save_type must be 'updated', 'original', 'snapshot', or 'next_version'")
        
        session_dir = os.path.join(base_path, f"session_{save_session_id}")
        
        # Create directories if they don't exist
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
            
        file_path = os.path.join(session_dir, file_name)
        
        # Prepare data for serialization
        data = {
            "user_id": self.user_id,
            "session_id": save_session_id,
            "user_portrait": self.user_portrait,
            "last_meeting_summary": self.last_meeting_summary,
            "interview_description": self.interview_description,
            "additional_notes": self.additional_notes,
            "interview_topic_manager": self.interview_topic_manager.to_dict(),
            "strategic_priorities": self.strategic_priorities,
            "emergent_insights": self.emergent_insights,
            "end_reason": self.end_reason,
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return file_path
    
    def all_core_topics_completed(self) -> bool:
        """Checks if all core topics have been completed."""
        return self.interview_topic_manager.check_all_core_topic_completion()
    
    def get_all_uncompleted_core_topics(self, return_ids: bool=False) -> List[str]:
        """Get all uncompleted core topics."""
        core_topic_list = self.interview_topic_manager.get_all_incomplete_core_topic()
        if return_ids:
            return [topic.id for topic in core_topic_list]
        else:
            return core_topic_list

    def get_user_portrait_str(self) -> str:
        """Returns formatted string of user portrait information."""
        if not self.user_portrait:
            return ""
            
        output = []
        for key, value in self.user_portrait.items():
            output.append(f"{key.replace('_', ' ').title()}: {value}")
        return "\n".join(output)

    def get_last_meeting_summary_str(self) -> str:
        """Returns a formatted string representation of the session agenda."""
        if not self.last_meeting_summary:
            return ""
        return self.last_meeting_summary

    def update_user_portrait_str(self, new_user_portrait: str):
        parsed_user_portrait = safe_parse_json(new_user_portrait)
        if parsed_user_portrait:
            # Only update if it's successful parsing it
            self.user_portrait = normalize_user_portrait(parsed_user_portrait)
        
    def update_last_meeting_summary_str(self, new_last_meeting_summary: str):
        self.last_meeting_summary = new_last_meeting_summary
    
    def format_qa(self, qa: InterviewQuestion, hide_answered: str = "", indent: int = 2) -> list[str]:
        """Formats a question and its sub-questions recursively in a structured, LLM-friendly way."""
        if hide_answered not in ["", "a", "qa"]:
            raise ValueError('hide_answered must be "", "a", or "qa"')

        lines = []
        prefix = " " * indent  # uniform indentation

        if not qa.question:
            return lines  # skip deleted questions

        if qa.notes:
            if hide_answered == "qa":
                lines.append(f"{prefix}* Question ID: {qa.question_id}")
                lines.append(f"{prefix}  Status: Answered")
            else:
                lines.append(f"{prefix}* Question ID: {qa.question_id}")
                lines.append(f"{prefix}  Question: {qa.question}")
                if qa.rubric and qa.rubric.labels:
                    for label, desc in zip(qa.rubric.labels, qa.rubric.descriptions):
                        lines.append(f"{prefix}  {label}: {desc}")
                if hide_answered != "a":
                    for note in qa.notes:
                        lines.append(f"{prefix}  [NOTE] {note}")
        else:
            # unanswered questions
            lines.append(f"{prefix}* Question ID: {qa.question_id}")
            lines.append(f"{prefix}  Question: {qa.question}")
            if qa.rubric and qa.rubric.labels:
                for label, desc in zip(qa.rubric.labels, qa.rubric.descriptions):
                    lines.append(f"{prefix}  {label}: {desc}")

        # recurse into sub-questions (deeper indent)
        if qa.sub_questions:
            for sub_qa in qa.sub_questions:
                lines.extend(self.format_qa(sub_qa, hide_answered=hide_answered, indent=indent + 2))

        return lines

    def get_questions_and_notes_str(self, hide_answered: str = "", active_topics_only: bool = True) -> str:
        """Returns a hierarchical, structured string for topics, subtopics, and questions."""
        if hide_answered not in ["", "a", "qa", "all"]:
            raise ValueError('hide_answered must be "", "a", "qa", or "all"')

        output = []
        if not active_topics_only:
            topics_list = self.interview_topic_manager.get_all_topics()
        else:
            topics_list = self.interview_topic_manager.get_active_topics()

        for topic in topics_list:
            topic_complete = self.interview_topic_manager.check_core_topic_completion(topic.topic_id)
            output.append("=== TOPIC ===")
            output.append(f"Topic ID: {topic.topic_id}")
            output.append(f"Topic Description: {topic.description}")
            output.append(f"Topic Priority Weight: {topic.priority_weight}")
            output.append(f"Topic Status: {'COVERED — do not ask any more questions about this topic' if topic_complete else 'NOT COVERED'}")
            output.append(f"Allow Emergent Subtopics: {'Yes' if topic.allow_emergent else 'No'}\n")

            for subtopic in topic:
                output.append("    --- SUBTOPIC ---")
                output.append(f"    Subtopic ID: {subtopic.subtopic_id}")
                output.append(f"    Subtopic Description: {subtopic.description}")
                output.append(f"    Subtopic Priority Weight: {subtopic.priority_weight}")
                if subtopic.max_followups is not None:
                    output.append(f"    Max Follow-ups: {subtopic.max_followups} — ask one open question, then at most {subtopic.max_followups} follow-up(s) before moving on. Prioritize breadth over depth here.")
                if subtopic.coverage_criteria:
                    output.append(f"    Coverage Criteria:")
                    statuses = subtopic.criteria_coverage
                    for i, criterion in enumerate(subtopic.coverage_criteria):
                        done = statuses[i] if i < len(statuses) else False
                        marker = "✓ DONE" if done else "✗ NOT YET"
                        output.append(f"        [{marker}] {criterion}")
                # TODO I think if it's covered, can give the summary of subtopics; otherwise just notes
                if subtopic.check_coverage():
                    output.append(f"    Subtopic Status: COVERED")
                    output.append(f"    [Subtopic SUMMARY]: {subtopic.get_final_summary()}")
                else:
                    output.append(f"    Subtopic Status: NOT COVERED")
                    notes_str = "\n         -".join(subtopic.notes) if subtopic.notes else ""
                    output.append(f"    [Subtopic NOTES]: {notes_str}")
                    emergent_insights_str = (
                        "\n         - ".join(insight.description for insight in subtopic.emergent_insights)
                        if subtopic.emergent_insights else ""
                    )
                    output.append(f"    [Subtopic Emergent Insights Observed So Far]: {emergent_insights_str}")
                    if len(subtopic.get_coverage_feedback_gap()) > 0:
                        output.append(f"    [Subtopic POSSIBLE GAPS TO EXPLORE]: {subtopic.get_coverage_feedback_gap()}")

                if not hide_answered.startswith("all"): # TODO fix this hack lol
                    for qa in subtopic:
                        output.extend(self.format_qa(qa, hide_answered=hide_answered, indent=8))
                output.append("")  # blank line between subtopics

            output.append("")  # blank line between topics

        return "\n".join(output)
    
    def get_all_topics_and_subtopics(self, active_topics_only: bool = True) -> str:
        output = []
        if active_topics_only:
            topics_list = self.interview_topic_manager.get_active_topics()
        else:
            topics_list = self.interview_topic_manager.get_all_topics()
        
        for topic in topics_list:
            output.append("=== TOPIC ===")
            output.append(f"Topic ID: {topic.topic_id}")
            output.append(f"Topic Description: {topic.description}")
            output.append(f"Topic Priority Weight: {topic.priority_weight}")
            output.append(f"Allow Emergent Subtopics: {'Yes' if topic.allow_emergent else 'No'}\n")

            for subtopic in topic:
                output.append("    --- SUBTOPIC ---")
                output.append(f"    Subtopic ID: {subtopic.subtopic_id}")
                output.append(f"    Subtopic Description: {subtopic.description}")
                output.append(f"    Subtopic Priority Weight: {subtopic.priority_weight}")
                if subtopic.coverage_criteria:
                    output.append(f"    Coverage Criteria:")
                    for criterion in subtopic.coverage_criteria:
                        output.append(f"        - {criterion}")
                output.append("")  # blank line between subtopics

            output.append("")  # blank line between topics

        return "\n".join(output)
    
    def update_subtopic_coverage(self, subtopic_id: str, aggregated_notes: str):
        """Updates the coverage status of a subtopic."""
        self.interview_topic_manager.update_subtopic_coverage(subtopic_id, aggregated_notes)
        
    def update_subtopic_criteria_coverage(self, subtopic_id: str, statuses: list):
        """Updates per-criterion coverage for a subtopic."""
        self.interview_topic_manager.update_subtopic_criteria_coverage(subtopic_id, statuses)

    def give_feedback_subtopic_coverage(self, subtopic_id: str, feedback: str):
        self.interview_topic_manager.give_feedback_subtopic_coverage(subtopic_id, feedback)
        
    def revise_agenda_after_update(self):
        self.interview_topic_manager.revise_agenda_after_update()
        
        # For snapshot debugging, save agenda
        self.save(save_type="snapshot")

    def get_additional_notes_str(self) -> str:
        """Returns formatted string of additional notes."""
        if not self.additional_notes:
            return ""
        return "\n".join(self.additional_notes)

    def clear_questions(self):
        """Clears all questions from the session agenda, 
        resetting it to an empty state."""
        # Clear all topics and questions
        self.interview_topic_manager.reset()
        
        # Clear additional notes
        self.additional_notes = []

    def visualize_topics(self) -> str:
        """Returns a tree visualization of topics and questions."""
        return str(self.interview_topic_manager)
    
    def add_emergent_subtopic(self, topic_id: str, subtopic_description: str) -> bool:
        """Adds a new emergent subtopic under the specified topic."""
        return self.interview_topic_manager.add_emergent_subtopic(core_topic_id=topic_id,
                                                                  new_subtopic_description=subtopic_description)

    def add_new_core_topic(self, description: str, subtopics: list) -> str:
        """Dynamically add a new core topic with given subtopics. Returns the new topic_id."""
        return self.interview_topic_manager.add_new_core_topic(description=description, subtopics=subtopics)

    def add_task_deep_dive(self, task_name: str, subtopics: list) -> str:
        """Queue or immediately create a Task Deep Dive topic. Returns 'created:<id>', 'queued', or 'exists'."""
        return self.interview_topic_manager.add_task_deep_dive(task_name=task_name, subtopics=subtopics)

    @classmethod
    def get_historical_session_summaries(cls, user_id: str) -> str:
        """Returns formatted string of all historical session summaries."""
        base_path = os.path.join(LOGS_DIR, user_id, "execution_logs")
        if not os.path.exists(base_path):
            return ""
            
        # Get all session directories
        session_dirs = [d for d in os.listdir(base_path) \
                       if d.startswith('session_') and \
                       os.path.isdir(os.path.join(base_path, d))]
        if not session_dirs:
            return ""
            
        # Sort directories by session number
        session_dirs.sort(key=lambda x: int(x.split('_')[1]))
        
        summaries = []
        for dir_name in session_dirs:
            session_id = int(dir_name.split('_')[1])
            file_path = os.path.join(base_path, dir_name, "session_agenda.json")
            
            if os.path.exists(file_path):
                # Load the session agenda
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summary = data.get('last_meeting_summary', '')
                    if summary:
                        summaries.append(f"Session {session_id}:\n{summary}")
        
        return "\n\n".join(summaries)

    def add_emergent_insight(self, subtopic_id: str, insight_data: Dict[str, Any]) -> None:
        """
        Add an emergent insight (counter-intuitive finding within a topic).

        Args:
            subtopic_id: The subtopic ID
            insight_data: Dictionary with keys:
                - description: str (what makes this emergent)
                - novelty_score: float (0-1, how unexpected)
                - evidence: List[str] (supporting conversation turns)
                - conventional_belief: str (what we expected instead)
        """
        return self.interview_topic_manager.add_emergent_insight_subtopic(subtopic_id, insight_data)

### EXAMPLE format ###
# Session 1 - January 1, 2024 at 10:00 AM
# --------------------------------------------------

# User Information:
# --------------------
# Name: John Doe
# Age: 30

# Previous Session Summary:
# --------------------
# Last session focused on career goals...

# Interview Notes:
# --------------------

# # Career Goals
# - 1. What are your long-term career aspirations?
#   → Wants to become a senior developer
#   → Interested in leadership roles
# - 1.1. What timeline do you have in mind?
#   → 3-5 years for senior role
# - 1.2. What skills do you need to develop?
# - 2. What challenges are you facing currently?
#   → Time management issues
#   → Technical skill gaps

# # Work-Life Balance
# - 1. How do you manage stress?
# - 2. What's your current work schedule like?
# - 2.1. Are you satisfied with it?
# - 2.2. What would you change?
# - 2.2.1. How would those changes impact your productivity?

# Additional Notes:
# --------------------
# - Follow up on technical training opportunities
# - Schedule monthly check-ins
