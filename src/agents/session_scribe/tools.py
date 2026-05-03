from typing import Type, Optional, List, Callable, Dict, Any, Union
import json
import pathlib


from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field, SkipValidation, field_validator

from src.content.memory_bank.memory_bank_base import MemoryBankBase, Memory
from src.content.session_agenda.session_agenda import SessionAgenda
from src.content.question_bank.question_bank_base import QuestionBankBase, Rubric


class UpdateSessionNoteInput(BaseModel):
    subtopic_id: str = Field(
        description=(
            "The ID of the subtopic"
        )
    )
    note: str = Field(
        description="A concise note to be added to the question, or as an additional note.")


class UpdateSessionNote(BaseTool):
    """Tool for updating the session agenda."""
    name: str = "update_session_agenda"
    description: str = "A tool for updating the session agenda."
    args_schema: Type[BaseModel] = UpdateSessionNoteInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        question_id: str,
        note: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        self.session_agenda.add_note(question_id=str(question_id), note=note)
        target_question = question_id if question_id else "additional note"
        return f"Successfully added the note for `{target_question}`."


class UpdateMemoryBankAndSessionInput(BaseModel):
    title: str = Field(description="A concise but descriptive title for the memory")
    text: str = Field(description="A clear summary of the information")
    subtopic_links: Union[str, List[Dict[str, Any]]] = Field(
        description=(
            "List of subtopics this memory relates to. "
            "Format: (a list of JSON dictionary) where "
            "each entry must contain: subtopic_id (from topics_list), "
            "importance (1-10 for that subtopic), and relevance (explanation). "
            "Example: '[{\"subtopic_id\": \"the id\", \"importance\": 1-10, \"relevance\": \"why it matters\"}, ...]'"
        )
    )
    metadata: Optional[dict] = Field(description=(
        "Additional metadata about the memory. "
        "Format: A valid JSON dictionary."
        "This can include topics, people mentioned, emotions, locations, dates, relationships, life events, achievements, goals, aspirations, beliefs, values, preferences, hobbies, interests, education, work experience, skills, challenges, fears, dreams, etc. "
        ),
        default={}
    )
    
    @field_validator('subtopic_links', mode='before')
    @classmethod
    def parse_subtopic_links(cls, v):
        """Parse subtopic_links from string (JSON array or NDJSON) to list."""
        # If it's a string, parse it
        if isinstance(v, str):
            v = v.strip()
            # Remove markdown code blocks
            v = v.removeprefix('```json').removeprefix('```').removesuffix('```').strip()

            # Try parsing as JSON array first
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    v = parsed
                elif isinstance(parsed, dict):
                    # Single object, wrap in list
                    v = [parsed]
                else:
                    raise ValueError(f"Expected list or dict, got {type(parsed).__name__}")
            except json.JSONDecodeError:
                # Failed as JSON array, try parsing multiple JSON objects
                # LLM can output various formats:
                # 1. Newline-separated: {...}\n{...}
                # 2. Space-separated: {...} {...}
                # 3. Mixed: {...}\n{...} {...}

                v_list = []
                # Use a simple approach: find all {...} patterns
                # Match JSON objects (simple heuristic: balanced braces)
                current_obj = ""
                brace_count = 0

                for char in v:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1

                    current_obj += char

                    # When braces are balanced and we have content, parse it
                    if brace_count == 0 and current_obj.strip():
                        try:
                            obj = json.loads(current_obj.strip())
                            v_list.append(obj)
                            current_obj = ""
                        except json.JSONDecodeError:
                            # Skip invalid JSON, continue accumulating
                            pass

                if not v_list:
                    raise ValueError(f"Could not parse any valid JSON objects from: {v[:200]}...")

                v = v_list

        if isinstance(v, dict):
            v = [v]

        # Validate it's a list
        if not isinstance(v, list):
            raise ValueError(f"subtopic_links must be a list, got {type(v).__name__}")

        for i, link in enumerate(v):
            if not isinstance(link, dict):
                raise ValueError(f"Link {i} must be a dictionary, got {type(link).__name__}")

            # Required fields
            for field in ("subtopic_id", "importance", "relevance"):
                if field not in link:
                    raise ValueError(f"Link {i} missing required field '{field}'")

            # Validate importance range
            importance = link["importance"]
            if not isinstance(importance, (int, float)) or not (1 <= importance <= 10):
                raise ValueError(
                    f"Link {i} importance must be between 1-10, got {importance}"
                )

        return v

class UpdateMemoryBankAndSession(BaseTool):
    """Tool for updating the memory bank and session agenda."""
    name: str = "update_memory_bank_and_session"
    description: str = "A tool for storing new memories in the memory bank and updating the session agenda."
    args_schema: Type[BaseModel] = UpdateMemoryBankAndSessionInput
    memory_bank: MemoryBankBase = Field(...)
    session_agenda: SessionAgenda = Field(...)
    on_memory_added: SkipValidation[Callable[[Memory], None]] = Field(...)
    update_memory_map: SkipValidation[Callable[[str, str], None]] = Field(
        description="Callback function to update the memory ID mapping"
    )
    get_current_qa: SkipValidation[Callable[[], str]] = Field(
        description="Function to get the current interviewer's question and user's response"
    )

    def _run(
        self,
        title: str,
        text: str,
        subtopic_links: Union[str, List[Dict[str, Any]]],
        metadata: Optional[dict] = {},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Parse subtopic_links using the same logic as the validator
            # (LangChain bypasses the validator and passes raw strings to _run)
            subtopic_links = UpdateMemoryBankAndSessionInput.parse_subtopic_links(subtopic_links)

            # Ensure metadata is a valid dict, default to empty if not
            if not isinstance(metadata, dict):
                metadata = {}

            # First add memory to memory bank
            question_text, response_text = self.get_current_qa()
            memory = self.memory_bank.add_memory(
                title=title,
                text=text,
                subtopic_links=subtopic_links,
                metadata=metadata,
                source_interview_question=question_text,
                source_interview_response=response_text,
            )
            
            # # Use callback to update the mapping
            # self.update_memory_map(memory.id)
            
            # Trigger callback to track newly added memory
            self.on_memory_added(memory)
            
            # Now, add notes to session agenda
            for item_link in subtopic_links:
                self.session_agenda.add_note(subtopic_id=item_link['subtopic_id'], note=text)

            return f"Successfully stored memory and note for: {title}"
        except Exception as e:
            raise ToolException(f"Error storing memory: {e}")


class UpdateExistingMemoryInput(BaseModel):
    memory_id: str = Field(description="The ID of the existing memory to update/merge into")
    text: str = Field(description="The updated, aggregated summary that combines the old and new information")
    new_subtopic_links: Union[str, List[Dict[str, Any]]] = Field(
        description=(
            "List of NEW or updated subtopic links from this response. "
            "Format: same as update_memory_bank_and_session subtopic_links."
        )
    )
    title: Optional[str] = Field(
        description="Updated title if the scope of the memory has broadened. Leave empty to keep original.",
        default=None
    )
    metadata: Optional[dict] = Field(
        description="Additional metadata to merge into the existing memory.",
        default={}
    )

    @field_validator('new_subtopic_links', mode='before')
    @classmethod
    def parse_subtopic_links(cls, v):
        return UpdateMemoryBankAndSessionInput.parse_subtopic_links(v)


class UpdateExistingMemory(BaseTool):
    """Tool for updating/merging new information into an existing memory."""
    name: str = "update_existing_memory"
    description: str = (
        "Update an existing memory by merging new information from the current "
        "transcript turn. Use this instead of creating a new memory when the user "
        "provides additional details about something already stored."
    )
    args_schema: Type[BaseModel] = UpdateExistingMemoryInput
    memory_bank: MemoryBankBase = Field(...)
    session_agenda: SessionAgenda = Field(...)
    on_memory_added: SkipValidation[Callable[[Memory], None]] = Field(...)
    get_current_qa: SkipValidation[Callable[[], str]] = Field(
        description="Function to get the current interviewer's question and user's response"
    )

    def _run(
        self,
        memory_id: str,
        text: str,
        new_subtopic_links: Union[str, List[Dict[str, Any]]],
        title: Optional[str] = None,
        metadata: Optional[dict] = {},
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            new_subtopic_links = UpdateExistingMemoryInput.parse_subtopic_links(new_subtopic_links)
            if not isinstance(metadata, dict):
                metadata = {}

            question_text, response_text = self.get_current_qa()
            updated_memory = self.memory_bank.update_memory(
                memory_id=memory_id,
                text=text,
                new_subtopic_links=new_subtopic_links,
                source_interview_question=question_text,
                source_interview_response=response_text,
                title=title,
                additional_metadata=metadata,
            )

            if updated_memory is None:
                raise ToolException(f"Memory with ID {memory_id} not found")

            self.on_memory_added(updated_memory)

            # Update session agenda notes for all linked subtopics
            for item_link in new_subtopic_links:
                self.session_agenda.add_note(subtopic_id=item_link['subtopic_id'], note=text)

            return f"Successfully updated memory {memory_id}: {updated_memory.title}"
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(f"Error updating memory: {e}")


class UpdateCriteriaCoverageInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic whose criteria are being evaluated. Example: '1.1'."
    )
    criteria_statuses: List[bool] = Field(
        description=(
            "A list of booleans, one per coverage criterion, in the same order they appear "
            "in the subtopic's Coverage Criteria list. True = criterion is met by the notes, "
            "False = criterion is not yet met. Must match the number of criteria exactly."
        )
    )

class UpdateCriteriaCoverage(BaseTool):
    """Tool for reporting which individual coverage criteria are met for a subtopic."""
    name: str = "update_criteria_coverage"
    description: str = (
        "Report which individual coverage criteria are met for a subtopic. "
        "Only call this for subtopics that have Coverage Criteria defined. "
        "Pass a boolean for each criterion in order."
    )
    args_schema: Type[BaseModel] = UpdateCriteriaCoverageInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        criteria_statuses: List[bool],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            coerced = [
                s if isinstance(s, bool) else str(s).lower() in ('true', '1', 'yes')
                for s in criteria_statuses
            ]
            self.session_agenda.update_subtopic_criteria_coverage(
                subtopic_id=str(subtopic_id),
                statuses=coerced,
            )
            met = sum(coerced)
            total = len(coerced)
            return f"Updated criteria coverage for subtopic {subtopic_id}: {met}/{total} criteria met."
        except Exception as e:
            raise ToolException(f"Error updating criteria coverage: {e}")


class UpdateSubtopicCoverageInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic to mark as covered (must exist in topics_list). Example: '1.1'."
    )
    aggregated_notes: str = Field(
        description="Final synthesis of the discussion or notes for this subtopic."
    )
class UpdateSubtopicCoverage(BaseTool):
    """Tool for updating the subtopics coverage."""
    name: str = "update_subtopic_coverage"
    description: str = "A tool for updating the coverage of subtopics along with the summary."
    args_schema: Type[BaseModel] = UpdateSubtopicCoverageInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        aggregated_notes: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Ensure metadata is a valid dict, default to empty if not
            self.session_agenda.update_subtopic_coverage(subtopic_id=str(subtopic_id),
                                                         aggregated_notes=str(aggregated_notes))
                
            return f"Successfully updated the coverage for subtopic ID: {subtopic_id}"
        except Exception as e:
            raise ToolException(f"Error updating subtopic coverage: {e}")
        
        
class FeedbackSubtopicCoverageInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic that is not yet fully covered (must exist in topics_list). Example: '1.1'."
    )
    feedback: str = Field(
        description="Concise feedback of the missing elements or reasoning gaps for current subtopic."
    )
class FeedbackSubtopicCoverage(BaseTool):
    """Tool for giving feedback regarding the subtopic's coverage."""
    name: str = "feedback_subtopic_coverage"
    description: str = "A tool for providing feedback about the coverage of subtopic."
    args_schema: Type[BaseModel] = FeedbackSubtopicCoverageInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        feedback: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Ensure metadata is a valid dict, default to empty if not
            self.session_agenda.give_feedback_subtopic_coverage(subtopic_id=str(subtopic_id),
                                                         feedback=str(feedback))
                
            return f"Successfully provide feedback regarding the coverage for subtopic ID: {subtopic_id}"
        except Exception as e:
            raise ToolException(f"Error providing feedback for subtopic coverage: {e}")


class UpdateSubtopicNotesInput(BaseModel):
    subtopic_id: str = Field(
        description="The unique ID of the subtopic to be associated with the notes."
    )
    note_list: List[str] = Field(
        description="List of notes taken."
    )
class UpdateSubtopicNotes(BaseTool):
    """Tool for updating the subtopics note."""
    name: str = "update_subtopic_notes"
    description: str = "A tool for updating the notes of subtopics."
    args_schema: Type[BaseModel] = UpdateSubtopicNotesInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        subtopic_id: str,
        note_list: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            for note in note_list:
                self.session_agenda.add_note(str(subtopic_id), note=note)
                
            return f"Successfully updated the coverage for subtopic ID: {subtopic_id}"
        except Exception as e:
            raise ToolException(f"Error updating subtopic coverage: {e}")

class IdentifyEmergentInsightsInput(BaseModel):
    """Input for emergent insight identification tool"""

    emergent_insights: List[Dict[str, Any]] = Field(
        description=(
            "List of emergent insights. Each insight must be a dictionary with: "
            "subtopic_id (str), description (str), novelty_score from 1 (mildly unexpected) to 5 (highly counter-intuitive), "
            "evidence (str), conventional_belief (str)."
        )
    )

    @field_validator("emergent_insights", mode="before")
    @classmethod
    def validate_insights(cls, v):
        """Validate emergent insights structure"""
        if not isinstance(v, list):
            raise ValueError("emergent_insights must be a list")

        for i, insight in enumerate(v):
            if not isinstance(insight, dict):
                raise ValueError(f"Insight {i} must be a dictionary")

            required_fields = [
                "subtopic_id",
                "description",
                "novelty_score",
                "evidence",
                "conventional_belief",
            ]
            for field in required_fields:
                if field not in insight:
                    raise ValueError(f"Insight {i} missing required field '{field}'")

            # Validate novelty_score (1–5 integer)
            novelty = insight["novelty_score"]
            if not isinstance(novelty, int) or not (1 <= novelty <= 5):
                raise ValueError(
                    f"Insight {i} novelty_score must be an integer between 1 and 5"
                )

        return v

class IdentifyEmergentInsights(BaseTool):
    """
    Tool for identifying emergent insights (novel, counter-intuitive findings).

    Detects insights that:
    - Are unexpected or counter-intuitive
    - Contradict conventional wisdom
    - Are within topic scope but expand beyond existing subtopics
    """

    name: str = "identify_emergent_insights"
    description: str = (
        "Identify emergent insights that are novel and counter-intuitive. "
        "Only report insights with novelty_score >= min_novelty_score."
    )

    args_schema: Type[BaseModel] = IdentifyEmergentInsightsInput
    session_agenda: SessionAgenda = Field(...)
    min_novelty_score: int = Field(
        default=3,
        description="Minimum novelty threshold (1–5)"
    )

    def _run(
        self,
        emergent_insights: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Execute emergent insight identification"""
        added_count = 0

        for insight_data in emergent_insights:
            novelty_score = insight_data["novelty_score"]

            # Only add insights that meet the novelty threshold
            if novelty_score < self.min_novelty_score:
                continue

            # Construct insight payload
            insight = {
                "description": insight_data["description"],
                "novelty_score": novelty_score,
                "evidence": insight_data.get("evidence", ""),
                "conventional_belief": insight_data.get("conventional_belief", ""),
            }

            # Attach insight at the SUBTOPIC level
            self.session_agenda.add_emergent_insight(
                subtopic_id=insight_data["subtopic_id"],
                insight_data=insight,
            )

            added_count += 1

        return (
            f"Identified {added_count} emergent insight(s) "
            f"(minimum novelty_score: {self.min_novelty_score})"
        )


class AddHistoricalQuestionInput(BaseModel):
    content: str = Field(description="The question text to add")
    temp_memory_ids: List[str] = Field(
        description="Single-line list of temporary memory IDs relevant to this question. "
        "These should match the temporary IDs used in update_memory_and_session calls. "
        "Format should be a JSON-compatible list with quoted strings: "
        "['MEM_TEMP_1', 'MEM_TEMP_2']"
        "Empty list [] is also acceptable.",
        default=[]
    )


class AddHistoricalQuestion(BaseTool):
    """Tool for adding historical questions to the question bank."""
    name: str = "add_historical_question"
    description: str = (
        "A tool for storing questions that were asked in the interview. "
        "Use this when saving a question that has been asked, "
        "along with any memories that contain information relevant to this question."
    )
    args_schema: Type[BaseModel] = AddHistoricalQuestionInput
    question_bank: QuestionBankBase = Field(...)
    memory_bank: MemoryBankBase = Field(...)
    get_real_memory_ids: SkipValidation[Callable[[List[str]], List[str]]] = Field(
        description="Callback function to get real memory IDs from temporary IDs"
    )

    def _run(
        self,
        content: str,
        temp_memory_ids: List[str] = None,
        rubric: Rubric = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            # Use empty list if None
            temp_memory_ids = temp_memory_ids or []
            
            # Get real memory IDs through callback
            real_memory_ids = self.get_real_memory_ids(temp_memory_ids)
            
            # Link the question to memories
            # TODO link question with topic id
            question = self.question_bank.add_question(
                content=content,
                memory_ids=real_memory_ids,
                rubric=rubric,
            )
            
            # Link memories to the question
            for memory_id in real_memory_ids:
                self.memory_bank.link_question(memory_id, question.id)
            
            return f"Successfully stored question: {content}"
        except Exception as e:
            raise ToolException(f"Error storing question: {e}")


_DEEP_DIVE_CONFIG_PATH = pathlib.Path(__file__).resolve().parents[3] / "configs" / "task_deep_dive_subtopics.json"

with open(_DEEP_DIVE_CONFIG_PATH, "r") as _f:
    TASK_DEEP_DIVE_SUBTOPICS = json.load(_f)


class AddTaskDeepDiveTopicInput(BaseModel):
    task_name: str = Field(
        description=(
            "A short, descriptive name for the task the user described "
            "(e.g., 'Weekly status report', 'Client onboarding', 'Code review')."
        )
    )


class AddTaskDeepDiveTopic(BaseTool):
    """Tool for adding a Task Deep Dive topic for a specific task the user described."""
    name: str = "add_task_deep_dive_topic"
    description: str = (
        "Add a new 'Task Deep Dive' core topic for a specific task the user described. "
        "Call this once per distinct task when the user names a task they perform. "
        "It creates a structured topic with nine subtopics: Action, Object, Outcome, Tools, "
        "AI Involvement, Flow of Work, Team, Experience Requirements, and Pain Points and Highlights."
    )
    args_schema: Type[BaseModel] = AddTaskDeepDiveTopicInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        task_name: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            result = self.session_agenda.add_task_deep_dive(
                task_name=task_name,
                subtopics=TASK_DEEP_DIVE_SUBTOPICS,
            )
            if result == "queued":
                return (
                    f"Task Deep Dive for '{task_name}' has been queued. "
                    f"It will be created automatically once a current deep dive batch completes."
                )
            elif result == "exists":
                return f"Task Deep Dive for '{task_name}' already exists — skipping."
            else:
                topic_id = result.split(":", 1)[1]
                return (
                    f"Added Task Deep Dive topic for '{task_name}' as topic ID {topic_id} "
                    f"with 9 subtopics (Action, Object, Outcome, Tools, AI Involvement, "
                    f"Flow of Work, Team, Experience Requirements, Pain Points and Highlights)."
                )
        except Exception as e:
            raise ToolException(f"Error adding task deep dive topic: {e}")


class AddClarificationSubtopicInput(BaseModel):
    topic_id: str = Field(
        description="The ID of the topic this clarification subtopic belongs under."
    )
    description: str = Field(
        description=(
            "Description of what needs to be clarified. Be specific about what is missing. "
            "Example: \"Clarify task 'experiments': ask what specific action they do on experiments.\""
        )
    )


class AddClarificationSubtopic(BaseTool):
    """Tool for adding clarification subtopics that the Interviewer should cover."""
    name: str = "add_clarification_subtopic"
    description: str = (
        "Add a new subtopic when a captured task or answer needs more clarification. "
        "Use this when a task is too vague, lacks a clear action, or lacks a clear object. "
        "The Interviewer will cover this subtopic like any other uncovered subtopic."
    )
    args_schema: Type[BaseModel] = AddClarificationSubtopicInput
    session_agenda: SessionAgenda = Field(...)

    def _run(
        self,
        topic_id: str,
        description: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            added = self.session_agenda.add_emergent_subtopic(
                topic_id=str(topic_id),
                subtopic_description=description
            )
            if added:
                return f"Added clarification subtopic under topic {topic_id}: {description}"
            else:
                return f"Subtopic not added (may be duplicate or topic not found): {description}"
        except Exception as e:
            raise ToolException(f"Error adding clarification subtopic: {e}")
