from datetime import datetime
import string
import random
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, model_validator

class Rubric(BaseModel):
    """Represents a rubric for quantifiable information."""
    labels: List[str] = Field(description="Labels for each scale point")
    descriptions: List[str] = Field(description="Descriptions for each scale point")

    @model_validator(mode='after')
    def check_lengths_match(self) -> 'Rubric':
        """Ensure labels and descriptions have the same length."""
        if self.labels and self.descriptions and len(self.labels) != len(self.descriptions):
            raise ValueError('The number of labels must match the number of descriptions.')
        return self

class Question(BaseModel):
    """Model for storing interview questions with their associated memories."""
    id: str
    content: str
    memory_ids: list[str]  # IDs of memories related to this question
    timestamp: datetime
    
    # New fields for workforce interview support
    subtopic_id: Optional[str] = Field(default=None, description="Link to a subtopic if this question captures information about the subtopic")
    rubric: Optional[Rubric] = Field(default=None, description="Rubric if this question uses a rubric")

    def to_dict(self) -> dict:
        """Convert Question object to dictionary."""
        return {
            'id': self.id,
            'content': self.content,
            'memory_ids': self.memory_ids,
            'timestamp': self.timestamp.isoformat(),
            'subtopic_id': self.subtopic_id,
            'rubric': self.rubric.model_dump() if self.rubric else None,
        }

    @classmethod
    def from_dict(cls, question_dict: dict) -> 'Question':
        """Create Question object from dictionary."""
        rubric_data = question_dict.get('rubric')
        return cls(
            id=question_dict['id'],
            content=question_dict['content'],
            memory_ids=question_dict['memory_ids'],
            timestamp=datetime.fromisoformat(question_dict['timestamp']),
            subtopic_id=question_dict.get('subtopic_id'),
            rubric=Rubric(**rubric_data) if rubric_data else None,
        )
        
    @classmethod
    def generate_question_id(cls) -> str:
        """Generate a short, unique question ID.
        Format: Q_MMDDHHMM_{random_chars}
        Example: Q_03121423_X7K (March 12, 14:23)
        """
        timestamp = datetime.now().strftime("%m%d%H%M")
        random_chars = ''.join(random.choices(
            string.ascii_uppercase + string.digits, k=3))
        return f"Q_{timestamp}_{random_chars}"

class QuestionSearchResult(Question):
    """Model for question search results that includes similarity score."""
    similarity_score: float = Field(ge=0, le=1)  # Score between 0 and 1

    @classmethod
    def from_question(cls, question: Question, similarity_score: float) -> 'QuestionSearchResult':
        """Create a search result from a Question object and similarity score."""
        return cls(
            id=question.id,
            content=question.content,
            memory_ids=question.memory_ids,
            timestamp=question.timestamp,
            subtopic_id=question.subtopic_id,
            rubric=question.rubric,
            similarity_score=similarity_score
        )

class SimilarQuestionsGroup(BaseModel):
    """Model for grouping a proposed question with its similar existing questions."""
    proposed: str
    similar: List[QuestionSearchResult] 

class InterviewQuestion(BaseModel):
    """Model for an interview question, including sub-questions and notes."""
    subtopic_id: str
    question_id: str
    question: str
    rubric: Optional[Rubric] = Field(default=None, description="Rubric if this question uses a rubric")
    notes: List[str] = Field(default_factory=list)
    sub_questions: List['InterviewQuestion'] = Field(default_factory=list)
    
    def add_sub_question(self, current_index: str, sub_question: 'InterviewQuestion') -> bool:
        """Return True if successfully adding sub question."""
        # Splits recursively for '.'
        if len(current_index) == 1:
            # Ensure that the current_index is extending the current sub questions list
            current_index_int = int(current_index) - 1
            if current_index_int >= 0 and current_index_int == len(self.sub_questions):
                self.sub_questions.append(sub_question)
                return True
            else:
                return False
        else:
            sub_question_list_index = int(current_index.split(".")[0]) - 1
            new_index = ".".join(current_index.split(".")[1:])
            if sub_question_list_index < len(self.sub_questions):
                return self.sub_questions[sub_question_list_index].add_sub_question(new_index, sub_question)
            else:
                return False
    
    def to_dict(self) -> dict:
        """Convert InterviewQuestion object to dictionary."""
        return {
            'subtopic_id': self.subtopic_id,
            'question_id': self.question_id,
            'question': self.question,
            'rubric': self.rubric.model_dump() if self.rubric else None,
            'notes': self.notes,
            'sub_questions': [sub_q.to_dict() for sub_q in self.sub_questions],
        }

    @classmethod
    def from_dict(cls, question_dict: dict) -> 'InterviewQuestion':
        """Create InterviewQuestion object from dictionary."""
        sub_questions_data = question_dict.get('sub_questions', [])
        rubric_data = question_dict.get('rubric')
        return cls(
            subtopic_id=question_dict['subtopic_id'],
            question_id=question_dict['question_id'],
            question=question_dict['question'],
            rubric=Rubric(**rubric_data) if rubric_data else None,
            notes=question_dict.get('notes', []),
            sub_questions=[cls.from_dict(sub_q) for sub_q in sub_questions_data],
        )
