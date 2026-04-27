from datetime import datetime
from enum import Enum
from pydantic import BaseModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.interview_session.interview_session import InterviewSession

class MessageType(str, Enum):
    CONVERSATION = "conversation"
    FEEDBACK = "feedback"       # for detailed feedback
    LIKE = "like"               # for like action
    SKIP = "skip"               # for skip action
    TIME_SPLIT_WIDGET = "time_split_widget"  # triggers time-allocation slider in UI
    AI_USAGE_WIDGET = "ai_usage_widget"  # triggers AI-vs-manual bucket selection widget
    FEEDBACK_WIDGET = "feedback_widget"  # triggers end-of-session feedback form in UI
    PROFILE_CONFIRM_WIDGET = "profile_confirm_widget"  # triggers role/profile review panel after first topic
    TASK_VALIDATION_WIDGET = "task_validation_widget"  # triggers batch task validation UI after job description
    AI_TASK_WIDGET = "ai_task_widget"  # triggers AI task selection widget after task inventory is complete

class Message(BaseModel):
    id: str
    type: MessageType
    role: str
    content: str
    timestamp: datetime
    metadata: dict

class Participant:
    def __init__(self, title: str, interview_session: 'InterviewSession'):
        self.title: str = title
        self.interview_session = interview_session
    
    async def on_message(self, message: Message):
        """Handle new message notification"""
        pass
