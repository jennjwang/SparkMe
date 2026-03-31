from datetime import datetime
from typing import List
from pydantic import BaseModel, Field
import uuid


class TaskEntry(BaseModel):
    description: str  # Free-text in the user's own words
    time_share: float = 0.0  # 0.0–1.0 fraction of work week
    ai_involved: bool = False
    ai_tool: str = ""  # Which AI tool, if any
    ai_purpose: str = ""  # What the AI tool was used for

    def to_dict(self) -> dict:
        return {
            "description": self.description,
            "time_share": self.time_share,
            "ai_involved": self.ai_involved,
            "ai_tool": self.ai_tool,
            "ai_purpose": self.ai_purpose,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskEntry":
        return cls(**d)


class WeeklySnapshot(BaseModel):
    snapshot_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: str
    session_id: int
    week_number: int  # ISO week number (1–53)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # This week's work
    tasks: List[TaskEntry] = []
    collaborators_this_week: List[str] = []  # Role/relationship descriptions (no PII)
    notable_events: str = ""  # Anything surprising or out of pattern

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "week_number": self.week_number,
            "timestamp": self.timestamp.isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
            "collaborators_this_week": self.collaborators_this_week,
            "notable_events": self.notable_events,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeeklySnapshot":
        d = dict(d)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["tasks"] = [TaskEntry.from_dict(t) for t in d.get("tasks", [])]
        return cls(**d)
