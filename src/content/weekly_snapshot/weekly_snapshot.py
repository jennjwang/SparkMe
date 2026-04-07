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

    # Topic 1: Tasks and Deliverables
    tasks: List[TaskEntry] = []
    time_allocation: dict = {}  # e.g. {"coding": 0.4, "meetings": 0.3}

    # Topic 2: Tools and Methods
    tools: List[str] = []  # General tools/methods used (e.g. "Figma", "JIRA")
    ai_tools: List[str] = []  # AI tools used (e.g. "ChatGPT", "Copilot")

    # Topic 3: Collaboration
    collaborators: List[str] = []  # Role/relationship descriptions (no PII)

    # Topic 4: Pain Points and Bright Spots
    pain_points: List[str] = []  # Frustrations or inefficiencies
    notable_changes: List[str] = []  # What went well or changed

    # Summary
    session_summary: str = ""

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "user_id": self.user_id,
            "session_id": self.session_id,
            "week_number": self.week_number,
            "timestamp": self.timestamp.isoformat(),
            "tasks": [t.to_dict() for t in self.tasks],
            "time_allocation": self.time_allocation,
            "tools": self.tools,
            "ai_tools": self.ai_tools,
            "collaborators": self.collaborators,
            "pain_points": self.pain_points,
            "notable_changes": self.notable_changes,
            "session_summary": self.session_summary,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "WeeklySnapshot":
        d = dict(d)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["tasks"] = [TaskEntry.from_dict(t) for t in d.get("tasks", [])]
        return cls(**d)
