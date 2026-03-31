import json
import os
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.utils.llm.engines import get_engine, invoke_engine
from src.content.weekly_snapshot.weekly_snapshot import WeeklySnapshot


class SnapshotDiff(BaseModel):
    """Structured differences between a previous snapshot and the current user portrait."""
    disappeared_tasks: List[str] = []       # Tasks from prev snapshot not in current portrait
    new_tasks: List[str] = []               # Tasks in current portrait not in prev snapshot
    time_allocation_shifts: List[str] = []  # Human-readable shift descriptions, e.g. "analysis: 20% → 40%"
    new_tools: List[str] = []
    dropped_tools: List[str] = []
    new_ai_tools: List[str] = []
    collaboration_changes: List[str] = []
    other_notable_changes: List[str] = []   # Catch-all for qualitative shifts


_DIFF_PROMPT = """You are comparing a worker's task and work profile from last week to their current known profile to identify meaningful changes.

Previous weekly snapshot (what they reported last week):
<previous_snapshot>
{previous_snapshot}
</previous_snapshot>

Current user profile (accumulated knowledge about this person):
<current_portrait>
{current_portrait}
</current_portrait>

Identify meaningful differences between the two. Focus on:
1. Tasks that appeared last week but are absent or less prominent now (disappeared_tasks)
2. Tasks in the current profile that didn't appear last week (new_tasks)
3. Meaningful shifts in time allocation across activity categories (time_allocation_shifts)
4. Tools that are new this week vs. last week (new_tools, new_ai_tools)
5. Tools that were used last week but no longer appear (dropped_tools)
6. Changes in who they work with or delegation patterns (collaboration_changes)
7. Any other notable qualitative changes (other_notable_changes)

Be precise and use the person's own language where possible. Only report genuine differences — do not flag minor rephrasing as a change. If something is unclear or ambiguous, omit it.

Return your response as a JSON object with these exact keys (all values are lists of strings):
{{
  "disappeared_tasks": [...],
  "new_tasks": [...],
  "time_allocation_shifts": [...],
  "new_tools": [...],
  "dropped_tools": [...],
  "new_ai_tools": [...],
  "collaboration_changes": [...],
  "other_notable_changes": [...]
}}

Return only the JSON object, no other text.
"""


class SnapshotDiffer:
    """Uses an LLM to compute meaningful differences between snapshots and the current user portrait."""

    def __init__(self):
        self._engine = get_engine(
            os.getenv("MODEL_NAME", "gpt-4.1-mini")
        )

    def compute_diff(
        self,
        prev_snapshot: WeeklySnapshot,
        current_portrait: Dict[str, Any],
    ) -> SnapshotDiff:
        """Compare prev_snapshot against current_portrait and return a SnapshotDiff.

        Uses an LLM call so it can handle semantic equivalence (e.g. "client deck prep"
        ≈ "preparing client presentations") and surface qualitative shifts.
        """
        prompt = _DIFF_PROMPT.format(
            previous_snapshot=json.dumps(prev_snapshot.to_dict(), indent=2),
            current_portrait=json.dumps(current_portrait, indent=2),
        )
        response = invoke_engine(self._engine, prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Strip markdown code fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]

        try:
            data = json.loads(text.strip())
            return SnapshotDiff(**{k: v for k, v in data.items() if k in SnapshotDiff.model_fields})
        except (json.JSONDecodeError, Exception):
            # Return an empty diff rather than crashing the session
            return SnapshotDiff()
