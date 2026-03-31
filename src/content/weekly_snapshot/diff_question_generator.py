import os
import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from src.utils.llm.engines import get_engine, invoke_engine
from src.content.weekly_snapshot.snapshot_differ import SnapshotDiff


class DiffQuestion(BaseModel):
    """A question generated from a specific diff item, grounded in prior memories."""
    question: str
    subtopic_hint: str          # Which weekly check-in subtopic this targets
    diff_type: str              # e.g. "disappeared_task", "new_tool", "time_shift"
    source_diff_item: str       # The raw diff entry that triggered this question
    memory_context: str = ""    # Relevant memory text used to personalize the question
    priority: int = 5           # 1-10, higher = more important to ask


_QUESTION_GEN_PROMPT = """You are preparing context-aware opening questions for a weekly work check-in interview.

Below are the differences detected between this person's work last week and their current profile,
along with relevant memories from past sessions that provide more context.

Diff items and their associated memories:
<diff_items>
{diff_items_with_memories}
</diff_items>

For each diff item, generate ONE concise, natural-sounding follow-up question. The question should:
- Reference the person's actual prior experience where memory context is available
- Sound conversational, not clinical (avoid "I detected a change in...")
- Be open-ended and not lead the answer
- Be specific enough that it can't be answered with yes/no alone
- Target the most meaningful diffs — skip trivial ones

Also specify:
- subtopic_hint: which of these best matches: "tasks_this_week", "tools_used", "time_allocation", "collaboration", "changes_and_observations"
- diff_type: one of disappeared_task, new_task, time_shift, new_tool, dropped_tool, new_ai_tool, collaboration_change, other
- priority: 1-10 (10 = must ask, 1 = nice to have). Prioritize disappeared tasks and time shifts highly.

Return a JSON array:
[
  {{
    "question": "...",
    "subtopic_hint": "...",
    "diff_type": "...",
    "source_diff_item": "...",
    "memory_context": "...",
    "priority": 8
  }},
  ...
]

Return only the JSON array, no other text.
"""


class DiffQuestionGenerator:
    """Generates context-aware check-in questions from a SnapshotDiff,
    grounding each question in relevant memories retrieved from the memory bank.
    """

    def __init__(self, memory_bank=None):
        """
        Args:
            memory_bank: VectorMemoryBank instance (optional at init, can pass to generate()).
        """
        self._engine = get_engine(
            os.getenv("MODEL_NAME", "gpt-4.1-mini")
        )
        self._memory_bank = memory_bank

    def generate(
        self,
        diff: SnapshotDiff,
        memory_bank=None,
        top_k_memories: int = 3,
    ) -> List[DiffQuestion]:
        """Generate diff-grounded questions, enriched with memory bank search results.

        Args:
            diff: SnapshotDiff produced by SnapshotDiffer.compute_diff()
            memory_bank: VectorMemoryBank to search for relevant memories.
                         Falls back to self._memory_bank if not provided.
            top_k_memories: Number of memories to retrieve per diff item.

        Returns:
            List of DiffQuestion objects sorted by priority (descending).
        """
        bank = memory_bank or self._memory_bank
        diff_items_with_memories = self._collect_diff_items_with_memories(
            diff, bank, top_k_memories
        )

        if not diff_items_with_memories:
            return []

        prompt = _QUESTION_GEN_PROMPT.format(
            diff_items_with_memories=json.dumps(diff_items_with_memories, indent=2)
        )
        response = invoke_engine(self._engine, prompt)
        text = response.content if hasattr(response, "content") else str(response)

        # Strip markdown fences
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
            text = text.rsplit("```", 1)[0]

        try:
            raw = json.loads(text.strip())
            questions = [DiffQuestion(**item) for item in raw if isinstance(item, dict)]
        except Exception:
            return []

        return sorted(questions, key=lambda q: q.priority, reverse=True)

    def _collect_diff_items_with_memories(
        self,
        diff: SnapshotDiff,
        memory_bank,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Pair each diff item with retrieved memories for richer question grounding."""
        items = []

        diff_entries: List[tuple[str, str]] = (
            [("disappeared_task", t) for t in diff.disappeared_tasks]
            + [("new_task", t) for t in diff.new_tasks]
            + [("time_shift", t) for t in diff.time_allocation_shifts]
            + [("new_tool", t) for t in diff.new_tools]
            + [("dropped_tool", t) for t in diff.dropped_tools]
            + [("new_ai_tool", t) for t in diff.new_ai_tools]
            + [("collaboration_change", t) for t in diff.collaboration_changes]
            + [("other", t) for t in diff.other_notable_changes]
        )

        for diff_type, item_text in diff_entries:
            memories_text = ""
            if memory_bank is not None:
                results = memory_bank.search_memories(item_text, k=top_k)
                if results:
                    memories_text = "\n".join(
                        f"- [{r.title}] {r.text}" for r in results
                    )

            items.append({
                "diff_type": diff_type,
                "item": item_text,
                "relevant_memories": memories_text or "(no prior memories found)",
            })

        return items
