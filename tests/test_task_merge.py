"""Tests for the task inventory merge utilities.

`_task_core` and `merge_task_inventories` are helper functions that implement
the dedup semantics used in tests and available as a fallback. The live portrait
extraction pipeline delegates merging to the LLM via the {prior_tasks} prompt
block, so these functions are not called in production but are kept as a
reference implementation and are tested here to document the intended semantics.
"""
import pytest
from src.interview_session.interview_session import (
    _task_core,
    merge_task_inventories,
    merge_tasks_by_action_object,
)


# ---------------------------------------------------------------------------
# _task_core
# ---------------------------------------------------------------------------

class TestTaskCore:
    def test_strips_purpose_clause(self):
        assert _task_core("running experiments to publish a paper") == "running experiments"

    def test_no_purpose_clause(self):
        assert _task_core("attending lab meetings") == "attending lab meetings"

    def test_normalizes_case(self):
        assert _task_core("Running Experiments to Test Hypotheses") == "running experiments"

    def test_strips_whitespace(self):
        assert _task_core("  reading papers  ") == "reading papers"

    def test_keeps_first_to_segment_only(self):
        # "to" appears multiple times — only the first segment is kept
        assert _task_core("writing proposals to secure funding to advance research") == "writing proposals"

    def test_empty_string(self):
        assert _task_core("") == ""


# ---------------------------------------------------------------------------
# merge_task_inventories
# ---------------------------------------------------------------------------

class TestMergeTaskInventories:
    def test_no_prior_returns_new(self):
        new = ["running experiments to publish", "reading papers to stay current"]
        assert merge_task_inventories(new, []) == new

    def test_prior_task_preserved_when_absent_from_new(self):
        new = ["running experiments to publish a paper"]
        prior = ["reading papers to understand the field"]
        merged = merge_task_inventories(new, prior)
        assert "reading papers to understand the field" in merged

    def test_prior_task_dropped_when_covered_by_new(self):
        # Same action+object, different purpose — new version should cover prior
        new = ["running experiments to publish a paper"]
        prior = ["running experiments to test hypotheses"]
        merged = merge_task_inventories(new, prior)
        assert len(merged) == 1
        assert merged[0] == "running experiments to publish a paper"

    def test_prefix_substring_match_covers_prior(self):
        # "attending meetings" (core) matches "attending meetings to exchange updates" (new core)
        new = ["attending meetings to exchange updates and feedback"]
        prior = ["attending meetings"]
        merged = merge_task_inventories(new, prior)
        assert len(merged) == 1

    def test_cadence_word_ignored_in_matching(self):
        # "weekly" should be stripped so both resolve to the same core
        new = ["attending weekly project meetings to sync on status"]
        prior = ["attending project meetings"]
        merged = merge_task_inventories(new, prior)
        assert len(merged) == 1

    def test_distinct_action_objects_both_preserved(self):
        new = ["running experiments to publish"]
        prior = ["attending advisor meetings to get feedback"]
        merged = merge_task_inventories(new, prior)
        assert len(merged) == 2
        assert "attending advisor meetings to get feedback" in merged

    def test_multiple_prior_partially_covered(self):
        new = [
            "running experiments to publish a paper",
            "analyzing results to interpret findings",
        ]
        prior = [
            "running experiments to test hypotheses",   # covered — same action+object
            "reading papers to track the field",        # NOT covered — different action+object
            # "analyzing experiment results" differs from "analyzing results" (extra word)
            # so it is NOT deduplicated by substring match — both are preserved
            "analyzing experiment results to publish",
        ]
        merged = merge_task_inventories(new, prior)
        assert "reading papers to track the field" in merged
        assert "analyzing experiment results to publish" in merged
        assert len(merged) == 4  # 2 new + 2 preserved prior

    def test_empty_new_preserves_all_prior(self):
        prior = ["task a", "task b"]
        merged = merge_task_inventories([], prior)
        assert set(merged) == set(prior)

    def test_case_insensitive_matching(self):
        new = ["Running Experiments to Publish"]
        prior = ["running experiments to test hypotheses"]
        merged = merge_task_inventories(new, prior)
        assert len(merged) == 1

    def test_real_world_near_duplicate_weekly_cadence(self):
        # Regression: cadence word "weekly" should be stripped so both resolve
        # to the same core ("participating in lab meetings") and are treated as one task.
        new = ["participating in lab meetings to present work and receive feedback"]
        prior = ["participating in weekly lab meetings to present work and receive feedback"]
        merged = merge_task_inventories(new, prior)
        assert len(merged) == 1

    def test_all_tasks_from_conversation_captured(self):
        # Simulates the session that had 5 tasks when it should have had 10+.
        # Verifies that all tasks from the user's stated week are preserved.
        new = [
            "running experiments using AI tools to publish papers",
            "analyzing experiment results to publish papers",
            "attending weekly project meetings to sync on work status",
            "preparing for advisor meetings to efficiently present updates",
            "attending advisor meetings to get directional feedback",
            "meeting with peers and PhD students to hear what they are working on",
            "reading research papers to understand the field",
            "attending lab meetings to give feedback on lab members' work",
            "attending research talks as part of PhD activities",
            "writing project proposal writeups",
            "writing paper submission writeups",
        ]
        merged = merge_task_inventories(new, [])
        assert len(merged) == 11


class TestMergeTasksByActionObject:
    def test_merges_same_action_object_even_with_different_objectives(self):
        tasks = [
            "running experiments to test hypotheses",
            "running experiments to publish a paper",
        ]
        merged = merge_tasks_by_action_object(tasks)
        assert len(merged) == 1
        assert merged[0] == "running experiments to publish a paper"

    def test_keeps_distinct_action_object_tasks(self):
        tasks = [
            "running experiments to publish",
            "attending advisor meetings to get feedback",
        ]
        merged = merge_tasks_by_action_object(tasks)
        assert len(merged) == 2

    def test_prefers_objective_variant_over_bare_action_object(self):
        tasks = [
            "attending lab meetings",
            "attending lab meetings to get feedback",
        ]
        merged = merge_tasks_by_action_object(tasks)
        assert len(merged) == 1
        assert merged[0] == "attending lab meetings to get feedback"
