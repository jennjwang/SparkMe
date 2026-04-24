"""Tests for the task hierarchy organizer.

Covers the pure-logic helpers (_sanitize, _collect_covered, _extract_json)
and organize_tasks end-to-end with a mocked LLM engine.
"""
import json
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from src.utils.task_hierarchy import (
    _build_screen_prompt,
    _collapse_core_duplicates,
    _collect_covered,
    _compose_merge_maps,
    _dedup_tasks_fuzzy,
    _dedup_tasks_llm,
    _extract_dedup_json,
    _extract_json,
    _flat_fallback,
    _is_context_only_work_mode_task,
    _is_descriptive_group_label,
    _needs_smart_refine,
    _norm,
    _pre_dedup_by_core,
    _screen_tasks,
    _sanitize,
    _to_impersonal_task_statement,
    _tool_agnostic_task_core,
    _rewrite_high_level_task_names,
    organize_tasks,
)


# ---------------------------------------------------------------------------
# _norm
# ---------------------------------------------------------------------------

class TestNorm:
    def test_lowercases(self):
        assert _norm("Running Experiments") == "running experiments"

    def test_collapses_whitespace(self):
        assert _norm("  reading   papers  ") == "reading papers"

    def test_empty(self):
        assert _norm("") == ""

    def test_none(self):
        assert _norm(None) == ""


# ---------------------------------------------------------------------------
# High-level statement rewriting
# ---------------------------------------------------------------------------

class TestHighLevelTaskRewrite:
    def test_to_impersonal_task_statement_strips_first_person(self):
        rewritten = _to_impersonal_task_statement("take a class during my free time")
        assert rewritten == "take a class during free time"

    def test_rewrite_high_level_task_names_only_touches_roots(self):
        tree = [
            {
                "name": "my planning work to keep my projects on track",
                "is_group": True,
                "children": [
                    {"name": "reviewing my notes to prepare for meetings", "children": []},
                    {"name": "writing updates to share progress", "children": []},
                ],
            }
        ]
        result = _rewrite_high_level_task_names(tree)
        assert "my " not in result[0]["name"].lower()
        # Child wording should remain unchanged (only high-level nodes are rewritten).
        assert result[0]["children"][0]["name"] == "reviewing my notes to prepare for meetings"


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_parses_hierarchy_tags(self):
        payload = '[{"name": "task a", "children": []}]'
        result = _extract_json(f"<hierarchy>{payload}</hierarchy>")
        assert result[0]["name"] == "task a"

    def test_strips_markdown_fences(self):
        payload = '```json\n[{"name": "task a", "children": []}]\n```'
        result = _extract_json(f"<hierarchy>{payload}</hierarchy>")
        assert result[0]["name"] == "task a"

    def test_falls_back_to_raw_json(self):
        payload = '[{"name": "task a", "children": []}]'
        result = _extract_json(payload)
        assert result[0]["name"] == "task a"


# ---------------------------------------------------------------------------
# LLM dedup parsing / behavior
# ---------------------------------------------------------------------------

class TestDedupParsingAndBehavior:
    def test_extract_dedup_json_accepts_tagged_fenced_payload(self):
        text = "<duplicates>```json\n[{\"indices\":[0]}]\n```</duplicates>"
        parsed = _extract_dedup_json(text)
        assert isinstance(parsed, list)
        assert parsed == [{"indices": [0]}]

    def test_dedup_falls_back_to_longest_when_merged_statement_missing(self):
        tasks = ["writing grants", "writing grants to secure funding"]
        resp = MagicMock()
        resp.content = "<duplicates>" + json.dumps([{"indices": [0, 1]}]) + "</duplicates>"
        with patch("src.utils.task_hierarchy.invoke_engine", return_value=resp):
            kept, merge_map = _dedup_tasks_llm(tasks, engine=MagicMock())
        assert kept == ["writing grants to secure funding"]
        assert merge_map == {"writing grants to secure funding": ["writing grants"]}

    def test_dedup_adds_singletons_for_omitted_indices(self):
        tasks = ["task a", "task b", "task c"]
        resp = MagicMock()
        # Model only accounts for indices 0 and 1; index 2 should be appended.
        resp.content = "<duplicates>" + json.dumps([{"indices": [0, 1], "merged_statement": "task a+b"}]) + "</duplicates>"
        with patch("src.utils.task_hierarchy.invoke_engine", return_value=resp):
            kept, merge_map = _dedup_tasks_llm(tasks, engine=MagicMock())
        assert "task a+b" in kept
        assert "task c" in kept
        assert merge_map == {"task a+b": ["task a", "task b"]}

    def test_dedup_ignores_invalid_and_duplicate_indices(self):
        tasks = ["alpha", "beta", "gamma"]
        resp = MagicMock()
        resp.content = (
            "<duplicates>"
            + json.dumps(
                [
                    {"indices": [0, 1], "merged_statement": "alpha/beta"},
                    {"indices": [1, 999, -1]},  # duplicate and out-of-range indices ignored
                ]
            )
            + "</duplicates>"
        )
        with patch("src.utils.task_hierarchy.invoke_engine", return_value=resp):
            kept, merge_map = _dedup_tasks_llm(tasks, engine=MagicMock())
        assert "alpha/beta" in kept
        assert "gamma" in kept
        assert merge_map == {"alpha/beta": ["alpha", "beta"]}

    def test_fuzzy_substring_merges_long_and_short_variant(self):
        tasks = [
            "writing research papers to communicate research findings",
            "writing research papers",
            "meeting with post docs",
        ]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)
        assert "writing research papers to communicate research findings" in kept
        assert "meeting with post docs" in kept
        assert merge_map == {
            "writing research papers to communicate research findings": [
                "writing research papers"
            ]
        }

    def test_fuzzy_merges_preparing_for_variant(self):
        tasks = ["preparing presentations", "preparing for presentations"]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)
        assert len(kept) == 1
        assert "preparing for presentations" in kept
        assert merge_map == {"preparing for presentations": ["preparing presentations"]}

    def test_fuzzy_does_not_merge_different_actions(self):
        tasks = ["reading research papers", "writing research papers"]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)
        assert kept == tasks
        assert merge_map == {}

    def test_fuzzy_transitive_chain_collapses_all_variants(self):
        tasks = [
            "writing grant applications",
            "writing grant applications to secure funding",
            "writing grant applications for funding opportunities",
        ]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)
        assert len(kept) == 1
        winner = kept[0]
        assert winner == "writing grant applications for funding opportunities"
        assert set(merge_map[winner]) == {
            "writing grant applications",
            "writing grant applications to secure funding",
        }

    def test_fuzzy_handles_case_and_punctuation_noise(self):
        tasks = [
            "Writing grant applications",
            "writing grant applications!!!",
        ]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)
        assert len(kept) == 1
        assert kept[0] == "writing grant applications!!!"
        assert merge_map == {"writing grant applications!!!": ["Writing grant applications"]}

    def test_fuzzy_does_not_merge_same_action_different_objects(self):
        tasks = ["writing grant applications", "writing grant reports"]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)
        assert kept == tasks
        assert merge_map == {}

    def test_fuzzy_merges_short_plus_purpose_clause_pairs(self):
        cases = [
            (
                "writing grants to secure funding",
                "writing grants",
            ),
            (
                "working from home to perform work remotely",
                "working from home",
            ),
            (
                "providing career development guidance to support trainee growth",
                "providing career development guidance for trainees",
            ),
            (
                "coming up with new research ideas to advance research",
                "coming up with new research ideas",
            ),
        ]
        for long_form, short_form in cases:
            kept, merge_map = _dedup_tasks_fuzzy([long_form, short_form])
            assert len(kept) == 1
            winner = kept[0]
            assert winner in {long_form, short_form}
            assert merge_map[winner] == [long_form if winner == short_form else short_form]

    def test_fuzzy_screenshot_example_regression(self):
        # Regression: full screenshot example should collapse the obvious
        # long-form vs short-form duplicate pairs.
        tasks = [
            "Participating in meetings and messaging colleagues to communicate and collaborate",
            "Having meetings with research team members to discuss research progress and collaboration",
            "Having a mix of meetings",
            "Slack messaging with colleagues, especially those that report to me",
            "Preparing for presentations",
            "Writing grants to secure funding",
            "Writing research papers to publish findings",
            "Providing career development guidance to support trainee growth",
            "Coming up with new research ideas to advance research",
            "Working from home to perform work remotely",
            "Working from home",
            "Writing grants",
            "Coming up with new research ideas",
            "Writing research papers",
            "Providing career development guidance for trainees",
        ]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)

        assert len(kept) == 10
        assert merge_map == {
            "Writing grants to secure funding": ["Writing grants"],
            "Writing research papers to publish findings": ["Writing research papers"],
            "Providing career development guidance to support trainee growth": [
                "Providing career development guidance for trainees"
            ],
            "Coming up with new research ideas to advance research": [
                "Coming up with new research ideas"
            ],
            "Working from home to perform work remotely": ["Working from home"],
        }

    def test_fuzzy_latest_screenshot_regression(self):
        # Regression: latest UI example where several semantically-equivalent
        # long/short variants were still shown as separate leaves.
        tasks = [
            "Communicating with collaborators and trainees to coordinate work and share information",
            "Attending meetings with research trainees and coordinators to plan and coordinate research activities",
            "Attending meetings with co-authors to coordinate writing and research activities",
            "Messaging colleagues on Slack to coordinate work and provide support",
            "Producing and disseminating research outputs to share findings with academic and professional audiences",
            "Writing and submitting research papers to publish findings in academic journals",
            "Preparing presentations to communicate work effectively",
            "Writing grants to secure research funding",
            "Developing new research ideas to advance the research program",
            "Providing career development guidance for trainees to support trainee growth and career progression",
            "Writing grants",
            "Developing new research ideas",
            "Providing career development guidance for trainees",
            "Meeting co-authors",
            "Slack messaging with colleagues, especially those that report to me",
            "Preparing for presentations",
        ]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)

        assert len(kept) == 10
        assert merge_map == {
            "Attending meetings with co-authors to coordinate writing and research activities": [
                "Meeting co-authors"
            ],
            "Messaging colleagues on Slack to coordinate work and provide support": [
                "Slack messaging with colleagues, especially those that report to me"
            ],
            "Preparing presentations to communicate work effectively": [
                "Preparing for presentations"
            ],
            "Writing grants to secure research funding": ["Writing grants"],
            "Developing new research ideas to advance the research program": [
                "Developing new research ideas"
            ],
            "Providing career development guidance for trainees to support trainee growth and career progression": [
                "Providing career development guidance for trainees"
            ],
        }

    def test_fuzzy_merges_reordered_action_object_with_overlap(self):
        tasks = [
            "messaging colleagues on slack to coordinate work and provide support",
            "slack messaging with colleagues, especially those that report to me",
        ]
        kept, merge_map = _dedup_tasks_fuzzy(tasks)

        assert len(kept) == 1
        assert kept[0] == "messaging colleagues on slack to coordinate work and provide support"
        assert merge_map == {
            "messaging colleagues on slack to coordinate work and provide support": [
                "slack messaging with colleagues, especially those that report to me"
            ]
        }


# ---------------------------------------------------------------------------
# _collect_covered
# ---------------------------------------------------------------------------

class TestCollectCovered:
    def test_flat_leaf_covered(self):
        nodes = [{"name": "running experiments", "children": []}]
        assert "running experiments" in _collect_covered(nodes)

    def test_merged_from_covered(self):
        nodes = [{"name": "running experiments", "merged_from": ["analyzing results"], "children": []}]
        covered = _collect_covered(nodes)
        assert "analyzing results" in covered
        assert "running experiments" in covered

    def test_group_name_not_covered(self):
        # Invented group labels are not input tasks — only their children count
        nodes = [{
            "name": "meetings group",
            "is_group": True,
            "children": [
                {"name": "attending lab meetings", "children": []},
            ],
        }]
        covered = _collect_covered(nodes)
        assert "meetings group" not in covered
        assert "attending lab meetings" in covered

    def test_nested_children_covered(self):
        nodes = [{
            "name": "experiments group",
            "is_group": True,
            "children": [
                {"name": "running experiments", "children": []},
                {"name": "analyzing results", "children": []},
            ],
        }]
        covered = _collect_covered(nodes)
        assert "running experiments" in covered
        assert "analyzing results" in covered


# ---------------------------------------------------------------------------
# _sanitize
# ---------------------------------------------------------------------------

class TestSanitize:
    def test_basic_leaf_passthrough(self):
        nodes = [{"name": "running experiments", "children": []}]
        result = _sanitize(nodes)
        assert len(result) == 1
        assert result[0]["name"] == "running experiments"

    def test_restores_verbatim_casing(self):
        verbatim = {"running experiments": "Running Experiments using AI"}
        nodes = [{"name": "running experiments", "children": []}]
        result = _sanitize(nodes, verbatim)
        assert result[0]["name"] == "Running Experiments using AI"

    def test_single_child_group_demoted(self):
        # An invented group with only 1 child should be dissolved
        nodes = [{
            "name": "meetings",
            "is_group": True,
            "children": [{"name": "attending lab meetings", "children": []}],
        }]
        result = _sanitize(nodes)
        assert len(result) == 1
        assert result[0]["name"] == "attending lab meetings"
        assert not result[0].get("is_group")

    def test_two_child_group_kept(self):
        nodes = [{
            "name": "meetings",
            "is_group": True,
            "children": [
                {"name": "attending lab meetings", "children": []},
                {"name": "attending project meetings", "children": []},
            ],
        }]
        result = _sanitize(nodes)
        assert len(result) == 1
        assert result[0]["is_group"] is True
        assert len(result[0]["children"]) == 2

    def test_empty_group_name_skipped(self):
        nodes = [{"name": "", "children": []}]
        result = _sanitize(nodes)
        assert result == []

    def test_non_dict_entries_skipped(self):
        nodes = ["not a dict", {"name": "valid task", "children": []}]
        result = _sanitize(nodes)
        assert len(result) == 1
        assert result[0]["name"] == "valid task"

    def test_merged_from_verbatim_restored(self):
        verbatim = {"running experiments to test hypotheses": "Running experiments to test hypotheses"}
        nodes = [{
            "name": "running experiments",
            "merged_from": ["running experiments to test hypotheses"],
            "children": [],
        }]
        result = _sanitize(nodes, verbatim)
        assert result[0]["merged_from"] == ["Running experiments to test hypotheses"]

    def test_grandchildren_are_dropped_to_enforce_max_depth_two(self):
        nodes = [{
            "name": "coordination work to align execution",
            "is_group": True,
            "children": [
                {
                    "name": "attending team meetings to share updates",
                    "children": [{"name": "nested leaf should be dropped", "children": []}],
                },
                {"name": "writing project updates to track progress", "children": []},
            ],
        }]
        result = _sanitize(nodes)
        assert len(result) == 1
        assert result[0]["is_group"] is True
        assert len(result[0]["children"]) == 2
        assert result[0]["children"][0]["children"] == []


# ---------------------------------------------------------------------------
# Group-label quality / refine trigger
# ---------------------------------------------------------------------------

class TestGroupLabelQuality:
    def test_descriptive_label_passes(self):
        assert _is_descriptive_group_label(
            "participating in meetings and talks to exchange updates and feedback"
        )

    def test_short_noun_phrase_fails(self):
        assert not _is_descriptive_group_label("Meetings and talks")

    def test_missing_purpose_clause_fails(self):
        assert not _is_descriptive_group_label("writing research documents")


class TestNeedsSmartRefine:
    def test_triggers_on_underspecified_group_labels_even_for_small_inputs(self):
        tree = [
            {
                "name": "Meetings and talks",
                "is_group": True,
                "children": [
                    {"name": "attending project meetings to sync on status", "children": []},
                    {"name": "attending talks to learn from research presentations", "children": []},
                ],
            }
        ]
        tasks = [
            "attending project meetings to sync on status",
            "attending talks to learn from research presentations",
            "reading papers to track the field",
        ]
        assert _needs_smart_refine(tree, tasks) is True


# ---------------------------------------------------------------------------
# organize_tasks — end-to-end with mocked LLM
# ---------------------------------------------------------------------------

def _make_engine_response(tree: List[Dict[str, Any]]) -> MagicMock:
    """Return a mock LLM response wrapping the given tree as a hierarchy tag."""
    payload = f"<hierarchy>{json.dumps(tree)}</hierarchy>"
    mock_resp = MagicMock()
    mock_resp.content = payload
    return mock_resp


class TestOrganizeTasks:
    def _run(self, tasks, tree_response):
        mock_resp = _make_engine_response(tree_response)
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", return_value=mock_resp):
            mock_get_engine.return_value = MagicMock()
            return organize_tasks(tasks, screen=False)

    def test_flat_list_passthrough(self):
        tasks = ["task a"]
        result = organize_tasks(tasks)
        assert result == [{"name": "task a", "children": []}]

    def test_default_model_is_advanced(self):
        tasks = ["task a", "task b"]
        mock_resp = _make_engine_response([{"name": "task a", "children": []}])
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", return_value=mock_resp):
            mock_get_engine.return_value = MagicMock()
            organize_tasks(tasks, screen=False)
        mock_get_engine.assert_called_once_with("gpt-5.4")

    def test_grouping_feedback_is_added_to_prompt(self):
        tasks = ["task a", "task b"]
        mock_resp = _make_engine_response([{"name": "task a", "children": []}])
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", return_value=mock_resp) as mock_invoke:
            mock_get_engine.return_value = MagicMock()
            organize_tasks(
                tasks,
                screen=False,
                grouping_feedback="Group meetings with collaboration tasks.",
            )
        prompt = mock_invoke.call_args.args[1]
        assert "User feedback on grouping" in prompt
        assert "Group meetings with collaboration tasks." in prompt

    def test_prompt_requires_impersonal_high_level_labels(self):
        tasks = ["task a", "task b"]
        mock_resp = _make_engine_response([{"name": "task a", "children": []}])
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", return_value=mock_resp) as mock_invoke:
            mock_get_engine.return_value = MagicMock()
            organize_tasks(tasks, screen=False)
        prompt = mock_invoke.call_args.args[1]
        assert "impersonal third-person gerund statements" in prompt
        assert '"I", "my", "me", "we", "our", "your"' in prompt

    def test_empty_input(self):
        assert organize_tasks([]) == []

    def test_llm_response_used(self):
        tasks = ["running experiments", "analyzing results"]
        tree = [
            {
                "name": "experiments",
                "is_group": True,
                "children": [
                    {"name": "running experiments", "children": []},
                    {"name": "analyzing results", "children": []},
                ],
            }
        ]
        result = self._run(tasks, tree)
        assert result[0]["is_group"] is True
        assert len(result[0]["children"]) == 2

    def test_safety_net_appends_uncovered_task(self):
        # LLM drops "reading papers" — the safety net should append it
        tasks = ["running experiments", "reading papers"]
        tree = [{"name": "running experiments", "children": []}]
        result = self._run(tasks, tree)
        names = [n["name"] for n in result]
        assert "reading papers" in names

    def test_can_disable_safety_net_task_append(self):
        tasks = ["running experiments", "reading papers"]
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps(
                [
                    {"indices": [0]},
                    {"indices": [1]},
                ]
            )
            + "</duplicates>"
        )
        group_resp = _make_engine_response([{"name": "running experiments", "children": []}])
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False, append_uncovered_tasks=False)

        names = [n["name"] for n in result]
        assert "running experiments" in names
        assert "reading papers" not in names

    def test_dedup_via_merged_from(self):
        # LLM merges two tasks — the merged_from list covers the dropped original
        tasks = ["running experiments to test", "running experiments to publish"]
        tree = [{
            "name": "running experiments to publish",
            "merged_from": ["running experiments to test"],
            "children": [],
        }]
        result = self._run(tasks, tree)
        assert len(result) == 1
        assert result[0]["merged_from"] == ["running experiments to test"]

    def test_ai_tool_variant_is_not_separate_task(self):
        tasks = [
            "writing code for experiments to support research",
            "using AI tools to write code for experiments starting from a blank slate",
        ]
        # LLM dedup decides they are the same action+object and emits one merged statement.
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps(
                [
                    {
                        "indices": [0, 1],
                        "merged_statement": "writing code for experiments to support research with AI tools",
                        "merge_reason": "same action+object; tool qualifier only",
                    }
                ]
            )
            + "</duplicates>"
        )
        group_resp = _make_engine_response(
            [{"name": "writing code for experiments to support research with AI tools", "children": []}]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False)
        assert len(result) == 1
        assert result[0]["name"] == "writing code for experiments to support research with AI tools"
        merged_from = result[0].get("merged_from") or []
        assert "using AI tools to write code for experiments starting from a blank slate" in merged_from

    def test_ai_task_not_merged_when_underlying_work_differs(self):
        tasks = [
            "writing code for experiments to support research",
            "using AI tools to summarize papers to prepare literature reviews",
        ]
        tree = [
            {"name": "writing code for experiments to support research", "children": []},
            {"name": "using AI tools to summarize papers to prepare literature reviews", "children": []},
        ]
        result = self._run(tasks, tree)
        assert len(result) == 2

    def test_llm_dedup_merges_preparing_presentations_pair(self):
        tasks = [
            "preparing presentations",
            "preparing for presentations",
        ]
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps(
                [
                    {
                        "indices": [0, 1],
                        "merged_statement": "preparing presentations to communicate research findings",
                        "merge_reason": "same action and object",
                    }
                ]
            )
            + "</duplicates>"
        )
        group_resp = _make_engine_response(
            [
                {
                    "name": "preparing presentations to communicate research findings",
                    "children": [],
                }
            ]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False)

        assert len(result) == 1
        assert result[0]["name"] == "preparing presentations to communicate research findings"
        assert "preparing for presentations" in (result[0].get("merged_from") or [])

    def test_organize_tasks_fuzzy_backstop_merges_when_llm_returns_singletons(self):
        tasks = [
            "writing grant applications to secure funding",
            "writing grant applications",
        ]
        # LLM dedup misses and returns both as singleton outputs.
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps(
                [
                    {"indices": [0]},
                    {"indices": [1]},
                ]
            )
            + "</duplicates>"
        )
        # Grouping still returns both leaves; fuzzy pass should collapse them.
        group_resp = _make_engine_response(
            [
                {"name": "writing grant applications to secure funding", "children": []},
                {"name": "writing grant applications", "children": []},
            ]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False)

        assert len(result) == 1
        assert result[0]["name"] == "writing grant applications to secure funding"
        assert "writing grant applications" in (result[0].get("merged_from") or [])

    def test_feedback_regroup_still_dedups_top_level_variants_against_group_children(self):
        tasks = [
            "Writing grants to secure research funding",
            "Writing research papers to disseminate research findings",
            "Writing grants",
            "Writing research papers",
        ]
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps([{"indices": [i]} for i in range(len(tasks))])
            + "</duplicates>"
        )
        group_resp = _make_engine_response(
            [
                {
                    "name": "writing research documents to secure funding and disseminate research findings",
                    "is_group": True,
                    "children": [
                        {"name": "Writing grants to secure research funding", "children": []},
                        {
                            "name": "Writing research papers to disseminate research findings",
                            "children": [],
                        },
                    ],
                },
                {"name": "Writing grants", "children": []},
                {"name": "Writing research papers", "children": []},
            ]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(
                tasks,
                screen=False,
                grouping_feedback="group writing tasks together",
            )

        top_names = [n["name"] for n in result]
        assert "Writing grants" not in top_names
        assert "Writing research papers" not in top_names

        group = next(
            n
            for n in result
            if n.get("is_group")
            and n.get("name")
            == "writing research documents to secure funding and disseminate research findings"
        )
        by_child_name = {c["name"]: c for c in (group.get("children") or [])}
        assert "Writing grants" in (
            by_child_name["Writing grants to secure research funding"].get("merged_from") or []
        )
        assert "Writing research papers" in (
            by_child_name["Writing research papers to disseminate research findings"].get("merged_from")
            or []
        )

    def test_screenshot_slack_variants_are_deduped(self):
        # "Communicating with colleagues via Slack messages" and
        # "Slack messaging with colleagues" are synonymous — the LLM dedup
        # pass is responsible for catching this (the fuzzy pass can't bridge
        # synonym-based equivalences like "communicating" ↔ "messaging").
        tasks = [
            "Attending meetings to coordinate work and collaborate with colleagues",
            "Having meetings with post docs",
            "Having meetings with ph. d students",
            "Having meetings with research coordinators",
            "Meeting co-authors",
            "Writing research documents",
            "Writing grants",
            "Writing research papers",
            "Communicating with colleagues via Slack messages",
            "Preparing presentation materials",
            "Providing career development guidance for trainees",
            "Developing new research ideas",
            "Slack messaging with colleagues",
        ]
        # LLM dedup merges the two Slack variants (indices 8 and 12).
        dedup_entries = [{"indices": [i]} for i in range(len(tasks)) if i not in (8, 12)]
        dedup_entries.append({
            "indices": [8, 12],
            "merged_statement": "Communicating with colleagues via Slack messages",
            "merge_reason": "same activity, different phrasing",
        })
        dedup_resp = MagicMock()
        dedup_resp.content = "<duplicates>" + json.dumps(dedup_entries) + "</duplicates>"

        group_resp = _make_engine_response(
            [
                {
                    "name": "Attending meetings to coordinate work and collaborate with colleagues",
                    "children": [
                        {"name": "Having meetings with post docs", "children": []},
                        {"name": "Having meetings with ph. d students", "children": []},
                        {"name": "Having meetings with research coordinators", "children": []},
                        {"name": "Meeting co-authors", "children": []},
                    ],
                },
                {
                    "name": "Writing research documents",
                    "children": [
                        {"name": "Writing grants", "children": []},
                        {"name": "Writing research papers", "children": []},
                    ],
                },
                {"name": "Communicating with colleagues via Slack messages", "children": []},
                {"name": "Preparing presentation materials", "children": []},
                {"name": "Providing career development guidance for trainees", "children": []},
                {"name": "Developing new research ideas", "children": []},
            ]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False)

        top_level_names = [n["name"] for n in result]
        assert "Communicating with colleagues via Slack messages" in top_level_names
        # Absorbed variant must not survive as a separate top-level leaf.
        assert "Slack messaging with colleagues" not in top_level_names
        # Absorbed original must appear in merged_from of the winner node.
        slack_node = next(n for n in result if n["name"] == "Communicating with colleagues via Slack messages")
        assert "Slack messaging with colleagues" in slack_node.get("merged_from", [])

    def test_organize_tasks_latest_screenshot_regression_with_singleton_llm_dedup(self):
        tasks = [
            "Communicating with collaborators and trainees to coordinate work and share information",
            "Attending meetings with research trainees and coordinators to plan and coordinate research activities",
            "Attending meetings with co-authors to coordinate writing and research activities",
            "Messaging colleagues on Slack to coordinate work and provide support",
            "Producing and disseminating research outputs to share findings with academic and professional audiences",
            "Writing and submitting research papers to publish findings in academic journals",
            "Preparing presentations to communicate work effectively",
            "Writing grants to secure research funding",
            "Developing new research ideas to advance the research program",
            "Providing career development guidance for trainees to support trainee growth and career progression",
            "Writing grants",
            "Developing new research ideas",
            "Providing career development guidance for trainees",
            "Meeting co-authors",
            "Slack messaging with colleagues, especially those that report to me",
            "Preparing for presentations",
        ]
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps([{"indices": [i]} for i in range(len(tasks))])
            + "</duplicates>"
        )
        group_resp = _make_engine_response([{"name": t, "children": []} for t in tasks])

        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[dedup_resp, group_resp]):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False)

        names = [n["name"] for n in result]
        assert len(result) == 10
        assert "Writing grants" not in names
        assert "Meeting co-authors" not in names
        assert "Preparing for presentations" not in names
        assert "Attending meetings with co-authors to coordinate writing and research activities" in names
        assert "Writing grants to secure research funding" in names

        by_name = {n["name"]: n for n in result}
        assert "Writing grants" in (by_name["Writing grants to secure research funding"].get("merged_from") or [])
        assert "Meeting co-authors" in (
            by_name["Attending meetings with co-authors to coordinate writing and research activities"].get("merged_from") or []
        )

    def test_fallback_on_llm_failure(self):
        tasks = ["task a", "task b"]
        with patch("src.utils.task_hierarchy.get_engine", side_effect=Exception("no engine")):
            result = organize_tasks(tasks)
        assert result == [{"name": "task a", "children": []}, {"name": "task b", "children": []}]

    def test_fallback_on_bad_llm_json(self):
        tasks = ["task a", "task b"]
        mock_resp = MagicMock()
        mock_resp.content = "<hierarchy>not json</hierarchy>"
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", return_value=mock_resp):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks)
        assert result == [{"name": "task a", "children": []}, {"name": "task b", "children": []}]

    def test_bad_json_triggers_repair_retry_and_recovers(self):
        tasks = ["task a", "task b"]
        bad = MagicMock()
        bad.content = "<hierarchy>not json</hierarchy>"
        repaired = _make_engine_response(
            [
                {"name": "task a", "children": []},
                {"name": "task b", "children": []},
            ]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", side_effect=[bad, repaired]) as mock_invoke:
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks)
        assert result == [{"name": "task a", "children": []}, {"name": "task b", "children": []}]
        assert mock_invoke.call_count == 2

    def test_child_not_duplicated_at_root(self):
        # If LLM returns a child AND a top-level leaf with the same name, deduplicate
        tasks = ["attending lab meetings", "attending project meetings"]
        tree = [
            {
                "name": "meetings",
                "is_group": True,
                "children": [
                    {"name": "attending lab meetings", "children": []},
                    {"name": "attending project meetings", "children": []},
                ],
            },
            {"name": "attending lab meetings", "children": []},  # duplicate at root
        ]
        result = self._run(tasks, tree)
        top_names = [n["name"] for n in result]
        assert top_names.count("attending lab meetings") == 0  # removed from root (lives in group)

    def test_pass3_refine_disabled_by_default(self):
        tasks = [
            "attending project meetings to sync on status",
            "attending talks to learn from research presentations",
            "reading papers to track the field",
        ]
        first_pass_tree = [
            {
                "name": "Meetings and talks",
                "is_group": True,
                "children": [
                    {"name": "attending project meetings to sync on status", "children": []},
                    {"name": "attending talks to learn from research presentations", "children": []},
                ],
            },
            {"name": "reading papers to track the field", "children": []},
        ]
        mock_resp = _make_engine_response(first_pass_tree)
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch("src.utils.task_hierarchy.invoke_engine", return_value=mock_resp) as mock_invoke:
            mock_get_engine.return_value = MagicMock()
            organize_tasks(tasks, screen=False)
        # Two LLM calls run by default: dedicated dedup pass + grouping pass.
        # The optional pass-3 smart refine should NOT run unless TASK_HIERARCHY_ENABLE_PASS3 is enabled.
        assert mock_invoke.call_count == 2

    def test_organize_tasks_rewrites_top_level_names_impersonally(self):
        tasks = [
            "take a class during my free time",
            "reviewing notes to prepare for class",
        ]
        tree = [
            {"name": "take a class during my free time", "children": []},
            {"name": "reviewing notes to prepare for class", "children": []},
        ]
        result = self._run(tasks, tree)
        names = [n["name"] for n in result]
        assert "take a class during free time" in names


# ---------------------------------------------------------------------------
# Screening
# ---------------------------------------------------------------------------

class TestScreening:
    def test_screen_prompt_requires_work_only_tasks(self):
        prompt = _build_screen_prompt(["doing taxes", "reviewing project proposals"])
        assert "Work-related (not personal-life admin)?" in prompt
        assert '"doing taxes"' in prompt
        assert '"filing personal taxes"' in prompt

    def test_context_only_work_mode_detector(self):
        assert _is_context_only_work_mode_task("working from home")
        assert _is_context_only_work_mode_task("working from home to perform work remotely")
        assert _is_context_only_work_mode_task("working remotely")
        assert not _is_context_only_work_mode_task("writing papers from home")
        assert not _is_context_only_work_mode_task("running experiments remotely")

    def test_screen_tasks_filters_context_only_even_if_llm_passes(self):
        tasks = [
            "working from home",
            "writing research papers",
        ]
        screen_resp = MagicMock()
        screen_resp.content = (
            "<screen>"
            + json.dumps(
                [
                    {"index": 0, "status": "pass", "rewritten": "", "reason": ""},
                    {"index": 1, "status": "pass", "rewritten": "", "reason": ""},
                ]
            )
            + "</screen>"
        )
        with patch("src.utils.task_hierarchy.invoke_engine", return_value=screen_resp):
            result = _screen_tasks(tasks, engine=MagicMock())
        assert result == ["writing research papers"]

    def test_organize_tasks_screen_true_drops_context_only_task(self):
        tasks = [
            "working from home",
            "writing research papers",
            "writing grants",
        ]
        screen_resp = MagicMock()
        screen_resp.content = (
            "<screen>"
            + json.dumps(
                [
                    {"index": 0, "status": "pass", "rewritten": "", "reason": ""},
                    {"index": 1, "status": "pass", "rewritten": "", "reason": ""},
                    {"index": 2, "status": "pass", "rewritten": "", "reason": ""},
                ]
            )
            + "</screen>"
        )
        dedup_resp = MagicMock()
        dedup_resp.content = (
            "<duplicates>"
            + json.dumps([{"indices": [0]}, {"indices": [1]}])
            + "</duplicates>"
        )
        group_resp = _make_engine_response(
            [
                {"name": "writing research papers", "children": []},
                {"name": "writing grants", "children": []},
            ]
        )
        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch(
                 "src.utils.task_hierarchy.invoke_engine",
                 side_effect=[screen_resp, dedup_resp, group_resp],
             ):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=True)
        names = [n["name"] for n in result]
        assert "working from home" not in names
        assert "writing research papers" in names
        assert "writing grants" in names


# ---------------------------------------------------------------------------
# Post-cluster duplicate collapse (near-duplicate aware)
# ---------------------------------------------------------------------------

class TestCollapseCoreDuplicatesNearDuplicates:
    def test_merges_near_duplicate_siblings(self):
        tree = [
            {
                "name": "attending meetings with co-authors to coordinate writing and research activities",
                "children": [],
            },
            {"name": "meeting co-authors", "children": []},
        ]
        result = _collapse_core_duplicates(tree)
        assert len(result) == 1
        assert result[0]["name"] == (
            "attending meetings with co-authors to coordinate writing and research activities"
        )
        assert "meeting co-authors" in (result[0].get("merged_from") or [])

    def test_merges_near_duplicate_parent_child(self):
        tree = [
            {
                "name": "writing grants to secure research funding",
                "children": [
                    {"name": "writing grants", "children": []},
                ],
            }
        ]
        result = _collapse_core_duplicates(tree)
        assert len(result) == 1
        assert result[0]["name"] == "writing grants to secure research funding"
        assert result[0]["children"] == []
        assert "writing grants" in (result[0].get("merged_from") or [])

    def test_merges_top_level_leaf_with_group_child(self):
        tree = [
            {
                "name": "writing research documents to secure funding and disseminate research findings",
                "is_group": True,
                "children": [
                    {"name": "Writing grants to secure research funding", "children": []},
                    {
                        "name": "Writing research papers to disseminate research findings",
                        "children": [],
                    },
                ],
            },
            {"name": "Writing grants", "children": []},
            {"name": "Writing research papers", "children": []},
        ]

        result = _collapse_core_duplicates(tree)
        top_level_names = [str(n.get("name") or "") for n in result]
        assert "Writing grants" not in top_level_names
        assert "Writing research papers" not in top_level_names

        by_name = {str(n.get("name") or ""): n for n in result}
        grants_leaf = None
        papers_leaf = None
        for child in by_name[
            "writing research documents to secure funding and disseminate research findings"
        ].get("children") or []:
            if child.get("name") == "Writing grants to secure research funding":
                grants_leaf = child
            if child.get("name") == "Writing research papers to disseminate research findings":
                papers_leaf = child

        assert grants_leaf is not None
        assert papers_leaf is not None
        assert "Writing grants" in (grants_leaf.get("merged_from") or [])
        assert "Writing research papers" in (papers_leaf.get("merged_from") or [])


# ---------------------------------------------------------------------------
# _tool_agnostic_task_core — core extractor guards and transforms
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Legacy deterministic-core dedup tests; runtime dedup is now LLM-only.")
class TestToolAgnosticTaskCore:
    def test_strips_purpose_tail(self):
        assert (
            _tool_agnostic_task_core("writing grants to secure funding")
            == _tool_agnostic_task_core("writing grants")
        )

    def test_strips_beneficiary_tail(self):
        assert (
            _tool_agnostic_task_core("writing grants for my advisor")
            == _tool_agnostic_task_core("writing grants")
        )

    def test_strips_means_tail(self):
        assert (
            _tool_agnostic_task_core("writing grants via the ORSP portal")
            == _tool_agnostic_task_core("writing grants")
        )

    def test_iterates_over_chained_tails(self):
        # "to X for Y" should collapse to the same core as the bare action+object
        assert (
            _tool_agnostic_task_core("writing grants to secure funding for my lab")
            == _tool_agnostic_task_core("writing grants")
        )

    def test_guard_protects_preparing_for_object(self):
        # "preparing for presentations" — the "for presentations" IS the object;
        # stripping would leave just "preparing", which is one word. Guard kicks in.
        core = _tool_agnostic_task_core("preparing for presentations")
        assert len(core.split()) >= 2

    def test_preparing_presentations_vs_preparing_for_presentations(self):
        # Same action, same object — the "for" is just a grammatical variant.
        transitive = _tool_agnostic_task_core("preparing presentations")
        intransitive = _tool_agnostic_task_core("preparing for presentations")
        assert transitive == intransitive
        # Neither collapses to bare "preparing" — the object is preserved.
        assert transitive != "preparing"
        assert "presentations" in transitive

    def test_verb_for_particle_is_general(self):
        # The rule is general across verbs, not hard-coded to "preparing".
        assert (
            _tool_agnostic_task_core("planning for the retreat")
            == _tool_agnostic_task_core("planning the retreat")
        )

    def test_verb_for_particle_only_at_start(self):
        # The particle rule only drops "for" between the verb and its object —
        # a later "for <beneficiary>" clause is still handled by tail stripping.
        # "writing grants for my advisor" and "writing grants" already collapse
        # via the beneficiary-tail regex, independent of this rule.
        assert (
            _tool_agnostic_task_core("writing grants for my advisor")
            == _tool_agnostic_task_core("writing grants")
        )

    def test_verb_synonym_drafting_to_writing(self):
        assert (
            _tool_agnostic_task_core("drafting a paper")
            == _tool_agnostic_task_core("writing a paper")
        )

    def test_verb_synonym_participating_to_attending(self):
        assert (
            _tool_agnostic_task_core("participating in lab meetings")
            == _tool_agnostic_task_core("attending lab meetings")
        )

    def test_does_not_merge_different_objects(self):
        # Regression guard: different objects must keep different cores.
        assert (
            _tool_agnostic_task_core("reading papers")
            != _tool_agnostic_task_core("writing papers")
        )
        assert (
            _tool_agnostic_task_core("writing proposals")
            != _tool_agnostic_task_core("writing manuscripts")
        )

    def test_empty_input(self):
        assert _tool_agnostic_task_core("") == ""
        assert _tool_agnostic_task_core(None) == ""


# ---------------------------------------------------------------------------
# _pre_dedup_by_core — deterministic dedup pre-pass
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Legacy deterministic-core dedup tests; runtime dedup is now LLM-only.")
class TestPreDedupByCore:
    def test_collapses_screenshot_pairs(self):
        # Pairs reported from the UI screenshot that LLM dedup previously missed.
        tasks = [
            "writing grants to secure funding",
            "writing grants",
            "reading papers to track the field",
            "reading papers",
            "attending lab meetings to sync",
            "attending lab meetings",
            "preparing slides for presentations",
            "preparing slides",
        ]
        kept, merge_map = _pre_dedup_by_core(tasks)
        assert len(kept) == 4
        # Longest variant wins in each pair.
        assert "writing grants to secure funding" in kept
        assert "reading papers to track the field" in kept
        assert "attending lab meetings to sync" in kept
        assert "preparing slides for presentations" in kept
        assert merge_map["writing grants to secure funding"] == ["writing grants"]

    def test_picks_longest_as_winner(self):
        tasks = ["writing grants", "writing grants to secure funding for my lab"]
        kept, merge_map = _pre_dedup_by_core(tasks)
        assert kept == ["writing grants to secure funding for my lab"]
        assert merge_map == {
            "writing grants to secure funding for my lab": ["writing grants"],
        }

    def test_preserves_order_of_first_occurrence(self):
        tasks = ["reading papers", "running experiments", "reading papers to track"]
        kept, _ = _pre_dedup_by_core(tasks)
        # First group anchored by "reading papers" stays before the experiments group
        assert kept.index("reading papers to track") < kept.index("running experiments")

    def test_does_not_over_merge_different_objects(self):
        tasks = ["reading papers", "writing papers"]
        kept, merge_map = _pre_dedup_by_core(tasks)
        assert set(kept) == {"reading papers", "writing papers"}
        assert merge_map == {}

    def test_empty_and_single(self):
        assert _pre_dedup_by_core([]) == ([], {})
        assert _pre_dedup_by_core(["only one"]) == (["only one"], {})


# ---------------------------------------------------------------------------
# _compose_merge_maps — transitive folding across chained passes
# ---------------------------------------------------------------------------

class TestComposeMergeMaps:
    def test_second_pass_absorbs_first_pass_winner(self):
        first = {"writing grants to secure funding": ["writing grants"]}
        # L2 absorbs the L1 winner into a new canonical statement.
        second = {
            "writing grant proposals": ["writing grants to secure funding"],
        }
        composed = _compose_merge_maps(first, second)
        assert "writing grant proposals" in composed
        # Both original strings should now roll up under the new winner.
        assert set(composed["writing grant proposals"]) == {
            "writing grants",
            "writing grants to secure funding",
        }
        # The former L1 winner key is gone (folded up).
        assert "writing grants to secure funding" not in composed

    def test_independent_maps_merge_side_by_side(self):
        first = {"a": ["a1"]}
        second = {"b": ["b1"]}
        composed = _compose_merge_maps(first, second)
        assert composed == {"a": ["a1"], "b": ["b1"]}

    def test_empty_inputs(self):
        assert _compose_merge_maps({}, {}) == {}
        assert _compose_merge_maps({"a": ["x"]}, {}) == {"a": ["x"]}
        assert _compose_merge_maps({}, {"a": ["x"]}) == {"a": ["x"]}

    def test_no_duplicates_in_bucket(self):
        first = {"a": ["x"]}
        second = {"a": ["x", "y"]}
        composed = _compose_merge_maps(first, second)
        assert composed["a"] == ["x", "y"]


# ---------------------------------------------------------------------------
# _collapse_core_duplicates — L3 post-clustering backstop
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Legacy post-cluster core-collapse tests; runtime dedup is now LLM-only.")
class TestCollapseCoreDuplicates:
    def test_parent_child_same_core_collapses(self):
        tree = [{
            "name": "writing grants to secure funding",
            "children": [
                {"name": "writing grants", "children": []},
            ],
        }]
        result = _collapse_core_duplicates(tree)
        assert len(result) == 1
        assert result[0]["name"] == "writing grants to secure funding"
        # Child should be absorbed — recorded in merged_from, removed from children.
        assert "writing grants" in (result[0].get("merged_from") or [])
        assert result[0]["children"] == []

    def test_parent_child_different_core_kept(self):
        tree = [{
            "name": "writing grants",
            "children": [
                {"name": "reviewing budgets", "children": []},
            ],
        }]
        result = _collapse_core_duplicates(tree)
        assert len(result[0]["children"]) == 1

    def test_sibling_duplicates_collapse(self):
        tree = [{
            "name": "writing group",
            "is_group": True,
            "children": [
                {"name": "writing grants to secure funding", "children": []},
                {"name": "writing grants", "children": []},
                {"name": "writing papers", "children": []},
            ],
        }]
        result = _collapse_core_duplicates(tree)
        children = result[0]["children"]
        assert len(children) == 2
        names = {c["name"] for c in children}
        assert "writing grants to secure funding" in names
        assert "writing papers" in names
        winner = next(c for c in children if c["name"] == "writing grants to secure funding")
        assert "writing grants" in (winner.get("merged_from") or [])

    def test_groups_left_alone(self):
        # Two invented group labels that happen to normalize the same must NOT be merged.
        tree = [
            {"name": "planning work", "is_group": True, "children": [
                {"name": "reviewing notes", "children": []},
            ]},
            {"name": "planning work", "is_group": True, "children": [
                {"name": "writing schedules", "children": []},
            ]},
        ]
        result = _collapse_core_duplicates(tree)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# Integration: L1 + L3 alone suffice when LLM returns nothing useful
# ---------------------------------------------------------------------------

@pytest.mark.skip(reason="Legacy deterministic fallback expectations; runtime dedup is now LLM-only.")
class TestDedupEndToEndWithoutLLMHelp:
    def test_screenshot_pairs_dedup_with_empty_llm_response(self):
        tasks = [
            "writing grants to secure funding",
            "writing grants",
            "reading papers to track the field",
            "reading papers",
            "attending lab meetings to sync",
            "attending lab meetings",
            "preparing slides for presentations",
            "preparing slides",
        ]

        # LLM dedup returns singleton entries for every input (no merges).
        # The deterministic pre-pass still collapses the four pairs.
        dedup_response = MagicMock()
        dedup_response.content = json.dumps(
            [{"indices": [i]} for i in range(4)]  # after L1, only 4 tasks remain
        )
        # LLM grouping just flattens the 4 deduped tasks as leaves.
        grouping_tree = [
            {"name": "writing grants to secure funding", "children": []},
            {"name": "reading papers to track the field", "children": []},
            {"name": "attending lab meetings to sync", "children": []},
            {"name": "preparing slides for presentations", "children": []},
        ]
        grouping_response = _make_engine_response(grouping_tree)

        with patch("src.utils.task_hierarchy.get_engine") as mock_get_engine, \
             patch(
                 "src.utils.task_hierarchy.invoke_engine",
                 side_effect=[dedup_response, grouping_response],
             ):
            mock_get_engine.return_value = MagicMock()
            result = organize_tasks(tasks, screen=False)

        names = [n["name"] for n in result]
        assert len(names) == 4
        assert "writing grants to secure funding" in names
        assert "reading papers to track the field" in names
        assert "attending lab meetings to sync" in names
        assert "preparing slides for presentations" in names
