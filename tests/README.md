# Tests

## Running

```bash
python -m pytest tests/ -v
```

Specific file:
```bash
python -m pytest tests/test_task_hierarchy.py -v
```

No API keys needed — all LLM calls are mocked with `unittest.mock.patch`.

---

## Test files

### `test_task_merge.py` — dedup semantics for Task Inventory

Tests the `_task_core` normalizer and `merge_task_inventories` utility, which
document the intended dedup semantics for Task Inventory lists. The live
pipeline delegates merging to the LLM (via the `{prior_tasks}` prompt block),
but these helpers serve as a reference implementation and catch regressions in
the normalization logic.

**`TestTaskCore`** — normalizes a task string to its action+object core for matching:

| Test | What it checks |
|------|---------------|
| `test_strips_purpose_clause` | `"running experiments to publish a paper"` → `"running experiments"` |
| `test_no_purpose_clause` | Tasks without `" to "` are returned as-is |
| `test_normalizes_case` | Input is lowercased before matching |
| `test_strips_whitespace` | Leading/trailing whitespace is trimmed |
| `test_keeps_first_to_segment_only` | Only the first `" to "` split is used; later ones are ignored |
| `test_empty_string` | Empty input returns empty string without error |

**`TestMergeTaskInventories`** — merging new tasks with tasks from a prior session:

| Test | What it checks |
|------|---------------|
| `test_no_prior_returns_new` | If there are no prior tasks, returns the new list unchanged |
| `test_prior_task_preserved_when_absent_from_new` | A prior task not covered by any new task is appended |
| `test_prior_task_dropped_when_covered_by_new` | Same action+object but different purpose → new version replaces prior |
| `test_prefix_substring_match_covers_prior` | `"attending meetings"` core matches `"attending meetings to exchange updates"` |
| `test_cadence_word_ignored_in_matching` | `"weekly"` is stripped before matching, so `"attending weekly project meetings"` covers `"attending project meetings"` |
| `test_distinct_action_objects_both_preserved` | Tasks with different action+object are treated as distinct and both kept |
| `test_multiple_prior_partially_covered` | Only the uncovered prior tasks are appended; covered ones are dropped |
| `test_empty_new_preserves_all_prior` | If new task list is empty, all prior tasks are preserved |
| `test_case_insensitive_matching` | Matching ignores case differences |
| `test_real_world_near_duplicate_weekly_cadence` | Regression: `"participating in weekly lab meetings"` and `"participating in lab meetings"` resolve to the same core after cadence stripping |
| `test_all_tasks_from_conversation_captured` | 11 distinct tasks from a real conversation are all preserved |

---

### `test_task_hierarchy.py` — tree organizer for the task widget

Tests the helpers inside `src/utils/task_hierarchy.py` that transform a flat
task list into a two-level hierarchy for the time-split widget and profile
panel. LLM calls are mocked so tests run offline.

**`TestNorm`** — string normalization used throughout the module:

| Test | What it checks |
|------|---------------|
| `test_lowercases` | Output is always lowercase |
| `test_collapses_whitespace` | Multiple spaces are collapsed to one |
| `test_empty` | Empty string returns empty string |
| `test_none` | `None` input returns empty string without error |

**`TestExtractJson`** — parses the LLM's `<hierarchy>…</hierarchy>` response:

| Test | What it checks |
|------|---------------|
| `test_parses_hierarchy_tags` | Extracts JSON from `<hierarchy>` tags |
| `test_strips_markdown_fences` | Strips ` ```json ``` ` fences before parsing |
| `test_falls_back_to_raw_json` | If no tags are present, parses the whole string as JSON |

**`TestCollectCovered`** — the safety net that detects which input tasks the LLM accounted for:

| Test | What it checks |
|------|---------------|
| `test_flat_leaf_covered` | A non-group leaf node's name is treated as covered |
| `test_merged_from_covered` | Tasks listed in `merged_from` are also treated as covered |
| `test_group_name_not_covered` | Invented group labels (`is_group: true`) are NOT treated as input tasks |
| `test_nested_children_covered` | Children inside a group node are collected recursively |

**`TestSanitize`** — coerces raw LLM output into the shape the frontend expects:

| Test | What it checks |
|------|---------------|
| `test_basic_leaf_passthrough` | A well-formed leaf node passes through unchanged |
| `test_restores_verbatim_casing` | LLM title-casing is undone using the verbatim lookup |
| `test_single_child_group_demoted` | An invented group with only one child is dissolved; the child is promoted to top level |
| `test_two_child_group_kept` | A group with two or more children is kept as-is |
| `test_empty_group_name_skipped` | Nodes with empty names are dropped |
| `test_non_dict_entries_skipped` | Non-dict entries in the node list are silently skipped |
| `test_merged_from_verbatim_restored` | Verbatim casing is also restored on `merged_from` task strings |

**`TestOrganizeTasks`** — end-to-end `organize_tasks` with mocked LLM:

| Test | What it checks |
|------|---------------|
| `test_flat_list_passthrough` | A single-item list bypasses the LLM and returns a flat node |
| `test_empty_input` | An empty task list returns an empty list without error |
| `test_llm_response_used` | When the LLM returns a valid hierarchy, it is used |
| `test_safety_net_appends_uncovered_task` | If the LLM drops a task, it is appended as a top-level leaf |
| `test_dedup_via_merged_from` | A task absorbed by dedup appears in the survivor's `merged_from` list |
| `test_fallback_on_llm_failure` | If `get_engine` raises, returns a flat fallback list |
| `test_fallback_on_bad_llm_json` | If the LLM returns unparseable output, returns a flat fallback list |
| `test_child_not_duplicated_at_root` | A task that appears as both a group child and a root node is deduplicated at root level |

---

### `test_portrait_from_transcript.py` — portrait extraction from real pilot transcripts

End-to-end tests using actual conversations from the pilot study
(`pilot/<user_id>/execution_logs/session_1/chat_history.log`). Each test
loads a real transcript, constructs the extraction prompt, and asserts that
the prompt contains the right content and the portrait post-processing
works correctly. LLM calls are mocked with realistic golden outputs taken
from the corresponding `pilot/<user_id>/user_portrait.json`.

**`TestPromptConstruction`** — the prompt includes the transcript and prior tasks:

| Test | What it checks |
|------|---------------|
| `test_transcript_included_in_prompt` | The full conversation text appears in the formatted prompt |
| `test_prior_tasks_included_in_prompt` | Non-empty prior task list appears in the `{prior_tasks}` block |
| `test_empty_prior_tasks_placeholder` | When there are no prior tasks, the block shows `(none)` |

**`TestCsPhDTranscript`** — CS PhD student session (`61nehu8pvOOrH7Q3Rqoo6w`):

| Test | What it checks |
|------|---------------|
| `test_all_explicitly_mentioned_tasks_extracted` | All tasks the user explicitly named in the transcript appear in the portrait's Task Inventory |
| `test_no_tasks_without_user_mention` | The portrait doesn't invent tasks the user never described |
| `test_time_allocation_captured` | Percentage breakdowns stated by the user are captured in Time Allocation |

**`TestProfessorTranscript`** — Assistant professor session (`RAnWnbsJWkJYiGYGlvIytg`):

| Test | What it checks |
|------|---------------|
| `test_teaching_tasks_present` | Teaching-specific tasks (creating exercises, grading) appear in inventory |
| `test_research_tasks_present` | Research tasks (leading projects, analysis, writing, presenting) appear |
| `test_student_meeting_task_present` | Student advising/meeting tasks are captured |
| `test_prior_tasks_preserved` | Tasks confirmed in a prior session and absent from this one are retained in the merged output |

---

### `test_widget_triggers.py` — interviewer widget firing logic

Tests that the three UI widgets (profile-confirm, time-split, feedback) fire at
the right points in the interview. Uses a real `InterviewTopicManager` built
from `configs/topics_intake.json` so coverage checks run against genuine
subtopic objects, but all LLM and I/O calls are mocked.

**`TestProfileConfirmWidget`** — unit tests on `trigger_profile_confirm_widget`:

| Test | What it checks |
|------|---------------|
| `test_emits_message_on_first_call` | Calling the trigger emits a `PROFILE_CONFIRM_WIDGET` message |
| `test_idempotent_second_call_ignored` | A second call is a no-op; only one widget message is emitted |
| `test_sets_sent_flag` | The `_profile_confirm_widget_sent` flag is set after the first call |

**`TestFeedbackWidget`** — unit tests on `trigger_feedback_widget`:

| Test | What it checks |
|------|---------------|
| `test_emits_message` | Trigger emits a `FEEDBACK_WIDGET` message |
| `test_idempotent` | Second call is a no-op |
| `test_sets_sent_flag` | `_feedback_widget_sent` flag is set |

**`TestHandleResponseProfileConfirm`** — integration: `_handle_response` fires the profile widget at the right time:

| Test | What it checks |
|------|---------------|
| `test_fires_when_first_topic_covered` | Widget fires when the first topic's subtopics are all marked covered |
| `test_does_not_fire_when_first_topic_not_covered` | Widget doesn't fire when coverage is incomplete |
| `test_widget_appears_before_conversation_message` | The widget message is inserted before the interviewer's next conversation message |
| `test_fires_only_once_across_multiple_calls` | Widget fires at most once even if `_handle_response` is called multiple times |

**`TestTimeSplitWidget`** — integration: time-allocation subtopic triggers the time-split widget:

| Test | What it checks |
|------|---------------|
| `test_fires_for_time_allocation_subtopic` | `_handle_response` with subtopic `2.3` emits a `TIME_SPLIT_WIDGET` |
| `test_fires_only_once` | Widget fires at most once for repeated calls on the same subtopic |
| `test_does_not_fire_for_other_subtopics` | Non-time-allocation subtopics don't trigger the widget |

**`TestFeedbackWidgetSafetyNet`** — integration: safety net fires the feedback widget when the LLM ends the session without using `end_conversation`:

| Test | What it checks |
|------|---------------|
| `test_safety_net_fires_when_all_covered_no_end_conversation` | When all topics are covered and the LLM uses `respond_to_user` for goodbye, the safety net fires the feedback widget |
| `test_safety_net_does_not_fire_when_topics_incomplete` | Safety net doesn't fire mid-session when topics are still open |
| `test_safety_net_does_not_fire_when_session_already_ending` | Safety net is a no-op if the session is already in its ending flow |

**`TestEndConversationTool`** — integration: `end_conversation` tool in LLM output triggers the feedback widget:

| Test | What it checks |
|------|---------------|
| `test_end_conversation_triggers_feedback_widget` | When the LLM emits `<end_conversation>`, the feedback widget fires |

---

### `test_feedback_widget_api.py` — feedback widget API flow

Tests API-level behavior for end-of-session feedback: route dispatch, payload
validation, and widget replay on reconnect.

**`TestEndSessionApi`** — `/api/end-session` dispatch behavior:

| Test | What it checks |
|------|---------------|
| `test_end_session_dispatches_feedback_widget_on_session_loop` | With a session loop, route uses `loop.call_soon_threadsafe(trigger_feedback_widget)` |
| `test_end_session_calls_trigger_directly_without_loop` | Without a loop, route calls `trigger_feedback_widget()` directly |

**`TestSubmitFeedbackApi`** — `/api/submit-feedback` behavior:

| Test | What it checks |
|------|---------------|
| `test_submit_feedback_rejects_missing_feedback_payload` | Rejects invalid request when `feedback` is missing |
| `test_submit_feedback_dispatches_on_session_loop` | With a loop, route dispatches `submit_feedback` via `call_soon_threadsafe` |
| `test_submit_feedback_calls_submit_directly_without_loop` | Without a loop, route calls `submit_feedback(feedback)` directly |

**`TestGetMessagesReplayForWidgets`** — `/api/get-messages?full=true` replay semantics:

| Test | What it checks |
|------|---------------|
| `test_full_history_replays_feedback_and_profile_widgets` | Full-history replay includes `feedback_widget` and `profile_confirm_widget`, while excluding `time_split_widget` |

---

### `conftest.py` — shared fixtures

| Fixture | What it provides |
|---------|-----------------|
| `topic_manager` | A real `InterviewTopicManager` loaded from `configs/topics_intake.json` |
| `fake_session` | A `MagicMock` session with real topic state, a stub memory bank, and side-effected trigger methods that actually write to `chat_history` |
| `interviewer` | A real `Interviewer` instance with its engine mocked (no API calls); `call_engine_async` is an `AsyncMock` |
| `silence_session_logger` (autouse) | Patches `SessionLogger.log_to_file` so tests don't raise errors from missing log directories |
