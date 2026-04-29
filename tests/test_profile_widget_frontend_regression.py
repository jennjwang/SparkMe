"""Regression tests for profile-confirm widget handling in the chat frontend.

These are source-level tests over chat.html to prevent regressions where
`profile_confirm_widget` is ignored on full-history replay.
"""

from pathlib import Path
import re


CHAT_TEMPLATE = (
    Path(__file__).resolve().parents[1] / "src" / "web" / "templates" / "chat.html"
)


def _chat_template_text() -> str:
    return CHAT_TEMPLATE.read_text(encoding="utf-8")


class TestProfileWidgetReplayFrontendRegression:
    def test_profile_widget_not_blocked_by_full_history_guard(self):
        text = _chat_template_text()
        # Regression guard: this exact gate prevented profile widget replay on reconnect.
        assert not re.search(
            r"if\s*\(\s*!seenMessageIds\.has\(msg\.id\)\s*&&\s*!fullHistory\s*\)",
            text,
        )

    def test_profile_pending_buffer_not_blocked_by_full_history_guard(self):
        text = _chat_template_text()
        # Regression guard: this gate let interviewer messages bypass buffering
        # during profile confirmation on full-history replay.
        assert not re.search(
            r"if\s*\(\s*profileConfirmPending\s*&&\s*!fullHistory\s*\)",
            text,
        )

    def test_full_history_has_unresolved_profile_widget_replay_logic(self):
        text = _chat_template_text()
        assert "const unresolvedProfileWidgetIds = new Set();" in text
        assert re.search(
            r"const\s+shouldShow\s*=\s*!fullHistory\s*\|\|\s*unresolvedProfileWidgetIds\.has\(msg\.id\)",
            text,
        )


class TestProfileWidgetContinueUxRegression:
    def test_profile_hint_has_looks_good_button(self):
        text = _chat_template_text()
        assert 'id="profile-looks-good-btn"' in text
        assert "Looks good" in text

    def test_looks_good_button_calls_profile_resolve_gate(self):
        text = _chat_template_text()
        assert re.search(
            r"profile-looks-good-btn.*?addEventListener\(\"click\",.*?resolveProfileConfirmGate\(\)",
            text,
            re.DOTALL,
        )

    def test_save_changes_auto_continues_when_profile_confirm_pending(self):
        text = _chat_template_text()
        assert re.search(
            r"if\s*\(\s*profileConfirmPending\s*\)\s*\{\s*resolveProfileConfirmGate\(\);\s*\}",
            text,
        )

    def test_save_changes_hides_profile_when_confirm_not_pending(self):
        text = _chat_template_text()
        assert re.search(
            r"if\s*\(\s*profileConfirmPending\s*\)\s*\{\s*resolveProfileConfirmGate\(\);\s*\}\s*else\s*\{\s*document\s*\.getElementById\(\"profile-panel\"\)\s*\.classList\.remove\(\"visible\"\);\s*document\.getElementById\(\"profile-btn\"\)\.classList\.remove\(\"active\"\);",
            text,
            re.DOTALL,
        )

    def test_profile_resolve_gate_hides_profile_panel(self):
        text = _chat_template_text()
        assert re.search(
            r"function\s+resolveProfileConfirmGate\(\)\s*\{.*?profileConfirmPending\s*=\s*false;.*?profile-panel\"\)\.classList\.remove\(\"visible\"\);.*?profile-btn\"\)\.classList\.remove\(\"active\"\);",
            text,
            re.DOTALL,
        )


class TestFeedbackWidgetSaveRegression:
    def test_feedback_payload_defaults_current_tasks_to_performed(self):
        text = _chat_template_text()
        assert "const checkedTasks = new Set(tasks);" in text


class TestProfileWidgetRoleFirstRenderingRegression:
    def test_profile_confirm_sets_gate_before_reveal(self):
        text = _chat_template_text()
        assert re.search(
            r'if\s*\(msg\.type === "profile_confirm_widget"\)\s*\{.*?if\s*\(!seenMessageIds\.has\(msg\.id\)\s*&&\s*shouldShow\)\s*\{.*?profileConfirmPending\s*=\s*true;.*?showProfilePanelWhenRoleReady\(\);',
            text,
            re.DOTALL,
        )

    def test_role_ready_helper_exists(self):
        text = _chat_template_text()
        assert "function profileHasRole(portrait)" in text
        assert "function profileHasTenure(portrait)" in text
        assert "function profileHasRoleAndTenure(portrait)" in text
        assert "async function showProfilePanelWhenRoleReady()" in text
        assert re.search(
            r"if\s*\(\s*!profileConfirmSnapshot\s*&&\s*profileHasRoleAndTenure\(lastPortrait\)\s*\)",
            text,
        )


class TestProfileBuildingIndicatorLeadInRegression:
    def test_profile_building_indicator_lead_in_helper_exists(self):
        text = _chat_template_text()
        assert "const PROFILE_BUILDING_INDICATOR_MIN_VISIBLE_MS = 320;" in text
        assert "async function ensureProfileBuildingIndicatorLeadIn()" in text

    def test_role_ready_path_waits_for_indicator_before_show(self):
        text = _chat_template_text()
        assert re.search(
            r"showProfilePanelWhenRoleReady\(\)\s*\{.*?await ensureProfileBuildingIndicatorLeadIn\(\);\s*if \(!profileConfirmPending\) return;\s*showProfilePanel\(\);",
            text,
            re.DOTALL,
        )

    def test_profile_panel_reveal_paths_use_indicator_lead_in(self):
        text = _chat_template_text()
        assert text.count("await ensureProfileBuildingIndicatorLeadIn();") >= 3


class TestProfileBuildingIndicatorPlacementRegression:
    def test_pre_confirm_indicator_probe_exists(self):
        text = _chat_template_text()
        assert "let profileConfirmWidgetSeen = false;" in text
        assert "async function maybeShowProfileBuildingIndicatorBeforeConfirm()" in text
        assert "function isFirstTopicCoveredForProfileConfirm(topics)" in text

    def test_poll_triggers_pre_confirm_indicator_probe(self):
        text = _chat_template_text()
        assert re.search(
            r"if\s*\(!fullHistory\)\s*\{\s*maybeShowProfileBuildingIndicatorBeforeConfirm\(\);\s*\}",
            text,
        )

    def test_pre_confirm_probe_replaces_typing_with_profile_indicator(self):
        text = _chat_template_text()
        assert re.search(
            r"maybeShowProfileBuildingIndicatorBeforeConfirm\(\)\s*\{.*?typing-indicator.*?removeTyping\(\);\s*showProfileBuildingIndicator\(\);",
            text,
            re.DOTALL,
        )

    def test_normal_bot_message_clears_stale_profile_indicator(self):
        text = _chat_template_text()
        assert re.search(
            r"if \(appendMessage\(\"bot\", msg\.content, msg\.id\)\) \{.*?if \(!profileConfirmPending\) removeProfileBuildingIndicator\(\);.*?removeTyping\(\);",
            text,
            re.DOTALL,
        )


class TestProfileWidgetFreezeDuringConfirmRegression:
    def test_profile_confirm_snapshot_state_exists(self):
        text = _chat_template_text()
        assert "let profileConfirmSnapshot = null;" in text

    def test_profile_panel_opens_only_after_snapshot_ready(self):
        text = _chat_template_text()
        assert re.search(
            r"async function showProfilePanelWhenRoleReady\(\)\s*\{.*?if\s*\(\s*!profileConfirmSnapshot\s*&&\s*profileHasRoleAndTenure\(lastPortrait\)\s*\)\s*\{.*?profileConfirmSnapshot\s*=\s*JSON\.parse\(\s*JSON\.stringify\(lastPortrait\s*\|\|\s*\{\}\)\s*,?\s*\);.*?showProfilePanel\(\);",
            text,
            re.DOTALL,
        )

    def test_fetch_and_render_profile_uses_snapshot_while_pending(self):
        text = _chat_template_text()
        assert re.search(
            r"if\s*\(\s*profileConfirmPending\s*&&\s*profileConfirmSnapshot\s*\)\s*\{\s*renderProfile\(profileConfirmSnapshot\);\s*return;\s*\}",
            text,
            re.DOTALL,
        )

    def test_resolve_gate_clears_snapshot(self):
        text = _chat_template_text()
        assert re.search(
            r"function\s+resolveProfileConfirmGate\(\)\s*\{.*?profileConfirmPending\s*=\s*false;.*?profileConfirmSnapshot\s*=\s*null;",
            text,
            re.DOTALL,
        )


class TestProfileFieldCapitalizationRegression:
    def test_profile_text_fields_are_capitalized_consistently(self):
        text = _chat_template_text()
        assert "function capitalizeFieldValue(value)" in text
        assert "const displayVal = capitalizeFieldValue(val);" in text

    def test_profile_list_values_render_with_capitalized_first_word(self):
        text = _chat_template_text()
        assert text.count("escapeHtml(capitalizeFieldValue(item))") >= 2

    def test_profile_save_and_add_normalize_field_values(self):
        text = _chat_template_text()
        assert "lastPortrait[key].push(capitalizeFieldValue(val));" in text
        assert (
            "portrait[input.dataset.key] = capitalizeFieldValue(input.value.trim());"
            in text
        )


class TestProfileAutoCondenseRegression:
    def test_profile_condense_helpers_exist(self):
        text = _chat_template_text()
        assert "function condenseRoleText(value)" in text
        assert "function condenseTenureText(value)" in text
        assert "function condensePortraitForDisplay(portrait)" in text

    def test_render_profile_condenses_before_display(self):
        text = _chat_template_text()
        assert "const displayPortrait = condensePortraitForDisplay(portrait);" in text


class TestChatSubmitDebounceRegression:
    def test_send_inflight_state_exists(self):
        text = _chat_template_text()
        assert "let sendInFlight = false;" in text

    def test_submit_guard_prevents_double_send(self):
        text = _chat_template_text()
        assert re.search(
            r"document\.getElementById\(\"chat-form\"\)\.onsubmit\s*=\s*async function \(e\)\s*\{.*?if \(sendInFlight\) return;.*?sendInFlight = true;",
            text,
            re.DOTALL,
        )
        assert re.search(
            r"finally\s*\{.*?sendInFlight\s*=\s*false;\s*\}",
            text,
            re.DOTALL,
        )

    def test_submit_fetch_has_abort_timeout_and_signal(self):
        text = _chat_template_text()
        assert "const controller = new AbortController();" in text
        assert "setTimeout(() => controller.abort(), 20000);" in text
        assert "signal: controller.signal," in text
        assert re.search(
            r"if \(e && e\.name === \"AbortError\"\)\s*\{.*?Request timed out\. You can press Enter again\.",
            text,
            re.DOTALL,
        )
        assert "clearTimeout(timeoutId);" in text


class TestAiUsageWidgetCapitalizationRegression:
    def test_ai_usage_bucket_names_are_capitalized_on_render(self):
        text = _chat_template_text()
        assert (
            "bucketNames = dedupTasks(bucketNames.map((n) => capitalizeFieldValue(n))).slice(0, 4);"
            in text
        )
        assert (
            '<textarea class="aiw-name" data-id="${esc(r.id)}" rows="1">${esc(capitalizeFieldValue(r.name))}</textarea>'
            in text
        )

    def test_ai_usage_submit_text_capitalizes_bucket_names(self):
        text = _chat_template_text()
        assert (
            "return `${capitalizeFieldValue(r.name.trim())} — ${modeText}`;"
            in text
        )


class TestMobileLayoutRegression:
    def test_mobile_profile_drawer_and_backdrop_css_exist(self):
        text = _chat_template_text()
        assert "@media (max-width: 900px)" in text
        assert ".profile-backdrop" in text
        assert "position: fixed;" in text
        assert "transform: translateX(100%);" in text
        assert ".profile-backdrop.visible" in text

    def test_profile_backdrop_dom_node_exists(self):
        text = _chat_template_text()
        assert 'id="profile-backdrop"' in text

    def test_profile_toggle_and_close_update_backdrop_state(self):
        text = _chat_template_text()
        assert re.search(
            r"function toggleProfilePanel\(\)\s*\{.*?profile-backdrop.*?classList\.toggle\(\"visible\", isVisible\);",
            text,
            re.DOTALL,
        )
        assert re.search(
            r"profile-close-btn\".*?classList\.remove\(\"visible\"\);.*?profile-btn\".*?classList\.remove\(\"active\"\);.*?profile-backdrop.*?classList\.remove\(\"visible\"\);",
            text,
            re.DOTALL,
        )

    def test_mobile_avoids_auto_opening_profile_panel(self):
        text = _chat_template_text()
        assert (
            'if (!window.matchMedia("(max-width: 900px)").matches) {'
            in text
        )
        assert (
            'if (window.matchMedia("(max-width: 900px)").matches) {'
            in text
        )


class TestProfileLoadingSignalRegression:
    def test_profile_loading_styles_exist(self):
        text = _chat_template_text()
        assert ".profile-toggle-btn.loading" in text
        assert ".profile-header-badge.loading" in text
        assert "@keyframes profileSpin" in text

    def test_profile_toggle_triggers_loading_fetch_on_open(self):
        text = _chat_template_text()
        assert re.search(
            r"function toggleProfilePanel\(\)\s*\{.*?const isVisible = panel\.classList\.toggle\(\"visible\"\);.*?if \(isVisible\)\s*\{.*?fetchAndRenderProfile\(\{ showLoading: true \}\);",
            text,
            re.DOTALL,
        )

    def test_profile_fetch_can_enable_loading_signal(self):
        text = _chat_template_text()
        assert "function beginProfileLoadingSignal()" in text
        assert re.search(
            r"async function fetchAndRenderProfile\(options = \{\}\)\s*\{.*?const stopLoadingSignal = options\.showLoading\s*\?\s*beginProfileLoadingSignal\(\)\s*:\s*null;",
            text,
            re.DOTALL,
        )

    def test_silent_profile_load_uses_loading_signal(self):
        text = _chat_template_text()
        assert re.search(
            r"async function silentProfileLoad\(\)\s*\{.*?const stopLoadingSignal = beginProfileLoadingSignal\(\);.*?finally\s*\{\s*stopLoadingSignal\(\);\s*\}",
            text,
            re.DOTALL,
        )

    def test_profile_toggle_schedules_autosize_refresh_on_open(self):
        text = _chat_template_text()
        assert re.search(
            r"function toggleProfilePanel\(\)\s*\{.*?if \(isVisible\)\s*\{\s*scheduleProfileInputAutosizeRefresh\(\);\s*fetchAndRenderProfile\(\{ showLoading: true \}\);",
            text,
            re.DOTALL,
        )

    def test_profile_autosize_avoids_hidden_or_collapsed_panel(self):
        text = _chat_template_text()
        assert "function scheduleProfileInputAutosizeRefresh(delayMs = 380)" in text
        assert re.search(
            r"function autosizeProfileInput\(input\)\s*\{.*?!panel\.classList\.contains\(\"visible\"\).*?panel\.getBoundingClientRect\(\)\.width < 220",
            text,
            re.DOTALL,
        )


class TestTimeSplitGroupingFeedbackRegression:
    def test_time_split_includes_grouping_feedback_input_and_button(self):
        text = _chat_template_text()
        assert 'id="tsp-group-feedback-input"' in text
        assert 'id="tsp-regroup-btn"' in text

    def test_regroup_call_sends_grouping_feedback_to_api(self):
        text = _chat_template_text()
        assert re.search(
            r'fetch\("/api/organize-tasks".*?grouping_feedback:\s*feedback',
            text,
            re.DOTALL,
        )

    def test_task_tree_fetch_sends_saved_grouping_feedback(self):
        text = _chat_template_text()
        assert "function portraitGroupingFeedback(portrait)" in text
        assert re.search(
            r'body:\s*JSON\.stringify\(\{\s*session_token:\s*sessionToken,\s*tasks:\s*payload,\s*grouping_feedback:\s*feedback,',
            text,
            re.DOTALL,
        )

    def test_refine_next_persists_grouping_feedback_in_portrait(self):
        text = _chat_template_text()
        assert 'const TASK_GROUPING_FEEDBACK_KEY = "Task Grouping Feedback";' in text
        assert "const normalizedGroupingFeedback =" in text
        assert re.search(
            r"updatedPortrait\[TASK_GROUPING_FEEDBACK_KEY\]\s*=\s*normalizedGroupingFeedback;",
            text,
        )

    def test_profile_panel_hides_internal_task_grouping_tree_field(self):
        text = _chat_template_text()
        assert re.search(
            r'const visibleFields = fields\.filter\(\s*\(f\)\s*=>.*?f\.key !== "Task Grouping Tree"',
            text,
            re.DOTALL,
        )
        assert re.search(
            r'if\s*\(\s*key === "Priority Tasks"\s*\|\|\s*key === "Time Allocation"\s*\|\|\s*key === "Task Grouping Tree"',
            text,
            re.DOTALL,
        )


class TestTaskDedupCoreRegression:
    def test_dedup_uses_core_key_not_raw_string(self):
        text = _chat_template_text()
        assert "function taskCoreKey(t)" in text
        assert re.search(
            r"function\s+dedupKey\(t\)\s*\{\s*return\s+taskCoreKey\(t\);\s*\}",
            text,
            re.DOTALL,
        )

    def test_task_core_key_ignores_objective_and_but_tail(self):
        text = _chat_template_text()
        assert 'const toIdx = s.indexOf(" to ");' in text
        assert "if (toIdx > 0) s = s.slice(0, toIdx).trim();" in text
        assert 'const butIdx = s.indexOf(" but ");' in text
        assert "if (butIdx > 0) s = s.slice(0, butIdx).trim();" in text

    def test_task_core_key_normalizes_simple_verb_variants(self):
        text = _chat_template_text()
        assert 'reading: "read"' in text
        assert 'writing: "write"' in text
        assert 'running: "run"' in text
        assert "if (t.endsWith(\"ing\") && t.length > 5) return t.slice(0, -3);" in text


class TestTaskWidgetLoadLatencyRegression:
    def test_initial_task_page_renders_before_background_work(self):
        text = _chat_template_text()
        start = text.index("async function showTaskValidationWidget()")
        end = text.index("async function showAiTaskWidget()", start)
        block = text[start:end]

        assert "await Promise.all([" not in block
        assert "Promise.race([fetchAttentionChecks()" not in block
        assert "const attnTimeout" not in block
        assert "await fetchUntilFull(_PAGE_SIZE);" in block
        assert block.index("await fetchUntilFull(_PAGE_SIZE);") < block.index("showBatch(0);")
        assert block.index("showBatch(0);") < block.index("fetchAttentionChecks();")
        assert block.index("showBatch(0);") < block.index("_fetchAiDistractors();")
