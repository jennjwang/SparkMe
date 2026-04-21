import importlib.util
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils.transcript_task_derivation import (
    deduplicate_task_variants,
    parse_chat_history_lines,
    select_latest_valid_chat_history_log,
)


def _load_create_pilot_profiles_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "create_pilot_profiles.py"
    spec = importlib.util.spec_from_file_location("create_pilot_profiles_local", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cpp = _load_create_pilot_profiles_module()


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _notes_for(topics: list, subtopic_id: str) -> list[str]:
    for topic in topics:
        for subtopic in topic.get("subtopics", []):
            if subtopic.get("subtopic_id") == subtopic_id:
                return subtopic.get("notes", [])
    raise AssertionError(f"Missing subtopic_id={subtopic_id}")


def test_select_latest_valid_transcript_session(tmp_path: Path):
    pilot_dir = tmp_path / "pilot"
    user_id = "user_123"

    # session_2 is newest but malformed (no user turns)
    _write(
        pilot_dir / user_id / "execution_logs" / "session_2" / "chat_history.log",
        "2026-04-14 10:00:00,000 - INFO - Interviewer: Hello\n",
    )
    # session_1 is valid and should be selected
    _write(
        pilot_dir / user_id / "execution_logs" / "session_1" / "chat_history.log",
        (
            "2026-04-14 09:00:00,000 - INFO - Interviewer: What do you do?\n"
            "2026-04-14 09:00:01,000 - INFO - User: I run experiments.\n"
        ),
    )
    # session_0 exists but older
    _write(
        pilot_dir / user_id / "execution_logs" / "session_0" / "chat_history.log",
        (
            "2026-04-14 08:00:00,000 - INFO - Interviewer: Role?\n"
            "2026-04-14 08:00:01,000 - INFO - User: Research assistant.\n"
        ),
    )

    selected = select_latest_valid_chat_history_log(str(pilot_dir), user_id)
    assert selected is not None
    assert selected["session_name"] == "session_1"
    assert selected["log_path"].endswith("session_1/chat_history.log")


def test_parse_chat_history_multiline_user_message():
    lines = [
        "2026-04-14 09:00:00,000 - INFO - Interviewer: What are your main tasks?\n",
        "2026-04-14 09:00:01,000 - INFO - User: Reading papers\n",
        "\n",
        "and writing code\n",
        "2026-04-14 09:00:05,000 - INFO - Interviewer: Anything else?\n",
    ]
    messages = parse_chat_history_lines(lines)

    assert len(messages) == 3
    assert messages[1]["speaker"] == "User"
    assert messages[1]["content"] == "Reading papers\n\nand writing code"


def test_deduplicate_task_variants_collapses_simple_verb_form_duplicates():
    tasks = [
        "Read academic papers to inform research",
        "Reading academic papers",
        "Write scientific papers to publish findings",
    ]
    deduped = deduplicate_task_variants(tasks)
    assert deduped == [
        "Read academic papers to inform research",
        "Write scientific papers to publish findings",
    ]


def test_topics_filled_replaces_task_inventory_notes_only():
    portrait = {
        "Functional Role": "PhD student",
        "Seniority": "Year 3",
        "Work Rhythm": "Most time on experiments.",
        "Task Inventory": ["portrait task A", "portrait task B"],
        "Motivations and Goals": [],
    }
    topics = cpp._portrait_to_topics_filled(
        portrait,
        task_inventory_notes=["derived task X", "derived task Y"],
    )

    assert _notes_for(topics, "2.1") == ["derived task X", "derived task Y"]
    # 2.2 and 2.3 remain portrait-driven (existing behavior)
    assert _notes_for(topics, "2.2") == ["Most time on experiments."]
    assert _notes_for(topics, "2.3") == ["portrait task A", "portrait task B"]


def test_create_profile_falls_back_to_portrait_tasks_when_derivation_empty(tmp_path: Path):
    pilot_dir = tmp_path / "pilot"
    profiles_dir = tmp_path / "profiles"
    user_id = "user_fallback"

    portrait = {
        "Functional Role": "Researcher",
        "Seniority": "2 years",
        "Work Rhythm": "Split across coding and writing.",
        "Task Inventory": ["portrait task A", "portrait task B"],
        "Motivations and Goals": ["Publish strong work."],
    }
    _write(pilot_dir / user_id / "user_portrait.json", json.dumps(portrait))
    _write(
        pilot_dir / user_id / "execution_logs" / "session_1" / "chat_history.log",
        (
            "2026-04-14 09:00:00,000 - INFO - Interviewer: Main tasks?\n"
            "2026-04-14 09:00:01,000 - INFO - User: I code and write.\n"
        ),
    )

    with patch.object(cpp, "_get_client", return_value=MagicMock()), \
         patch.object(cpp, "_generate_bio_notes", return_value="# Biographical Notes\n\n- note"), \
         patch.object(cpp, "_generate_conversation_style", return_value="# style"), \
         patch.object(
             cpp,
             "derive_tasks_from_latest_valid_transcript",
             return_value={
                 "ok": False,
                 "reason": "empty_task_extraction",
                 "derived_tasks": [],
                 "raw_extracted_tasks": [],
                 "screened_task_tree": [],
             },
         ):
        ok = cpp.create_profile(
            user_id=user_id,
            pilot_dir=str(pilot_dir),
            profiles_dir=str(profiles_dir),
            dry_run=False,
            force=True,
            model="gpt-4.1-mini",
            task_source="transcript",
            task_derivation_model="gpt-4.1-mini",
        )

    assert ok is True

    topics_path = profiles_dir / user_id / f"{user_id}_topics_filled.json"
    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    assert _notes_for(topics, "2.1") == ["portrait task A", "portrait task B"]

    artifact_path = profiles_dir / user_id / f"{user_id}_derived_tasks_from_transcript.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["fallback_to_portrait"] is True
    assert artifact["reason"] == "empty_task_extraction"


def test_create_profile_uses_mocked_transcript_derived_tasks(tmp_path: Path):
    pilot_dir = tmp_path / "pilot"
    profiles_dir = tmp_path / "profiles"
    user_id = "user_integration"

    portrait = {
        "Functional Role": "Researcher",
        "Seniority": "2 years",
        "Work Rhythm": "Mostly experiments.",
        "Task Inventory": ["portrait task A", "portrait task B"],
        "Motivations and Goals": ["Deliver results."],
    }
    _write(pilot_dir / user_id / "user_portrait.json", json.dumps(portrait))
    _write(
        pilot_dir / user_id / "execution_logs" / "session_1" / "chat_history.log",
        (
            "2026-04-14 09:00:00,000 - INFO - Interviewer: Main tasks?\n"
            "2026-04-14 09:00:01,000 - INFO - User: I run experiments and analyze outputs.\n"
        ),
    )

    mocked_tasks = [
        "running experiments to test hypotheses",
        "analyzing experiment outputs to evaluate model behavior",
    ]
    with patch.object(cpp, "_get_client", return_value=MagicMock()), \
         patch.object(cpp, "_generate_bio_notes", return_value="# Biographical Notes\n\n- note"), \
         patch.object(cpp, "_generate_conversation_style", return_value="# style"), \
         patch.object(
             cpp,
             "derive_tasks_from_latest_valid_transcript",
             return_value={
                 "ok": True,
                 "reason": "",
                 "derived_tasks": mocked_tasks,
                 "raw_extracted_tasks": mocked_tasks,
                 "screened_task_tree": [],
                 "source_session": "session_1",
                 "source_chat_history_log": "pilot/user_integration/execution_logs/session_1/chat_history.log",
             },
         ):
        ok = cpp.create_profile(
            user_id=user_id,
            pilot_dir=str(pilot_dir),
            profiles_dir=str(profiles_dir),
            dry_run=False,
            force=True,
            model="gpt-4.1-mini",
            task_source="transcript",
            task_derivation_model="gpt-4.1-mini",
        )

    assert ok is True

    topics_path = profiles_dir / user_id / f"{user_id}_topics_filled.json"
    topics = json.loads(topics_path.read_text(encoding="utf-8"))
    assert _notes_for(topics, "2.1") == mocked_tasks

    artifact_path = profiles_dir / user_id / f"{user_id}_derived_tasks_from_transcript.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["derived_tasks"] == mocked_tasks
    assert artifact["fallback_to_portrait"] is False
