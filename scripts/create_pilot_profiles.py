"""
create_pilot_profiles.py — Generate simulation profile files for pilot users.

Reads each pilot user's user_portrait.json and chat history logs, then writes
three files into data/sample_user_profiles/{user_id}/:

  {user_id}_bio_notes.md        — rich bio derived from portrait + transcript
  {user_id}_topics_filled.json  — structured ground truth (intake topic format)
  conversation.md               — LLM-derived style guide from real utterances

Usage
-----
# All pilot users
python scripts/create_pilot_profiles.py

# Specific users
python scripts/create_pilot_profiles.py --user-ids TZgLroIuAN_47_xClBmRqQ seQCtyofmgnnQCZWjS-VWQ

# Preview without writing
python scripts/create_pilot_profiles.py --dry-run

# Force-regenerate conversation.md even if it already exists
python scripts/create_pilot_profiles.py --force
"""

import argparse
import json
import os
import re
import sys
import textwrap

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv(os.path.join(_root, ".env"), override=True)

from openai import OpenAI
from src.utils.transcript_task_derivation import (
    derive_tasks_from_latest_valid_transcript,
    load_chat_history_messages,
)

PILOT_DIR = os.path.join(_root, "pilot")
PROFILES_DIR = os.path.join(_root, "data", "sample_user_profiles")

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


# ---------------------------------------------------------------------------
# Chat log parsing
# ---------------------------------------------------------------------------

_SESSION_RE = re.compile(r"^session_(\d+)$")


def _extract_user_utterances(pilot_dir: str, user_id: str) -> list[str]:
    """Return all user turns across all sessions, in order."""
    logs_dir = os.path.join(pilot_dir, user_id, "execution_logs")
    if not os.path.isdir(logs_dir):
        return []

    utterances = []
    sessions = []
    for session in os.listdir(logs_dir):
        match = _SESSION_RE.match(session)
        if not match:
            continue
        sessions.append((int(match.group(1)), session))
    sessions.sort(key=lambda x: x[0])

    for _, session in sessions:
        log_path = os.path.join(logs_dir, session, "chat_history.log")
        if not os.path.exists(log_path):
            continue
        try:
            messages = load_chat_history_messages(log_path)
        except OSError:
            continue
        for msg in messages:
            if msg.get("speaker") == "User":
                text = str(msg.get("content", "")).strip()
                if text:
                    utterances.append(text)

    return [u for u in utterances if u]


# ---------------------------------------------------------------------------
# LLM-based style guide generation
# ---------------------------------------------------------------------------

_STYLE_PROMPT = textwrap.dedent("""\
    Below are verbatim responses a person gave during a work interview.
    Analyse these utterances and produce a concise Conversational Style Guide
    that would allow someone else to faithfully impersonate this speaker.

    Focus on:
    1. Tone & register (formal/casual, hedging language, emotional expressiveness)
    2. Typical response length and structure
    3. Recurring phrases, verbal tics, filler words, or distinctive habits
    4. How they handle uncertainty or vague questions
    5. Energy and engagement level

    Include concrete examples quoted directly from the utterances.
    End with a short "To replicate this speaker" checklist.

    ## Utterances
    {utterances}
""")


def _generate_conversation_style(utterances: list[str], model: str = "gpt-4.1-mini") -> str:
    if not utterances:
        return _GENERIC_CONVERSATION_STYLE

    sample = utterances[:40]  # cap at 40 turns to stay within token budget
    block = "\n\n".join(f'"{u}"' for u in sample)
    prompt = _STYLE_PROMPT.format(utterances=block)

    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1200,
    )
    return response.choices[0].message.content.strip()


_GENERIC_CONVERSATION_STYLE = textwrap.dedent("""\
    # Conversational Style Guide

    ## Tone & Register
    - Natural, professional tone appropriate for a work interview.
    - Direct and informative — answers questions without excessive hedging.

    ## Response Length & Structure
    - Typical answers are 2–4 sentences.
    - Uses complete sentences; avoids bullet-point style in speech.

    ## Verbal Habits
    - May use phrases like "I think", "I'd say", or "generally speaking".

    ## Engagement
    - Answers the question asked, nothing more.
    - Does not volunteer tangential details unprompted.
""")


# ---------------------------------------------------------------------------
# Bio notes generation
# ---------------------------------------------------------------------------

_BIO_PROMPT = textwrap.dedent("""\
    Below is a structured user portrait (JSON) and a sample of verbatim
    interview utterances for a real person. Write a rich set of bullet-point
    biographical notes that captures:
    - Their role, seniority, and domain
    - Their actual work rhythm and time allocation
    - Concrete tasks and how they talk about them
    - Pain points, bright spots, and motivations
    - Any distinctive opinions, habits, or personal details revealed in speech

    Write each note as a single bullet starting with "- ".
    Stay factual — do not invent details not present in the source material.
    Aim for 20–40 bullets.

    ## Portrait (JSON)
    {portrait}

    ## Sample utterances
    {utterances}
""")


def _generate_bio_notes(portrait: dict, utterances: list[str], model: str = "gpt-4.1-mini") -> str:
    sample = utterances[:30]
    ublock = "\n\n".join(f'"{u}"' for u in sample)
    prompt = _BIO_PROMPT.format(
        portrait=json.dumps(portrait, indent=2),
        utterances=ublock,
    )
    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1500,
    )
    return "# Biographical Notes\n\n" + response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# topics_filled.json (no LLM needed — straight from portrait)
# ---------------------------------------------------------------------------

def _clean_task_list(tasks: list[str]) -> list[str]:
    return [t for t in tasks if isinstance(t, str) and t.strip()]


def _portrait_to_topics_filled(
    portrait: dict,
    task_inventory_notes: list[str] | None = None,
) -> list:
    role = portrait.get("Functional Role", "")
    seniority = portrait.get("Seniority", "")
    rhythm = portrait.get("Work Rhythm", "")
    tasks = _clean_task_list(portrait.get("Task Inventory", []))
    seeded_tasks = _clean_task_list(task_inventory_notes if task_inventory_notes is not None else tasks)
    motivations = [m for m in portrait.get("Motivations and Goals", []) if m and m.strip()]

    time_notes = [rhythm] if rhythm else ["Time allocation not specified in detail."]
    priority_notes = motivations if motivations else (tasks[:2] if tasks else ["Not specified."])

    return [
        {
            "topic": "Role and Context",
            "subtopics": [
                {
                    "subtopic_id": "1.1",
                    "subtopic_description": "Their role or title",
                    "notes": [role] if role else ["Not specified."],
                },
                {
                    "subtopic_id": "1.2",
                    "subtopic_description": "Their tenure in the role",
                    "notes": [seniority] if seniority else ["Not specified."],
                },
                {
                    "subtopic_id": "1.3",
                    "subtopic_description": "Their domain or industry",
                    "notes": [role] if role else ["Not specified."],
                },
            ],
        },
        {
            "topic": "Task Inventory",
            "subtopics": [
                {
                    "subtopic_id": "2.1",
                    "subtopic_description": "Breadth: the list of tasks performed in a typical week",
                    "notes": seeded_tasks if seeded_tasks else ["Not specified."],
                },
                {
                    "subtopic_id": "2.2",
                    "subtopic_description": "Time allocation: how much time spent on each task",
                    "notes": time_notes,
                },
                {
                    "subtopic_id": "2.3",
                    "subtopic_description": "Priority tasks: the tasks considered most important",
                    "notes": priority_notes,
                },
            ],
        },
    ]


# ---------------------------------------------------------------------------
# Main profile creation
# ---------------------------------------------------------------------------

def create_profile(
    user_id: str,
    pilot_dir: str,
    profiles_dir: str,
    dry_run: bool = False,
    force: bool = False,
    model: str = "gpt-4.1-mini",
    task_source: str = "transcript",
    task_derivation_model: str = "gpt-4.1-mini",
) -> bool:
    portrait_path = os.path.join(pilot_dir, user_id, "user_portrait.json")
    if not os.path.exists(portrait_path):
        print(f"  [SKIP] No user_portrait.json for {user_id}")
        return False

    with open(portrait_path) as f:
        portrait = json.load(f)

    utterances = _extract_user_utterances(pilot_dir, user_id)
    portrait_tasks = _clean_task_list(portrait.get("Task Inventory", []))
    task_count = len(portrait_tasks)
    out_dir = os.path.join(profiles_dir, user_id)

    if dry_run:
        print(f"  [DRY-RUN] {user_id}  ({task_count} tasks, {len(utterances)} utterances)")
        return True

    os.makedirs(out_dir, exist_ok=True)

    # Always regenerate bio_notes and topics_filled
    print(f"  Generating bio notes for {user_id} ...")
    bio_notes = _generate_bio_notes(portrait, utterances, model=model)
    with open(os.path.join(out_dir, f"{user_id}_bio_notes.md"), "w") as f:
        f.write(bio_notes)

    transcript_task_artifact = None
    seeded_tasks = portrait_tasks
    if task_source == "transcript":
        transcript_task_artifact = derive_tasks_from_latest_valid_transcript(
            pilot_dir=pilot_dir,
            user_id=user_id,
            client=_get_client(),
            derivation_model=task_derivation_model,
            organizer_model=task_derivation_model,
        )
        derived_tasks = _clean_task_list(transcript_task_artifact.get("derived_tasks", []))
        if derived_tasks:
            seeded_tasks = derived_tasks
        else:
            reason = transcript_task_artifact.get("reason") or "unknown"
            print(
                f"  [WARN] Transcript task derivation failed for {user_id} "
                f"({reason}); using portrait Task Inventory."
            )

        transcript_task_artifact = {
            **transcript_task_artifact,
            "task_source": task_source,
            "fallback_to_portrait": not bool(derived_tasks),
            "portrait_task_count": len(portrait_tasks),
            "seeded_task_count": len(seeded_tasks),
        }

    topics_filled = _portrait_to_topics_filled(portrait, task_inventory_notes=seeded_tasks)
    with open(os.path.join(out_dir, f"{user_id}_topics_filled.json"), "w") as f:
        json.dump(topics_filled, f, indent=2)

    if transcript_task_artifact is not None:
        artifact_path = os.path.join(out_dir, f"{user_id}_derived_tasks_from_transcript.json")
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(transcript_task_artifact, f, indent=2, sort_keys=True)

    # Regenerate conversation.md if missing or --force
    conv_path = os.path.join(out_dir, "conversation.md")
    if force or not os.path.exists(conv_path):
        print(f"  Generating conversation style for {user_id} ({len(utterances)} utterances) ...")
        style = _generate_conversation_style(utterances, model=model)
        with open(conv_path, "w") as f:
            f.write(style)
    else:
        print(f"  Skipping conversation.md (already exists; use --force to regenerate)")

    print(
        f"  [OK] {user_id}  ({len(seeded_tasks)} seeded tasks from {task_source}, "
        f"{len(utterances)} utterances)"
    )
    return True


def list_pilot_users(pilot_dir: str) -> list[str]:
    users_json = os.path.join(pilot_dir, "users.json")
    if os.path.exists(users_json):
        with open(users_json) as f:
            return list(json.load(f).keys())
    return [
        d for d in os.listdir(pilot_dir)
        if os.path.isdir(os.path.join(pilot_dir, d))
        and os.path.exists(os.path.join(pilot_dir, d, "user_portrait.json"))
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Generate simulation profiles from pilot data"
    )
    parser.add_argument("--user-ids", nargs="*",
                        help="Pilot user IDs to process (default: all)")
    parser.add_argument("--pilot-dir", default=PILOT_DIR)
    parser.add_argument("--profiles-dir", default=PROFILES_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without writing files")
    parser.add_argument("--force", action="store_true",
                        help="Regenerate conversation.md even if it already exists")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="OpenAI model for style generation (default: gpt-4.1-mini)")
    parser.add_argument(
        "--task-source",
        choices=["transcript", "portrait"],
        default="transcript",
        help="Seed Task Inventory from transcript-derived tasks or portrait tasks (default: transcript)",
    )
    parser.add_argument(
        "--task-derivation-model",
        default="gpt-4.1-mini",
        help="OpenAI model for transcript task derivation (default: gpt-4.1-mini)",
    )
    args = parser.parse_args()

    user_ids = args.user_ids or list_pilot_users(args.pilot_dir)
    if not user_ids:
        print("No pilot users found.")
        return

    print(f"Processing {len(user_ids)} pilot user(s)  →  {args.profiles_dir}/\n")
    ok = 0
    for uid in user_ids:
        if create_profile(uid, args.pilot_dir, args.profiles_dir,
                          dry_run=args.dry_run, force=args.force, model=args.model,
                          task_source=args.task_source,
                          task_derivation_model=args.task_derivation_model):
            ok += 1

    print(f"\nDone: {ok}/{len(user_ids)} profiles {'(dry-run)' if args.dry_run else 'written'}.")


if __name__ == "__main__":
    main()
