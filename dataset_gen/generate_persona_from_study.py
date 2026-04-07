"""
Generate simulator profiles (topics_filled.json + bio_notes.md) from real
user-study data (memory banks and chat histories) instead of CSV metadata.

Usage:
    python dataset_gen/generate_persona_from_study.py \
        --study-dir user_study \
        --output-dir data/sample_user_profiles

    # Process specific users
    python dataset_gen/generate_persona_from_study.py \
        --user-ids 1T8lGuWK6w-0q4S-s2_KeA 2NCtn_zjmAYvVGLVZjFjmg

    # Limit + parallelism
    python dataset_gen/generate_persona_from_study.py --limit 5 --workers 8
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from llm_client import LLMClient
from prompts import build_distractor_facts_prompt
from generate_persona_facts import validate_json_structure
from generate_bio_notes import (
    extract_all_facts_from_topics_filled,
    get_occupation_from_facts,
    generate_distractor_facts,
    shuffle_facts,
    format_as_bio_notes,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def load_study_data(study_dir: Path) -> Dict[str, Any]:
    """Load memory bank and chat history from a user-study directory."""
    data: Dict[str, Any] = {"memories": [], "chat_history": ""}

    # Memory bank (try root first, then latest session)
    mem_path = study_dir / "memory_bank_content.json"
    if mem_path.exists():
        with open(mem_path, "r") as f:
            raw = json.load(f)
            data["memories"] = raw.get("memories", [])

    # Chat history – find latest session
    exec_dir = study_dir / "execution_logs"
    if exec_dir.is_dir():
        sessions = sorted(
            [d for d in exec_dir.iterdir() if d.name.startswith("session_")],
            key=lambda d: int(d.name.split("_")[1]),
            reverse=True,
        )
        for sess in sessions:
            chat_log = sess / "chat_history.log"
            if chat_log.exists():
                data["chat_history"] = chat_log.read_text()
                break

    return data


def format_memories_for_prompt(memories: List[Dict]) -> str:
    """Format memory bank entries into a concise text block."""
    lines = []
    for m in memories:
        title = m.get("title", "")
        text = m.get("text", "")
        lines.append(f"- [{title}] {text}")
    return "\n".join(lines)


def build_study_to_facts_prompt(
    memories_text: str,
    chat_history: str,
    topics: List[Dict],
) -> str:
    """Build the LLM prompt that converts study data → topics_filled.json."""

    subtopics_text = ""
    for topic in topics:
        subtopics_text += f"\n## {topic['topic']}\n"
        for i, subtopic in enumerate(topic["subtopics"], 1):
            subtopic_id = f"{topics.index(topic) + 1}.{i}"
            subtopics_text += f"{subtopic_id}. {subtopic}\n"

    # Truncate chat history to avoid exceeding context
    max_chat_chars = 12_000
    if len(chat_history) > max_chat_chars:
        chat_history = chat_history[:max_chat_chars] + "\n... [truncated]"

    prompt = f"""You are creating a synthetic persona profile that faithfully represents a real interview participant.

Below are two sources of ground-truth information from an actual interview session:

## Source 1 — Extracted Memories
These are structured facts the system extracted from the conversation:
<memories>
{memories_text}
</memories>

## Source 2 — Chat History
This is the raw interview transcript:
<chat_history>
{chat_history}
</chat_history>

## Instructions
1. Generate 3-5 specific, factual bullet points for EACH of the 48 subtopics below.
2. Base facts **strictly on what the participant actually said or implied** in the sources above.
3. When the sources are sparse for a subtopic, infer realistic details consistent with:
   - The person's stated occupation, role, and industry
   - Their education, experience level, and attitudes
   - What would be typical for someone in their position
4. Maintain GLOBAL CONSISTENCY across all subtopics — no contradictions.
5. Create SPECIFIC, VERIFIABLE facts, not generic statements.
   - BAD: "Uses various tools for work"
   - GOOD: "Uses Microsoft Copilot daily to review data for abnormalities and summarize emails"
6. Preserve the participant's own language and phrasing where possible.
7. Add depth where appropriate (4-5 facts for core work/AI topics, 3 for less-discussed topics).

## 48 Interview Subtopics:
{subtopics_text}

## Output Format
Return a JSON array matching this exact structure:

```json
[
  {{
    "topic": "Introduction & Background",
    "subtopics": [
      {{
        "subtopic_id": "1.1",
        "subtopic_description": "Educational background or training",
        "notes": [
          "Fact based on what participant said",
          "Another fact ...",
          "..."
        ]
      }},
      ...
    ]
  }},
  ...
]
```

CRITICAL: Include ALL 48 subtopics with 3-5 notes each. Ensure no contradictions across subtopics.
"""
    return prompt


def build_conversation_style_prompt(chat_history: str) -> str:
    """Build prompt to extract conversational style from the participant's responses."""

    # Truncate if needed
    max_chars = 15_000
    if len(chat_history) > max_chars:
        chat_history = chat_history[:max_chars] + "\n... [truncated]"

    return f"""Analyze the following interview transcript and produce a detailed conversational style guide for the participant (the "User" speaker). This guide will be used to make an LLM simulate this person's speaking style as faithfully as possible.

<chat_history>
{chat_history}
</chat_history>

Write a markdown document covering these aspects of HOW the participant communicates (not WHAT they said):

## Tone & Register
- Overall tone (formal/informal, warm/reserved, confident/tentative, etc.)
- Level of professionalism vs. casualness
- Emotional expressiveness or restraint

## Response Length & Structure
- Typical answer length (short/medium/long)
- Whether they use lists, run-on sentences, fragments, or structured paragraphs
- How they open and close responses (e.g., "So...", "I think...", trailing off, etc.)

## Verbal Habits & Filler Patterns
- Recurring phrases, hedges, or filler words (e.g., "period", "you know", "I think that", "really")
- Self-corrections or restarts
- Any distinctive verbal tics or patterns

## Depth & Elaboration Style
- Do they give the minimum answer or elaborate voluntarily?
- Do they use examples unprompted, or only when asked?
- How do they handle vague questions — ask for clarification, or just pick an interpretation?

## Engagement & Energy
- Are they enthusiastic, matter-of-fact, or guarded?
- Do they show curiosity about the interviewer's questions?
- How do they react to follow-ups — eager to expand, or repeat themselves?

## Notable Quirks
- Any unique patterns that distinguish this person's communication style
- Speech patterns that an LLM should replicate to sound authentic

Be specific and cite short examples from the transcript (quote 3-5 words at a time). The goal is to capture this person's voice so precisely that a reader familiar with them would recognize it."""


# ── per-user generation ──────────────────────────────────────────────────────

def generate_for_user(
    user_id: str,
    study_dir: Path,
    output_dir: Path,
    topics: List[Dict],
    llm_client: LLMClient,
) -> bool:
    """Generate topics_filled.json + bio_notes.md for one user."""

    output_dir.mkdir(parents=True, exist_ok=True)
    topics_path = output_dir / f"{user_id}_topics_filled.json"
    bio_path = output_dir / f"{user_id}_bio_notes.md"
    conv_path = output_dir / "conversation.md"

    # Skip if all already exist
    if topics_path.exists() and bio_path.exists() and conv_path.exists():
        print(f"  [SKIP] {user_id}: already generated")
        return True

    # Load source data
    study_data = load_study_data(study_dir)
    if not study_data["memories"] and not study_data["chat_history"]:
        print(f"  [SKIP] {user_id}: no study data found")
        return False

    print(f"  [GENERATING] {user_id} ({len(study_data['memories'])} memories, "
          f"{len(study_data['chat_history'])} chars chat)...")

    try:
        # ── Step 1: topics_filled.json ──
        if not topics_path.exists():
            memories_text = format_memories_for_prompt(study_data["memories"])
            prompt = build_study_to_facts_prompt(
                memories_text, study_data["chat_history"], topics
            )

            response = llm_client.call_gpt41(
                prompt=prompt, temperature=0.7, max_tokens=8192
            )

            # Strip markdown fences
            response = response.strip()
            for prefix in ("```json", "```"):
                if response.startswith(prefix):
                    response = response[len(prefix):]
                    break
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()

            topics_filled = json.loads(response)

            is_valid, err = validate_json_structure(topics_filled)
            if not is_valid:
                print(f"  [ERROR] {user_id}: invalid structure — {err}")
                return False

            with open(topics_path, "w") as f:
                json.dump(topics_filled, f, indent=2)

            total_facts = sum(
                len(st.get("notes", []))
                for t in topics_filled
                for st in t.get("subtopics", [])
            )
            print(f"    topics_filled.json: {total_facts} facts")
        else:
            with open(topics_path, "r") as f:
                topics_filled = json.load(f)

        # ── Step 2: bio_notes.md ──
        if not bio_path.exists():
            core_facts = extract_all_facts_from_topics_filled(topics_filled)
            occupation = get_occupation_from_facts(topics_filled)
            distractor_facts = generate_distractor_facts(
                occupation, core_facts, llm_client
            )

            all_facts = core_facts + distractor_facts
            shuffled = shuffle_facts(all_facts, user_id)
            bio_notes = format_as_bio_notes(shuffled)

            with open(bio_path, "w") as f:
                f.write(bio_notes)

            print(f"    bio_notes.md: {len(all_facts)} facts "
                  f"({len(core_facts)} core + {len(distractor_facts)} distractors)")

        # ── Step 3: conversation.md (tone & voice profile) ──
        if not conv_path.exists() and study_data["chat_history"]:
            conv_prompt = build_conversation_style_prompt(study_data["chat_history"])
            conv_response = llm_client.call_gpt41(
                prompt=conv_prompt, temperature=0.4, max_tokens=2048
            )
            with open(conv_path, "w") as f:
                f.write(conv_response.strip())
            print(f"    conversation.md: generated")

        print(f"  [SUCCESS] {user_id}")
        return True

    except json.JSONDecodeError as e:
        print(f"  [ERROR] {user_id}: JSON parse error — {e}")
        return False
    except Exception as e:
        print(f"  [ERROR] {user_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate simulator profiles from user-study interview data"
    )
    parser.add_argument(
        "--study-dir", type=str, default="user_study",
        help="Directory containing user-study subdirectories (default: user_study)"
    )
    parser.add_argument(
        "--topics-path", type=str, default="data/configs/topics.json",
        help="Path to topics.json schema"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/sample_user_profiles",
        help="Output directory for generated profiles"
    )
    parser.add_argument("--user-ids", nargs="+", help="Specific user IDs to process")
    parser.add_argument("--limit", type=int, help="Limit number of users")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers (default: 8)")

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    study_base = project_root / args.study_dir
    topics_path = project_root / args.topics_path
    output_base = project_root / args.output_dir

    print("=" * 60)
    print("Generate Simulator Profiles from User Study Data")
    print("=" * 60)
    print(f"Study directory : {study_base}")
    print(f"Topics schema   : {topics_path}")
    print(f"Output directory: {output_base}")
    print()

    # Load topics schema
    with open(topics_path, "r") as f:
        topics = json.load(f)
    print(f"Loaded {len(topics)} topics from schema")

    # Discover user directories
    if args.user_ids:
        user_dirs = [study_base / uid for uid in args.user_ids if (study_base / uid).is_dir()]
    else:
        user_dirs = sorted([d for d in study_base.iterdir() if d.is_dir()])

    if args.limit:
        user_dirs = user_dirs[: args.limit]

    print(f"Found {len(user_dirs)} user directories")
    print()

    success = 0
    fail = 0

    def process(user_dir: Path) -> bool:
        uid = user_dir.name
        out = output_base / uid
        client = LLMClient()
        return generate_for_user(uid, user_dir, out, topics, client)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, d): d.name for d in user_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating"):
            uid = futures[fut]
            try:
                if fut.result():
                    success += 1
                else:
                    fail += 1
            except Exception as e:
                print(f"\n  [ERROR] {uid}: {e}")
                fail += 1

    print()
    print("=" * 60)
    print(f"Done — {success} succeeded, {fail} failed out of {len(user_dirs)}")
    print("=" * 60)

    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
