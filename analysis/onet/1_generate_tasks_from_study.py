"""
Generate structured task objects from user-study memory banks.

For each study participant, extracts memories linked to subtopics 2.x
(Core Responsibilities) and 3.x (Task Proficiency, Challenge, Engagement),
then uses an LLM to produce tasks structured according to configs/task_probe.json.

Usage:
    python dataset_gen/generate_tasks_from_study.py \
        --study-dir user_study \
        --output data/study_tasks.json

    # Process specific users
    python dataset_gen/generate_tasks_from_study.py \
        --user-ids 1T8lGuWK6w-0q4S-s2_KeA 2NCtn_zjmAYvVGLVZjFjmg

    # Limit + parallelism
    python dataset_gen/generate_tasks_from_study.py --limit 5 --workers 4
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)

sys.path.append(str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient


RESPONSIBILITY_SUBTOPICS = {
    "2.1": "Primary job responsibilities and regular daily tasks",
    "2.2": "Approximate proportion of time spent on core activities",
    "2.3": "Level of autonomy and scope of decision-making in the role",
    "2.4": "Additional responsibilities or tools relevant to decision-making",
    "3.1": "Tasks that feel easiest or most natural to perform",
    "3.2": "Tasks perceived as most challenging or complex",
    "3.3": "Tasks that are repetitive, data-heavy, or suitable for automation",
    "3.4": "Tasks that are most enjoyable or engaging versus boring or tedious",
    "3.5": "Common pain points or inefficiencies in completing tasks",
    "3.6": "How enjoyment, skill level, and productivity relate to one another",
    "3.7": "Strategies or workarounds used for difficult tasks",
}


def load_responsibility_memories(user_dir: Path) -> List[Dict]:
    """Load memories linked to subtopics 2.x or 3.x from a participant's memory bank."""
    mem_path = user_dir / "memory_bank_content.json"
    if not mem_path.exists():
        return []

    with open(mem_path) as f:
        data = json.load(f)

    relevant = []
    for m in data.get("memories", []):
        subtopics = [s["subtopic_id"] for s in m.get("subtopic_links", [])]
        if any(s.startswith("2.") or s.startswith("3.") for s in subtopics):
            relevant.append({
                "title": m.get("title", ""),
                "text": m.get("text", ""),
                "subtopics": subtopics,
                "source_response": m.get("source_interview_response", ""),
            })
    return relevant


def load_occupation(user_dir: Path, llm: "LLMClient") -> Dict[str, str]:
    """Extract job title, sector, and industry from the participant's memory bank using LLM.

    Returns a dict with keys: occupation, sector, industry.
    """
    mem_path = user_dir / "memory_bank_content.json"
    if not mem_path.exists():
        return {"occupation": "", "sector": "", "industry": ""}

    with open(mem_path) as f:
        data = json.load(f)

    # Collect memories tagged with 1.x (background/role) subtopics
    background_memories = []
    for m in data.get("memories", []):
        subtopics = [s["subtopic_id"] for s in m.get("subtopic_links", [])]
        if any(s.startswith("1.") for s in subtopics):
            background_memories.append(f"- {m['title']}: {m['text']}")

    if not background_memories:
        return {"occupation": "", "sector": "", "industry": ""}

    prompt = (
        "Below are facts about a worker extracted from an interview.\n\n"
        + "\n".join(background_memories)
        + "\n\nBased only on the above, return a JSON object with three fields:\n"
        '- "occupation": their job title as a short noun phrase '
        "(e.g. 'Software Engineer', 'PhD Student', 'Operations Manager (Healthcare)'). "
        "Include industry in parentheses only if it meaningfully distinguishes the role.\n"
        '- "sector": the broad economic sector they work in '
        "(e.g. 'Healthcare', 'Technology', 'Finance', 'Education', 'Retail', "
        "'Manufacturing', 'Media & Entertainment', 'Government', 'Non-profit', 'Research & Academia').\n"
        '- "industry": the specific industry within that sector '
        "(e.g. 'orthopedic physician practice', 'electric vehicle / automotive', 'e-commerce').\n"
        "If a field cannot be determined, use an empty string. "
        "Return ONLY the JSON object, nothing else."
    )

    response = llm.call_gpt41(prompt=prompt, temperature=0, max_tokens=64).strip()
    try:
        result = json.loads(response)
        return {
            "occupation": result.get("occupation", "").strip().strip('"'),
            "sector":     result.get("sector", "").strip(),
            "industry":   result.get("industry", "").strip(),
        }
    except json.JSONDecodeError:
        # Fallback: treat the whole response as just the occupation title
        return {"occupation": response.strip('"'), "sector": "", "industry": ""}


def build_task_generation_prompt(
    occupation: str,
    memories: List[Dict],
    task_probe: List[Dict],
) -> str:
    """Build the LLM prompt that converts responsibility memories → structured tasks."""

    # Format the task_probe schema as a concise reference
    schema_lines = []
    for topic_obj in task_probe:
        schema_lines.append(f"\n### {topic_obj['topic']}")
        for st in topic_obj["subtopics"]:
            schema_lines.append(
                f"- **{st['id']}**: {st['description']}"
            )
    schema_text = "\n".join(schema_lines)

    # Format memories with IDs so the LLM can cite them
    memory_lines = []
    for i, m in enumerate(memories):
        subtopic_labels = ", ".join(
            RESPONSIBILITY_SUBTOPICS.get(s, s) for s in m["subtopics"]
            if s.startswith("2.") or s.startswith("3.")
        )
        quote = m.get("source_response", "").strip()
        memory_lines.append(
            f"[MEM-{i}] [{subtopic_labels}]\n"
            f"Title: {m['title']}\n"
            f"Detail: {m['text']}\n"
            f"Verbatim quote: \"{quote}\""
        )
    memories_text = "\n\n".join(memory_lines)

    return f"""You are building a structured task dataset from real worker interview data.

## Worker Occupation
{occupation}

## Raw Interview Memories (responsibilities & task proficiency)
The following facts were extracted from a real interview. Each memory is tagged with the interview subtopic(s) it covers.

<memories>
{memories_text}
</memories>

## Task Schema
Each task you generate must follow this schema exactly:

{schema_text}

## Instructions
1. Identify **3–6 distinct, concrete tasks** this worker performs based on the memories above.
   - A task is a meaningful unit of work (not a single click, not an entire job function).
   - Prefer tasks that are mentioned explicitly or strongly implied.
2. For each task, populate schema fields from the memories only.
   - Use ONLY information explicitly stated or directly implied in the memories. Do NOT infer or fabricate.
   - If a field is not covered by the memories, leave it as an empty string "".
   - Exception: `task_statement`, `action`, `object`, and `purpose` should always be filled.
   - Be specific: name tools, not "software"; name outputs, not "results".
3. For every non-empty field, also record the verbatim quote from the interview that supports it.
   - Use the `sources` object: keys are field names, values are the exact quote from the "Verbatim quote" of the relevant MEM-N.
   - Only cite quotes that directly support the field. Leave the source key absent if the field is empty.

## Output Format
Return a JSON array of task objects. Each object must have:
- All schema fields: task_statement, action, object, purpose, tools, information_sources, method,
  judgment, quality_criteria, work_context, frequency, duration, skills, experience, training
- A `sources` object mapping each populated field name → verbatim quote string

- **task_statement**: A single natural-language sentence combining action, object, and purpose.
  Example: "Review clinical billing data in Excel using Microsoft Copilot to flag abnormalities and missed revenue items."

Example skeleton (fill in real content — do NOT return this skeleton):
```json
[
  {{
    "task_statement": "...",
    "action": "...",
    "object": "...",
    "purpose": "...",
    "tools": "...",
    "information_sources": "...",
    "method": "...",
    "judgment": "...",
    "quality_criteria": "...",
    "work_context": "...",
    "frequency": "...",
    "duration": "...",
    "skills": "...",
    "experience": "...",
    "training": "...",
    "sources": {{
      "task_statement": "verbatim quote...",
      "action": "verbatim quote...",
      "tools": "verbatim quote..."
    }}
  }}
]
```

Return ONLY the JSON array. No markdown fences, no commentary.
"""


def generate_tasks_for_user(
    user_id: str,
    user_dir: Path,
    task_probe: List[Dict],
    llm: LLMClient,
) -> Optional[Dict]:
    """Generate structured tasks for one participant. Returns a result dict or None."""
    memories = load_responsibility_memories(user_dir)
    if not memories:
        return None

    occ = load_occupation(user_dir, llm)
    prompt = build_task_generation_prompt(occ["occupation"], memories, task_probe)

    try:
        response = llm.call_gpt41(prompt=prompt, temperature=0.4, max_tokens=4096)
        response = response.strip()
        # Strip markdown fences if present
        for fence in ("```json", "```"):
            if response.startswith(fence):
                response = response[len(fence):]
                break
        if response.endswith("```"):
            response = response[:-3]
        response = response.strip()

        tasks = json.loads(response)
        if not isinstance(tasks, list):
            print(f"  [ERROR] {user_id}: unexpected response type {type(tasks)}")
            return None

        return {
            "user_id":    user_id,
            "occupation": occ["occupation"],
            "sector":     occ["sector"],
            "industry":   occ["industry"],
            "num_memories": len(memories),
            "tasks": tasks,
        }
    except json.JSONDecodeError as e:
        print(f"  [ERROR] {user_id}: JSON parse error — {e}")
        return None
    except Exception as e:
        print(f"  [ERROR] {user_id}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate structured tasks from user-study responsibility memories"
    )
    parser.add_argument(
        "--study-dir", type=str, default="user_study",
        help="Directory containing user-study subdirectories (default: user_study)",
    )
    parser.add_argument(
        "--task-probe", type=str, default="configs/task_probe.json",
        help="Path to task_probe.json schema (default: configs/task_probe.json)",
    )
    parser.add_argument(
        "--output", type=str, default="analysis/onet/data/study_tasks.json",
        help="Output file path (default: data/study_tasks.json)",
    )
    parser.add_argument("--user-ids", nargs="+", help="Specific user IDs to process")
    parser.add_argument("--limit", type=int, help="Limit number of users to process")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers (default: 4)")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    study_base = root / args.study_dir
    task_probe_path = root / args.task_probe
    output_path = root / args.output

    print("=" * 60)
    print("Generate Tasks from User Study Responsibility Memories")
    print("=" * 60)
    print(f"Study directory : {study_base}")
    print(f"Task probe schema: {task_probe_path}")
    print(f"Output           : {output_path}")
    print()

    with open(task_probe_path) as f:
        task_probe = json.load(f)

    # Discover user directories
    if args.user_ids:
        user_dirs = [study_base / uid for uid in args.user_ids if (study_base / uid).is_dir()]
    else:
        user_dirs = sorted([d for d in study_base.iterdir() if d.is_dir()])

    if args.limit:
        user_dirs = user_dirs[: args.limit]

    print(f"Processing {len(user_dirs)} participants...\n")

    results = []
    errors = 0

    def process(user_dir: Path):
        uid = user_dir.name
        llm = LLMClient()
        return generate_tasks_for_user(uid, user_dir, task_probe, llm)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, d): d.name for d in user_dirs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Generating tasks"):
            uid = futures[fut]
            try:
                result = fut.result()
                if result:
                    results.append(result)
                    n = len(result["tasks"])
                    print(f"  [OK] {uid}: {n} task(s) ({result['occupation']} | {result['sector']} / {result['industry']})")
                else:
                    errors += 1
            except Exception as e:
                print(f"  [ERROR] {uid}: {e}")
                errors += 1

    # Sort by user_id for determinism
    results.sort(key=lambda r: r["user_id"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_tasks = sum(len(r["tasks"]) for r in results)
    print()
    print("=" * 60)
    print(f"Done — {len(results)} participants, {total_tasks} total tasks, {errors} errors")
    print(f"Output written to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
