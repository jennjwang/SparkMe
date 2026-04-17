"""
Aggregate task statements across participants in the same occupation category
into a canonical task list, applying O*NET and DOT heuristics.

Usage:
    python analysis/onet/3_aggregate_canonical_tasks.py
    python analysis/onet/3_aggregate_canonical_tasks.py --output analysis/onet/data/canonical_tasks.json
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient

# Load sub-modules with numeric prefixes via importlib
import importlib.util as _ilu
def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
_onet_dir = Path(__file__).parent
rewrite_batch  = _load("rewrite",  _onet_dir / "3a_rewrite_task_statements.py").rewrite_batch
dedup_category = _load("dedup",    _onet_dir / "3b_dedup_canonical_tasks.py").dedup_category


SCREEN_PROMPT = """You are a job analyst screening worker-reported task statements for validity.

## Occupation Category
{category}

## Raw Task Statements ({n_tasks} statements from {n_participants} participants)
Format: [Occupation] Task statement | action | object | purpose | tools | frequency

{task_list}

---

## Instructions

For each statement, apply these five checks:

1. **Specific action?** — Does it name a concrete, observable verb?
   Pass: "Debug", "Review", "Write", "Schedule", "Create", "Build", "Analyze", "Draft"
   Fail: "Support", "Manage", "Handle", "Facilitate", "Ensure", "Participate in"

2. **Concrete object?** — Does it act on a specific artifact, system, document, or person?
   Pass: "billing reports", "unit tests", "client contracts", "data pipelines"
   Fail: "tasks", "things", "work", "multiple priorities"

3. **Bounded activity?** — Does it have a start and end, or is it a standing trait/disposition?
   Pass: something that happens and finishes (recurring tasks like "weekly reports" still count)
   Fail: "Be a good communicator", "Understand the business context", "Know how to use Excel"

4. **Single task?** — Does it stand as one coherent work unit, or bundle 3+ unrelated activities?
   Pass: one meaningful unit of work (closely related sub-steps like "buy and sell" count as one)
   Fail: "Handle all aspects of onboarding, recruitment, and event planning"

5. **Common to this occupation?** — Would most workers in this occupation category perform this task?
   Pass: a recognizable, typical activity for the occupation (e.g. code review for SWEs)
   Fail: idiosyncratic to one person's unique situation (e.g. a SWE who also plans office parties)

## Rewriting and Splitting

If a statement FAILS one or more checks but describes a real work activity, try to fix it:
- Replace vague verbs with the most specific action implied by the statement
- If it bundles 2-3 related sub-steps that form one logical activity, keep it as one task
- If it bundles multiple distinct activities (e.g. "conduct experiments, including coding and
  debugging"), **split it** — return one object per distinct activity, each with status "split"
- Keep the purpose/objective — it does NOT need to be an immediate proximate outcome
- Use the format: <Action> <object> to <purpose>.
- Only use details present in the original statement — do not fabricate

If the statement is truly unsalvageable (describes a personality trait, a role rather than a task,
or is too vague to extract any concrete activity), mark it as rejected.

## Output

Return a JSON array. Each input statement produces **one or more** output objects:
- Normally: one object per input statement
- When splitting: multiple objects with `status: "split"`, one per distinct activity extracted

```
{{
  "statement": "<original statement text, copied exactly>",
  "occupation": "<occupation from the bracket prefix>",
  "checks": {{
    "specific_action": true | false,
    "concrete_object": true | false,
    "bounded_activity": true | false,
    "single_task": true | false,
    "common_to_occupation": true | false
  }},
  "status": "pass" | "rewritten" | "split" | "rejected",
  "rewritten": "<rewritten or split statement if status is 'rewritten' or 'split', empty string otherwise>",
  "reason": "<brief explanation of what was fixed, split, or why it was rejected>"
}}
```

Return ONLY the JSON array. No markdown fences.
"""


GROUP_PROMPT = """You are a job analyst grouping valid task statements into a canonical task list.

## Occupation Category
{category}

## Participants
{occupations}

## Valid Task Statements ({n_valid} statements from {n_participants} participants)
{valid_list}

---

## Instructions

### 1. Group and merge statements

For each statement, compare it against every other statement and ask:

1. **Identical?** — Same core activity described in different words.
   → Merge into one canonical statement.
2. **Partially redundant?** — Overlaps with another statement but adds something new
   (e.g. different tool, different context, additional sub-step).
   → Merge, but make sure the canonical statement preserves what is distinct.
3. **Part of another task?** — One statement is a sub-step or component of a broader task.
   → Absorb the narrower statement into the broader one.
4. **Distinct?** — Different activity entirely.
   → Keep as separate canonical tasks.

**Keep separate** when statements:
- Could be assigned to a different worker
- Require meaningfully different tools, skills, or methods
- Differ substantially in frequency or importance

**Across participants:** if multiple people describe the same core task in different words, merge into
one canonical statement that generalises across what was reported.

### 2. Write the canonical statement

Each statement must follow this structure:

    <action verb> <concrete object> to <immediate outcome>.

- **Action:** observable verb — not: support, facilitate, ensure, handle, manage, perform
- **Object:** specific artifact, system, document, or person — enough to distinguish from other tasks
- **Outcome:** the immediate result of doing this task — not downstream goals ("improve performance")
- **Tools:** include only when a specific named tool appears in the source statements

Examples:
- GOOD: "Review monthly billing reports in Excel using Copilot to flag revenue anomalies for the finance team."
- BAD: "Manage data tasks to support business goals." (vague action, vague object, vague outcome)

## Output

Return a JSON array. Each object has:

```
{{
  "canonical_statement": "<statement following the structure above>",
  "abbreviated_phrase": "<2-5 word label, e.g. 'code debugging', 'billing review'>",
  "source_count": <number of participants who contributed>,
  "source_statements": ["<original statement 1>", "<original statement 2>", ...],
  "merge_type": "identical" | "partial_overlap" | "subsumes" | "single",
  "notes": "<analyst notes — for merges, explain what was combined and why>"
}}
```

`merge_type` values:
- `identical`: merged statements that describe the same task in different words
- `partial_overlap`: merged statements that overlap but each added distinct details
- `subsumes`: a broader statement absorbed a narrower sub-step
- `single`: no merge — this statement stood on its own

Return ONLY the JSON array. No markdown fences, no commentary.
"""


def build_task_list_text(tasks: list) -> str:
    lines = []
    for t in tasks:
        parts = [
            f"[{t['occupation']}]",
            t['statement'] or "(no statement)",
        ]
        details = []
        if t.get('action'):
            details.append(f"action: {t['action']}")
        if t.get('object'):
            details.append(f"object: {t['object']}")
        if t.get('purpose'):
            details.append(f"purpose: {t['purpose']}")
        if t.get('tools'):
            details.append(f"tools: {t['tools']}")
        if t.get('frequency'):
            details.append(f"frequency: {t['frequency']}")
        if details:
            parts.append("| " + " | ".join(details))
        lines.append(" ".join(parts))
    return "\n".join(lines)


def _parse_json(response: str) -> list:
    """Strip markdown fences and parse JSON, with retry on failure."""
    response = response.strip()
    for fence in ("```json", "```"):
        if response.startswith(fence):
            response = response[len(fence):]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        # Try to extract the JSON array from the response
        start = response.find("[")
        end = response.rfind("]")
        if start != -1 and end != -1:
            return json.loads(response[start:end + 1])
        raise


def screen_tasks(category: str, tasks: list, llm: LLMClient) -> tuple[list, list, list]:
    """Stage 3a: screen each raw statement for validity. Returns (valid, rewritten, rejected)."""
    prompt = SCREEN_PROMPT.format(
        category=category,
        n_participants=len(set(t["occupation"] for t in tasks)),
        n_tasks=len(tasks),
        task_list=build_task_list_text(tasks),
    )

    response = llm.call(prompt, model="gpt-5.4", temperature=0.1, max_tokens=8192)
    screened = _parse_json(response)

    valid, rewritten, rejected = [], [], []
    for s in screened:
        status = s.get("status", "")
        if status == "pass":
            valid.append(s)
        elif status in ("rewritten", "split") and s.get("rewritten"):
            # Replace the statement text with the rewritten/split version
            s["original_statement"] = s.get("statement", "")
            s["statement"] = s["rewritten"]
            rewritten.append(s)
        else:
            rejected.append(s)
    return valid, rewritten, rejected


def group_tasks(category: str, valid_statements: list, occupations: list,
                llm: LLMClient) -> list:
    """Stage 3b: group and merge valid statements into canonical tasks."""
    valid_list = "\n".join(
        f"[{s['occupation']}] {s['statement']}" for s in valid_statements
    )

    prompt = GROUP_PROMPT.format(
        category=category,
        occupations="\n".join(f"- {o}" for o in occupations),
        n_valid=len(valid_statements),
        n_participants=len(occupations),
        valid_list=valid_list,
    )

    response = llm.call(prompt, model="gpt-5.4", temperature=0.2, max_tokens=8192)
    return _parse_json(response)


def aggregate_category(category: str, tasks: list, llm: LLMClient) -> dict:
    occupations = sorted(set(t["occupation"] for t in tasks))
    n_participants = len(occupations)

    # Stage 3a — Validity screening + rewrite recovery
    valid, rewritten, rejected = screen_tasks(category, tasks, llm)
    all_valid = valid + rewritten  # both pass through to grouping
    if rejected:
        print(f"    {category}: rejected {len(rejected)}/{len(tasks)} statements"
              f" (rewritten {len(rewritten)})")

    # Build mapping from rewritten/passed statement → original raw statement
    stmt_to_original = {}
    for s in all_valid:
        # The statement text used in grouping (may be rewritten)
        used_stmt = f"[{s['occupation']}] {s['statement']}"
        # The original raw statement from study_tasks.json
        original = s.get("original_statement", s.get("statement", ""))
        # Strip the [Occupation] prefix and metadata from original if present
        clean_original = original
        if "|" in clean_original:
            clean_original = clean_original.split("|")[0].strip()
        if clean_original.startswith("["):
            bracket_end = clean_original.find("]")
            if bracket_end > 0:
                clean_original = clean_original[bracket_end + 1:].strip()
        stmt_to_original[used_stmt] = clean_original

    # Stage 3b — Grouping and merging
    canonical = group_tasks(category, all_valid, occupations, llm)

    # Replace source_statements with original raw statements for quote lookup
    for t in canonical:
        original_sources = []
        for src in t.get("source_statements", []):
            original_sources.append(stmt_to_original.get(src, src))
        t["source_statements"] = original_sources

    after_group = [dict(t) for t in canonical]  # snapshot

    # Stage 3c — Rewrite to action/object/outcome structure
    canonical = rewrite_batch(category, canonical, llm)
    after_rewrite = [dict(t) for t in canonical]  # snapshot

    # Stage 3d — Deduplication
    canonical = dedup_category(category, canonical, llm)

    return {
        "category": category,
        "occupations": occupations,
        "n_participants": n_participants,
        "n_raw_tasks": len(tasks),
        "n_screened_out": len(rejected),
        "n_rewritten": len(rewritten),
        "canonical_tasks": canonical,
        "_trace": {
            "raw_tasks": tasks,
            "stage_3a_valid": valid,
            "stage_3a_rewritten": rewritten,
            "stage_3a_rejected": rejected,
            "stage_3b_grouped": after_group,
            "stage_3c_rewritten": after_rewrite,
            "stage_3d_deduped": canonical,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="analysis/onet/data/study_tasks.json")
    parser.add_argument("--output", default="analysis/onet/data/canonical_tasks.json")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--categories", nargs="+", help="Only process specific categories")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    input_path = root / args.input
    output_path = root / args.output

    with open(input_path) as f:
        data = json.load(f)

    # Group raw tasks by O*NET title
    by_cat: dict[str, list] = defaultdict(list)
    for r in data:
        cat = r.get("onet_title", "Other")
        if args.categories and cat not in args.categories:
            continue
        for t in r["tasks"]:
            by_cat[cat].append({
                "occupation": r["occupation"],
                "statement": t.get("task_statement", ""),
                "action": t.get("action", ""),
                "object": t.get("object", ""),
                "purpose": t.get("purpose", ""),
                "tools": t.get("tools", ""),
                "frequency": t.get("frequency", ""),
            })

    print(f"Aggregating {len(by_cat)} categories...\n")

    results = []
    errors = []

    def process(cat_tasks):
        cat, tasks = cat_tasks
        llm = LLMClient()
        return aggregate_category(cat, tasks, llm)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, item): item[0] for item in by_cat.items()}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Categories"):
            cat = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                n_canonical = len(result["canonical_tasks"])
                n_raw = result["n_raw_tasks"]
                print(f"  [OK] {cat}: {n_raw} raw → {n_canonical} canonical")
            except Exception as e:
                print(f"  [ERROR] {cat}: {e}")
                errors.append(cat)

    results.sort(key=lambda r: r["category"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save pipeline trace separately
    trace = [{
        "category": r["category"],
        **r.pop("_trace"),
    } for r in results]
    trace_path = output_path.parent / "pipeline_trace.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Pipeline trace: {trace_path}")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    total_canonical = sum(len(r["canonical_tasks"]) for r in results)
    total_raw = sum(r["n_raw_tasks"] for r in results)
    print(f"\nDone — {total_raw} raw tasks → {total_canonical} canonical tasks")
    print(f"Output: {output_path}")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
