"""
Compare canonical task statements (from user study) against O*NET task lists.

For each occupation category:
  1. Identify relevant O*NET codes (from participant mappings)
  2. Fetch O*NET task statements for those codes
  3. LLM comparison:
     - For each canonical task: find best O*NET match (exact/partial/none)
     - Classify novel tasks as ai_augmented, ai_new, or new_non_ai
     - Identify O*NET tasks absent from our study

Output: analysis/onet/data/onet_comparison.json

Usage:
    python analysis/onet/4_compare_onet_tasks.py
"""

import sys
import re
import json
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import openpyxl

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient


COMPARE_PROMPT = """You are a labor economist comparing worker-reported task statements against the O*NET task database.

## Occupation Category
{category}

## Our Canonical Tasks (from user study)
{canonical_tasks}

## O*NET Task Statements for relevant occupations in this category
{onet_tasks}

---

## Your Job

### Part 1 — For each canonical task, classify its O*NET coverage:
- "exact": O*NET has a task with the same core activity, object, and intent
- "partial": The canonical task is a recognizable instance of a broader O*NET task, or the
             O*NET task is one component of the canonical task — a subset/superset relationship.
             A worker doing one would immediately recognize the other as part of the same work.
             Being in the same domain is NOT enough. "Design computers" does NOT partially match
             "read academic papers and re-implement models".
- "novel": No O*NET task meaningfully covers this activity — mark novel even if a distantly
           related task exists.
- "unsure": You cannot confidently distinguish between "partial" and "novel" — the match is
            ambiguous. Use this instead of forcing a judgment.

For "partial" and "novel" tasks, classify the novelty type:
- "ai_augmented": a traditional task now performed with AI assistance (e.g. "summarize papers using AI tools")
- "ai_new": a genuinely new task created by AI (e.g. "verify AI-generated answers for accuracy")
- "new_non_ai": a new or missing task unrelated to AI

Set "best_onet_match" to the O*NET task text only when there is a genuine semantic overlap.
If no O*NET task meaningfully overlaps, set "best_onet_match" to null.

### Part 2 — O*NET tasks absent from our study:
List O*NET tasks that have NO match in our canonical list.
For each, note whether it was likely: "not_reported" (workers do it but didn't mention it)
or "out_of_scope" (outside what participants do).

---

Return a JSON object with:
{{
  "canonical_coverage": [
    {{
      "canonical_task": "<exact text>",
      "coverage": "exact" | "partial" | "novel" | "unsure",
      "novelty_type": "ai_augmented" | "ai_new" | "new_non_ai" | null,
      "best_onet_match": "<closest O*NET task text, or null>",
      "notes": "<brief explanation>"
    }}
  ],
  "onet_absent": [
    {{
      "onet_task": "<O*NET task text>",
      "onet_code": "<SOC code>",
      "status": "not_reported" | "out_of_scope",
      "notes": "<brief explanation>"
    }}
  ]
}}

Return ONLY the JSON object. No markdown fences.
"""


def load_onet_tasks(path: Path) -> dict[str, list[dict]]:
    """Load O*NET task statements, keyed by SOC code."""
    wb = openpyxl.load_workbook(path)
    ws = wb["Task Statements"]
    by_code = defaultdict(list)
    for row in ws.iter_rows(min_row=2, values_only=True):
        code, title, task_id, task, task_type = row[0], row[1], row[2], row[3], row[4]
        if code and task:
            by_code[code].append({
                "code": code,
                "title": title,
                "task": task,
                "task_type": task_type,
            })
    return dict(by_code)


def compare_category(
    category: str,
    canonical_tasks: list[dict],
    onet_codes: list[str],
    onet_by_code: dict,
    llm: LLMClient,
) -> dict:
    # Collect O*NET tasks for relevant codes (core tasks only to keep context manageable)
    onet_task_lines = []
    seen_tasks = set()
    for code in onet_codes:
        tasks = onet_by_code.get(code, [])
        for t in tasks:
            if t["task_type"] == "Core" and t["task"] not in seen_tasks:
                onet_task_lines.append(f"[{code} {t['title']}] {t['task']}")
                seen_tasks.add(t["task"])

    canonical_lines = "\n".join(
        f"{i+1}. {t['canonical_statement']}"
        for i, t in enumerate(canonical_tasks)
    )
    onet_lines = "\n".join(onet_task_lines) if onet_task_lines else "(no O*NET tasks found for these codes)"

    prompt = COMPARE_PROMPT.format(
        category=category,
        canonical_tasks=canonical_lines,
        onet_tasks=onet_lines,
    )

    response = llm.call(prompt, model="gpt-5.4", temperature=0.1, max_tokens=8192).strip()
    for fence in ("```json", "```"):
        if response.startswith(fence):
            response = response[len(fence):]
    if response.endswith("```"):
        response = response[:-3]

    result = json.loads(response.strip())
    # Strip number prefixes the LLM echoes back (e.g. "1. Write code..." → "Write code...")
    for item in result.get("canonical_coverage", []):
        item["canonical_task"] = re.sub(r'^\d+\.\s*', '', item["canonical_task"])
    result["category"] = category
    result["onet_codes"] = onet_codes
    result["n_canonical"] = len(canonical_tasks)
    result["n_onet_tasks"] = len(onet_task_lines)
    return result


def main():
    root = Path(__file__).parent.parent.parent
    onet_tasks_path = root / "analysis/onet/Task Statements O*NET 30.2.xlsx"
    canonical_path = root / "analysis/onet/data/canonical_tasks.json"
    study_path = root / "analysis/onet/data/study_tasks.json"
    output_path = root / "analysis/onet/data/onet_comparison.json"

    print("Loading O*NET task statements...")
    onet_by_code = load_onet_tasks(onet_tasks_path)
    print(f"  {sum(len(v) for v in onet_by_code.values())} tasks across {len(onet_by_code)} codes")

    with open(canonical_path) as f:
        canonical_data = json.load(f)

    with open(study_path) as f:
        study_data = json.load(f)

    # Build category → O*NET codes mapping from participant data
    # Categories in canonical_tasks.json may be O*NET titles or LLM-assigned categories
    cat_to_codes: dict[str, set] = defaultdict(set)
    for r in study_data:
        if r.get("onet_code"):
            cat_to_codes[r.get("category", "")].add(r["onet_code"])
            cat_to_codes[r.get("onet_title", "")].add(r["onet_code"])

    print(f"\nComparing {len(canonical_data)} categories...\n")

    results = []
    errors = []

    def process(r):
        cat = r["category"]
        onet_codes = sorted(cat_to_codes.get(cat, []))
        llm = LLMClient()
        return compare_category(cat, r["canonical_tasks"], onet_codes, onet_by_code, llm)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process, r): r["category"] for r in canonical_data}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Categories"):
            cat = futures[fut]
            try:
                result = fut.result()
                results.append(result)
                coverage = result.get("canonical_coverage", [])
                counts = {"exact": 0, "partial": 0, "novel": 0}
                for c in coverage:
                    counts[c.get("coverage", "novel")] = counts.get(c.get("coverage"), 0) + 1
                print(f"  [OK] {cat}: {counts['exact']} exact, {counts['partial']} partial, {counts['novel']} novel")
            except Exception as e:
                print(f"  [ERROR] {cat}: {e}")
                errors.append(cat)

    results.sort(key=lambda r: r["category"])
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary stats
    all_coverage = [c for r in results for c in r.get("canonical_coverage", [])]
    total = len(all_coverage)
    exact = sum(1 for c in all_coverage if c.get("coverage") == "exact")
    partial = sum(1 for c in all_coverage if c.get("coverage") == "partial")
    novel = sum(1 for c in all_coverage if c.get("coverage") == "novel")
    ai_aug = sum(1 for c in all_coverage if c.get("novelty_type") == "ai_augmented")
    ai_new = sum(1 for c in all_coverage if c.get("novelty_type") == "ai_new")

    print(f"\n=== Summary ===")
    print(f"Total canonical tasks: {total}")
    print(f"  Exact O*NET match:  {exact} ({100*exact//total}%)")
    print(f"  Partial match:      {partial} ({100*partial//total}%)")
    print(f"  Novel (no O*NET):   {novel} ({100*novel//total}%)")
    print(f"    → AI-augmented:   {ai_aug}")
    print(f"    → AI-new:         {ai_new}")
    print(f"    → New non-AI:     {novel - ai_aug - ai_new}")
    print(f"\nOutput: {output_path}")
    if errors:
        print(f"Errors: {errors}")


if __name__ == "__main__":
    main()
