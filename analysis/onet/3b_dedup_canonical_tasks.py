"""
Deduplicate canonical task statements within each category.

After aggregation, some tasks describe the same core work activity from different
angles or with different specificity. This pass identifies those groups and merges
them into a single canonical statement, combining their source lists.

Can be run standalone on the current canonical_tasks.json:
    python analysis/onet/3b_dedup_canonical_tasks.py

Or imported and called from aggregate_canonical_tasks.py:
    from 3b_dedup_canonical_tasks import dedup_category
"""

import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient


DEDUP_PROMPT = """You are a job analyst reviewing canonical task statements for a single occupation category.

## Occupation Category
{category}

## Canonical Tasks
{task_list}

---

## Your Job
Identify tasks that describe the **same core work activity** and should be merged.

Merge tasks when:
- They share the same action verb AND object domain (e.g. both are about discussing work status with teammates)
- One is a specific instance of the other (attending standups IS sharing progress updates with the team)
- Separating them would not be meaningful to a job analyst or worker

Do NOT merge tasks that:
- Use different tools or methods that create genuinely distinct work
- Differ substantially in object (e.g. "review code" vs "review documentation")
- Serve different audiences or have substantially different outcomes

For merged tasks, write a statement that covers all merged sources without losing specificity.
Use the same structure: <Action> <object> to <immediate outcome>.

## Output
Return a JSON array — one entry per output task:
- Single (no merge):  {{"indices": [i]}}
- Merged group:       {{"indices": [i, j, ...], "merged_statement": "<new statement>",
                        "merged_parts": {{"action": "...", "object": "...", "outcome": "..."}},
                        "merge_reason": "<brief explanation>"}}

Return ONLY the JSON array. No markdown fences.
"""


def dedup_category(category: str, tasks: list, llm: LLMClient) -> list:
    """Detect and merge semantically overlapping canonical tasks within a category."""
    if len(tasks) <= 1:
        return tasks

    task_list = "\n".join(f"{i}. {t['canonical_statement']}" for i, t in enumerate(tasks))
    prompt = DEDUP_PROMPT.format(category=category, task_list=task_list)

    response = llm.call(prompt, model="gpt-5.4", temperature=0.1, max_tokens=4096).strip()
    for fence in ("```json", "```"):
        if response.startswith(fence):
            response = response[len(fence):]
    if response.endswith("```"):
        response = response[:-3]

    groups = json.loads(response.strip())

    result = []
    for group in groups:
        indices = group["indices"]
        if len(indices) == 1 or not group.get("merged_statement"):
            result.append(tasks[indices[0]])
            continue

        # Merge: combine source_statements and sum source_counts
        merged_sources = []
        seen_sources = set()
        total_count = 0
        for i in indices:
            t = tasks[i]
            total_count += t.get("source_count", 1)
            for src in t.get("source_statements", []):
                if src not in seen_sources:
                    seen_sources.add(src)
                    merged_sources.append(src)

        merged = dict(tasks[indices[0]])
        merged["canonical_statement"] = group["merged_statement"]
        merged["source_count"] = total_count
        merged["source_statements"] = merged_sources
        merged["notes"] = f"[merged {len(indices)} tasks: {group.get('merge_reason', '')}]"
        if group.get("merged_parts"):
            merged["statement_parts"] = group["merged_parts"]
        result.append(merged)

    return result


def main():
    root = Path(__file__).parent.parent.parent
    input_path = root / "analysis/onet/data/canonical_tasks.json"

    with open(input_path) as f:
        data = json.load(f)

    total_before = sum(len(r["canonical_tasks"]) for r in data)
    print(f"Deduplicating {len(data)} categories ({total_before} tasks)...\n")

    def process(r):
        llm = LLMClient()
        deduped = dedup_category(r["category"], r["canonical_tasks"], llm)
        return r["category"], deduped

    results_map = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process, r): r["category"] for r in data}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Categories"):
            cat = futures[fut]
            try:
                cat_name, deduped = fut.result()
                results_map[cat_name] = deduped
                for r in data:
                    if r["category"] == cat_name:
                        n_before = len(r["canonical_tasks"])
                        n_after = len(deduped)
                        delta = f" (-{n_before - n_after})" if n_after < n_before else ""
                        print(f"  [OK] {cat_name}: {n_before} → {n_after}{delta}")
            except Exception as e:
                print(f"  [ERROR] {cat}: {e}")

    for r in data:
        if r["category"] in results_map:
            r["canonical_tasks"] = results_map[r["category"]]

    with open(input_path, "w") as f:
        json.dump(data, f, indent=2)

    total_after = sum(len(r["canonical_tasks"]) for r in data)
    print(f"\nDone — {total_before} → {total_after} tasks (-{total_before - total_after})")
    print(f"Output: {input_path}")


if __name__ == "__main__":
    main()
