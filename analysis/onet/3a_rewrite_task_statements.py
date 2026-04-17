"""
Rewrite canonical task statements to ensure each has a specific:
  - Action:  concrete verb (not vague: "support", "facilitate", "ensure")
  - Object:  specific target artifact, system, or material
  - Outcome: immediate, proximate result (not downstream business goal)

Format: <Action> <object> to <specific outcome>.

Usage:
    python analysis/onet/3a_rewrite_task_statements.py
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


REWRITE_PROMPT = """You are a job analyst editing task statements to follow a strict three-part structure.

Each task statement must have exactly:
  1. ACTION   — a specific, observable verb (e.g. "Review", "Write", "Debug", "Schedule")
                NOT vague: "use", "support", "facilitate", "ensure", "manage", "handle", "perform", "participate in"
  2. OBJECT   — the concrete artifact, system, person, or material the action is performed on
                If the statement says "use [tool] to do X", the REAL object is what X is done to — not the tool.
                (specific enough to distinguish this task from others using the same verb)
  3. OUTCOME  — the immediate, proximate result produced by doing this task
                Must be grounded in the source statements — do NOT invent artifacts or outputs.
                NOT downstream goals: "to improve performance", "to meet requirements", "to optimize"
                YES immediate results: "to flag billing errors", "to catch regressions before deployment"

Format: <Action> <object> to <specific outcome>.

Keep statements concise — one sentence, under 25 words ideally.
Do NOT fabricate specifics not present in the original statement or source statements.

---

Rewrite each of the following task statements using only details from the statement and its sources.
Return a JSON array of objects with:
  - "original": the original statement (copy exactly)
  - "action": the extracted action verb(s)
  - "object": the extracted object
  - "outcome": the extracted outcome
  - "rewritten": the rewritten statement following the format above

Statements to rewrite (each includes its source statements for grounding):
{statements}

Return ONLY the JSON array. No markdown fences, no commentary.
"""


CHUNK_SIZE = 4  # max tasks per LLM call to avoid truncation


def _rewrite_chunk(tasks: list, llm: LLMClient) -> list:
    """Rewrite a single chunk of tasks."""
    statements_text = "\n".join(
        f"{i+1}. {t['canonical_statement']}" for i, t in enumerate(tasks)
    )
    prompt = REWRITE_PROMPT.format(statements=statements_text)
    response = llm.call_gpt41(prompt=prompt, temperature=0.1, max_tokens=4096).strip()

    for fence in ("```json", "```"):
        if response.startswith(fence):
            response = response[len(fence):]
    if response.endswith("```"):
        response = response[:-3]

    rewrites = json.loads(response.strip())

    updated = []
    for i, task in enumerate(tasks):
        rewrite = rewrites[i] if i < len(rewrites) else {}
        t = dict(task)
        t["canonical_statement"] = rewrite.get("rewritten", task["canonical_statement"])
        t["statement_parts"] = {
            "action":  rewrite.get("action", ""),
            "object":  rewrite.get("object", ""),
            "outcome": rewrite.get("outcome", ""),
        }
        updated.append(t)
    return updated


def rewrite_batch(category: str, tasks: list, llm: LLMClient) -> list:
    """Rewrite all canonical statements for one category, chunked to avoid truncation."""
    result = []
    for i in range(0, len(tasks), CHUNK_SIZE):
        chunk = tasks[i: i + CHUNK_SIZE]
        result.extend(_rewrite_chunk(chunk, llm))
    return result


def main():
    root = Path(__file__).parent.parent.parent
    input_path = root / "analysis/onet/data/canonical_tasks.json"

    with open(input_path) as f:
        data = json.load(f)

    print(f"Rewriting statements across {len(data)} categories...\n")

    def process(r):
        llm = LLMClient()
        updated_tasks = rewrite_batch(r["category"], r["canonical_tasks"], llm)
        return r["category"], updated_tasks

    results_map = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(process, r): r["category"] for r in data}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Rewriting"):
            cat = futures[fut]
            try:
                cat_name, updated = fut.result()
                results_map[cat_name] = updated
                print(f"  [OK] {cat_name}: {len(updated)} statements rewritten")
            except Exception as e:
                print(f"  [ERROR] {cat}: {e}")

    # Rebuild data preserving category order
    for r in data:
        if r["category"] in results_map:
            r["canonical_tasks"] = results_map[r["category"]]

    with open(input_path, "w") as f:
        json.dump(data, f, indent=2)

    total = sum(len(r["canonical_tasks"]) for r in data)
    print(f"\nDone — {total} statements rewritten → {input_path}")


if __name__ == "__main__":
    main()
