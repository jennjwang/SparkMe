"""
Screen study_tasks.json using the ONET pipeline's screen_tasks function, per interview.
Saves kept tasks (pass + rewritten) with rewritten text in place.

Usage:
    python analysis/task_clustering/run_screening.py
"""

from __future__ import annotations
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))

import importlib.util as _ilu
def _load(name, path):
    spec = _ilu.spec_from_file_location(name, path)
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_onet_dir = Path(__file__).parent.parent / "onet"
screen_tasks = _load("agg", _onet_dir / "3_aggregate_canonical_tasks.py").screen_tasks

from llm_client import LLMClient

ROOT = Path(__file__).parent.parent.parent
INPUT  = ROOT / "analysis/onet/data/study_tasks.json"
OUTPUT = ROOT / "analysis/task_clustering/output/screened_study_tasks.json"


def process(record: dict) -> dict:
    llm = LLMClient()
    category = record.get("onet_title", record.get("occupation", "Unknown"))
    raw_tasks = record.get("tasks", [])

    # Format for ONET screen_tasks
    formatted = [
        {
            "occupation": record.get("occupation", ""),
            "statement":  t.get("task_statement", ""),
            "action":     t.get("action", ""),
            "object":     t.get("object", ""),
            "purpose":    t.get("purpose", ""),
            "tools":      t.get("tools", ""),
            "frequency":  t.get("duration", ""),
        }
        for t in raw_tasks
    ]

    valid, rewritten, rejected = screen_tasks(category, formatted, llm)

    # Map back by statement text
    orig_by_stmt = {t.get("task_statement", ""): t for t in raw_tasks}

    def enrich(s, status):
        orig_stmt = s.get("original_statement", s.get("statement", ""))
        orig_task = orig_by_stmt.get(orig_stmt, orig_by_stmt.get(s.get("statement", ""), {}))
        return {
            **orig_task,
            "task_statement": s.get("statement", orig_stmt),  # rewritten text if applicable
            "_original_statement": orig_stmt if status == "rewritten" else "",
            "_screen_status": status,
            "_screen_reason": s.get("reason", ""),
        }

    kept_tasks = [enrich(s, "pass") for s in valid] + \
                 [enrich(s, "rewritten") for s in rewritten]
    rejected_tasks = [enrich(s, "rejected") for s in rejected]

    uid = record.get("user_id", "?")
    print(f"  [{uid[:8]}] {category[:50]}: "
          f"{len(valid)} pass, {len(rewritten)} rewritten, {len(rejected)} rejected "
          f"/ {len(raw_tasks)} total")

    return {
        **{k: v for k, v in record.items() if k != "tasks"},
        "tasks": kept_tasks,
        "rejected_tasks": rejected_tasks,
    }


def main():
    with open(INPUT) as f:
        data = json.load(f)

    total_tasks = sum(len(r["tasks"]) for r in data)
    print(f"Screening {len(data)} interviews ({total_tasks} tasks)...\n")

    results = [None] * len(data)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process, r): i for i, r in enumerate(data)}
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(results, f, indent=2)

    total_kept     = sum(len(r["tasks"]) for r in results)
    total_rejected = sum(len(r["rejected_tasks"]) for r in results)
    total_rewritten = sum(
        sum(1 for t in r["tasks"] if t.get("_screen_status") == "rewritten")
        for r in results
    )
    print(f"\nDone — {total_kept} kept ({total_rewritten} rewritten), "
          f"{total_rejected} rejected out of {total_tasks} total")
    print(f"Output: {OUTPUT}")


if __name__ == "__main__":
    main()
