"""
Map each study participant's occupation to the best-matching O*NET SOC code.
Matches based on work activities (task statements), not just job title.
Adds onet_code and onet_title fields to each record in study_tasks.json.

Usage:
    python analysis/map_onet_occupations.py
"""

import sys
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv
import openpyxl

load_dotenv(override=True)
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient


MATCH_PROMPT = """You are an occupational analyst mapping workers to O*NET-SOC codes.

For each participant, follow this two-step process:

STEP 1 — Identify the industry/sector:
  Determine the primary industry the person works in (e.g. healthcare, software, retail, education,
  finance, manufacturing). Use the job title as the primary signal for industry, and tasks as
  confirmation. A task that is incidental to the role (e.g. a sales rep who also translates)
  should NOT override the industry identified from the job title.

STEP 2 — Identify the role within that industry:
  Find the O*NET occupation that best matches what this person primarily does day-to-day.
  Choose the most specific code available for that industry + role combination.
  If a worker performs a secondary function (e.g. translation, admin) alongside their primary role,
  match to the primary role, not the secondary one.

Additional guidance:
- PhD students doing technical research → match to the research scientist code for their field
- Students who also teach → match to their primary research role, not the teaching role
- Entrepreneurs → match to the functional role they perform day-to-day (e.g. buyer, manager)
- Interns → match to the occupation they are training for

## Participants (with their actual task statements)
{participants}

## O*NET Occupations (code | title | description)
{onet_list}

Return a JSON array with one object per participant, in the same order:
- "user_id": copy exactly
- "industry": the industry/sector identified in Step 1 (e.g. "electric vehicle / automotive")
- "category": a broad occupation category for grouping similar roles (e.g. "Software Engineering",
              "Data & ML", "Research", "Sales & Business Development", "Operations", "HR & Admin",
              "Design", "Finance & Analysis", "Teaching & Education", "Content Creation",
              "Students & Researchers in Training"). Use an existing category when it fits; create a new
              one only if none of the above apply.
- "onet_code": matched O*NET-SOC code (e.g. "15-2051.00")
- "onet_title": matched O*NET title (copy exactly)
- "match_notes": one sentence explaining the industry + role match

Return ONLY the JSON array. No markdown fences.
"""

BATCH_SIZE = 6  # participants per LLM call


def load_onet(path: Path) -> list[dict]:
    wb = openpyxl.load_workbook(path)
    ws = wb["Occupation Data"]
    return [
        {"code": row[0], "title": row[1], "description": row[2] or ""}
        for row in ws.iter_rows(min_row=2, values_only=True)
        if row[0] and row[1]
    ]


def build_onet_list(onet: list[dict]) -> str:
    lines = []
    for o in onet:
        # First sentence of description only
        desc = o["description"].split(".")[0].strip() if o["description"] else ""
        lines.append(f"{o['code']} | {o['title']} | {desc}")
    return "\n".join(lines)


def build_participant_text(record: dict) -> str:
    tasks = record.get("tasks", [])
    task_lines = "\n".join(
        f"  - {t.get('task_statement', '')}" for t in tasks if t.get("task_statement")
    )
    return (
        f"user_id: {record['user_id']}\n"
        f"Job title: {record['occupation']}\n"
        f"Tasks they perform:\n{task_lines}"
    )


def match_batch(records: list[dict], onet_list_text: str, llm: LLMClient) -> list[dict]:
    participants_text = "\n\n---\n\n".join(build_participant_text(r) for r in records)
    prompt = MATCH_PROMPT.format(
        participants=participants_text,
        onet_list=onet_list_text,
    )
    response = llm.call(prompt, model="gpt-5.4", temperature=0.0, max_tokens=4096).strip()
    for fence in ("```json", "```"):
        if response.startswith(fence):
            response = response[len(fence):]
    if response.endswith("```"):
        response = response[:-3]
    return json.loads(response.strip())


def main():
    root = Path(__file__).parent.parent.parent
    onet_path = root / "analysis/onet/Occupation Data O*NET.xlsx"
    tasks_path = root / "analysis/onet/data/study_tasks.json"

    onet = load_onet(onet_path)
    onet_list_text = build_onet_list(onet)
    print(f"Loaded {len(onet)} O*NET occupations")

    with open(tasks_path) as f:
        data = json.load(f)

    print(f"Matching {len(data)} participants in batches of {BATCH_SIZE}...\n")

    batches = [data[i:i + BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)]
    match_map = {}

    def process(batch):
        llm = LLMClient()
        return match_batch(batch, onet_list_text, llm)

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(process, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Batches"):
            try:
                results = fut.result()
                for m in results:
                    match_map[m["user_id"]] = m
            except Exception as e:
                batch = futures[fut]
                print(f"  [ERROR] batch starting {batch[0]['user_id']}: {e}")

    for r in data:
        m = match_map.get(r["user_id"], {})
        r["category"] = m.get("category", "Other")
        r["onet_code"] = m.get("onet_code", "")
        r["onet_title"] = m.get("onet_title", "")
        r["onet_industry"] = m.get("industry", "")
        r["onet_match_notes"] = m.get("match_notes", "")

    with open(tasks_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'Occupation':<45} {'Industry':<30} {'O*NET Code':<15} {'O*NET Title'}")
    print("-" * 140)
    for r in data:
        print(
            f"{r['occupation']:<45} {r.get('onet_industry',''):<30} "
            f"{r.get('onet_code',''):<15} {r.get('onet_title','')}"
        )

    print(f"\nDone — {len(data)} participants mapped → {tasks_path}")


if __name__ == "__main__":
    main()
