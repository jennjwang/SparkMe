"""
Hybrid LLooM: LLM open-coding concept discovery + LLooM soft scoring.

Concept discovery (no embedding clustering — full topic coverage):
  Pass 1 — GPT-4o proposes themes from batches of memory titles
  Pass 2 — Consolidate into canonical concept taxonomy
  Pass 3 — Generate LLooM-style eval prompt for each concept

Scoring (richer than hard assignment):
  LLooM score() — soft 0-1 per (memory, concept) with rationale + highlight

Output: user_study/lloom_results.json  (same format → lloom_viewer.py still works)

Usage:
    .venv-analysis/bin/python scripts/analysis/lloom_hybrid.py
"""

import asyncio
import json
import os
import uuid
from pathlib import Path

import builtins
import pandas as pd
import openai
from dotenv import load_dotenv

# Auto-confirm LLooM's interactive prompts
_real_input = builtins.input
def _auto_input(prompt=""):
    print(prompt + "y")
    return "y"
builtins.input = _auto_input

# Stub display() for non-notebook context
try:
    display
except NameError:
    import IPython.display as _ipd
    builtins.display = _ipd.display
except Exception:
    builtins.display = lambda *a, **kw: None

load_dotenv(dotenv_path=".env")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
OUT_PATH = USER_STUDY_DIR / "lloom_results.json"

PROPOSE_BATCH = 120    # memories per Pass 1 batch
CONSOLIDATE_BATCH = 60 # themes per intermediate consolidation batch
CONCEPTS_CACHE = USER_STUDY_DIR / "lloom_hybrid_concepts.json"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_memories():
    memories = []
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        if not mem_path.is_file():
            continue
        with open(mem_path) as f:
            mem_data = json.load(f)
        for m in mem_data.get("memories", []):
            links = m.get("subtopic_links", [])
            top = max(links, key=lambda l: l.get("importance", 0)) if links else None
            memories.append({
                "id": m["id"],
                "pid": pid,
                "title": m.get("title", ""),
                "text": m.get("text", ""),
                "top_subtopic": top["subtopic_id"] if top else "",
                "doc": f"{m.get('title', '')}. {m.get('text', '')}",
            })
    return memories


# ── Pass 1: Propose themes from batches of memory titles ─────────────────────

def gpt(client, messages, temperature=0.3, json_mode=False):
    kwargs = dict(model="gpt-4o", messages=messages, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return client.chat.completions.create(**kwargs).choices[0].message.content


def pass1_propose_themes(client, memories):
    print(f"  Pass 1: proposing themes from {len(memories)} memories "
          f"in batches of {PROPOSE_BATCH}...")
    all_themes = []
    n_batches = -(-len(memories) // PROPOSE_BATCH)

    for i in range(0, len(memories), PROPOSE_BATCH):
        batch = memories[i:i + PROPOSE_BATCH]
        descriptors = "\n".join(
            f"{j+1}. [{m['top_subtopic'] or '?'}] {m['title']}"
            for j, m in enumerate(batch)
        )
        prompt = (
            "You are analyzing interview memories from a user study about AI adoption at work.\n"
            "Below are memory entries tagged with interview subtopic IDs (1.x = background, "
            "2.x = responsibilities, 3.x = task proficiency, 4.x = tech learning, "
            "5.x = tools used, 6.x = AI experience, 7.x = AI interaction style, "
            "8.x = AI trust, 9.x = AI skill impact, 10.x = AI future outlook).\n\n"
            "Identify distinct THEMES across this batch. Each theme should:\n"
            "- Capture a coherent concept grouping multiple memories\n"
            "- Be 3–6 words (noun or gerund phrase)\n"
            "- Cover the FULL range of topics — not just AI use, but also background, "
            "job roles, tools, learning attitudes, etc.\n\n"
            "Return ONLY JSON: {\"themes\": [\"theme 1\", \"theme 2\", ...]}\n\n"
            f"Memories:\n{descriptors}"
        )
        raw = gpt(client, [{"role": "user", "content": prompt}], json_mode=True)
        themes = json.loads(raw).get("themes", [])
        all_themes.extend(themes)
        batch_num = i // PROPOSE_BATCH + 1
        print(f"    Batch {batch_num}/{n_batches}: {len(themes)} themes proposed")

    print(f"  Total raw themes: {len(all_themes)}")
    return all_themes


# ── Pass 2: Consolidate into canonical concept taxonomy ───────────────────────

def _consolidate_batch(client, themes, context=""):
    themes_str = "\n".join(f"- {t}" for t in themes)
    prompt = (
        "You are building a concept taxonomy for a user study about AI adoption at work.\n"
        f"{context}"
        "Merge overlapping themes, split compound ones, remove near-duplicates.\n"
        "Keep genuinely distinct concepts — including background, job roles, tools, and "
        "learning attitudes, not only AI-specific themes.\n"
        "Name each with a clear 3–6 word phrase.\n\n"
        "Return ONLY JSON: {\"concepts\": [\"name1\", \"name2\", ...]}\n\n"
        f"Themes:\n{themes_str}"
    )
    raw = gpt(client, [{"role": "user", "content": prompt}], json_mode=True)
    return json.loads(raw).get("concepts", [])


def pass2_consolidate(client, raw_themes):
    print(f"  Pass 2: consolidating {len(raw_themes)} raw themes...")

    intermediate = []
    for i in range(0, len(raw_themes), CONSOLIDATE_BATCH):
        batch = raw_themes[i:i + CONSOLIDATE_BATCH]
        merged = _consolidate_batch(client, batch)
        intermediate.extend(merged)
        print(f"    Batch {i//CONSOLIDATE_BATCH + 1}: {len(batch)} → {len(merged)}")

    print(f"  Intermediate: {len(intermediate)} — final merge...")
    canonical = _consolidate_batch(
        client, intermediate,
        context="These are already partially consolidated. Do a final merge.\n"
    )
    print(f"  Canonical concepts: {len(canonical)}")
    for i, c in enumerate(canonical):
        print(f"    {i:3d}: {c}")
    return canonical


# ── Pass 3: Generate LLooM-style eval prompts ─────────────────────────────────

def pass3_eval_prompts(client, canonical_names):
    """For each concept name, generate a 1-sentence LLM eval prompt."""
    print(f"  Pass 3: generating eval prompts for {len(canonical_names)} concepts...")
    names_str = "\n".join(f"{i}: {n}" for i, n in enumerate(canonical_names))
    prompt = (
        "For each numbered concept from a user study about AI adoption at work, "
        "write a single-sentence evaluation prompt. The prompt will be used by an LLM "
        "to judge whether a short memory snippet is related to that concept.\n\n"
        "Requirements:\n"
        "- Start with 'Does this memory describe' or 'Does this memory relate to'\n"
        "- Be specific enough to distinguish this concept from adjacent ones\n"
        "- End with a question mark\n\n"
        "Return ONLY JSON: {\"prompts\": {\"0\": \"...\", \"1\": \"...\", ...}}\n\n"
        f"Concepts:\n{names_str}"
    )
    raw = gpt(client, [{"role": "user", "content": prompt}], json_mode=True, temperature=0.2)
    prompts_map = json.loads(raw).get("prompts", {})

    result = []
    for i, name in enumerate(canonical_names):
        eval_prompt = prompts_map.get(str(i), f"Does this memory relate to {name}?")
        result.append({"name": name, "prompt": eval_prompt})
        print(f"    [{i:3d}] {name[:40]:<40} → {eval_prompt[:60]}...")
    return result


# ── Build LLooM workbench with injected concepts ──────────────────────────────

async def score_with_lloom(df, concept_defs):
    """Inject open-coded concepts into a LLooM workbench and run soft scoring."""
    import text_lloom.workbench as wb
    from text_lloom.concept import Concept

    print(f"  Building LLooM workbench ({len(df)} memories, {len(concept_defs)} concepts)...")
    l = wb.lloom(df=df, text_col="doc", id_col="id")

    # Inject concepts — no clustering needed
    l.concepts = {}
    for cd in concept_defs:
        c = Concept(
            name=cd["name"],
            prompt=cd["prompt"],
            example_ids=set(),
            active=True,
            summary="",
            seed=None,
        )
        l.concepts[c.id] = c

    # Both df_filtered and df_to_score must be set for the workbench internals
    doc_df = df[["id", "doc"]].copy()
    l.df_filtered = doc_df
    l.df_to_score = doc_df

    print(f"  Running LLooM score() — {len(l.concepts)} concepts × {len(df)} memories...")
    score_df = await l.score(
        score_all=True,
        batch_size=5,
        get_highlights=True,
        debug=True,
    )
    return l, score_df


# ── Save results (same format as lloom_clustering.py) ─────────────────────────

def save_results(l, score_df, memories_df):
    concepts = []
    for c_id, concept in l.concepts.items():
        concepts.append({
            "id": c_id,
            "name": getattr(concept, "name", str(c_id)),
            "prompt": getattr(concept, "prompt", ""),
            "summary": getattr(concept, "summary", ""),
        })

    if score_df is not None and len(score_df) > 0:
        sdf = score_df.copy()
        sdf["doc_id"] = sdf["doc_id"].astype(str)
        meta = memories_df[["id", "pid", "title", "top_subtopic"]].copy()
        meta["id"] = meta["id"].astype(str)
        merged = sdf.merge(meta, left_on="doc_id", right_on="id", how="left").drop(columns=["id"])
        records = merged.to_dict(orient="records")
    else:
        records = []

    out = {
        "n_concepts": len(concepts),
        "concepts": concepts,
        "scores": records,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results → {OUT_PATH}")
    print(f"  {len(concepts)} concepts, {len(records)} scored pairs")
    for c in concepts:
        print(f"    [{c['id'][:8]}] {c['name']}")


# ── Main ──────────────────────────────────────────────────────────────────────

async def main():
    print("=" * 55)
    print("HYBRID LLooM: Open-coding discovery + LLooM scoring")
    print("=" * 55)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("\n[1/5] Loading memories...")
    memories = load_memories()
    print(f"  {len(memories)} memories from {len(set(m['pid'] for m in memories))} participants")
    df = pd.DataFrame(memories)

    if CONCEPTS_CACHE.is_file():
        print(f"\n[2-4/5] Loading cached concepts from {CONCEPTS_CACHE}...")
        with open(CONCEPTS_CACHE) as f:
            concept_defs = json.load(f)
        print(f"  {len(concept_defs)} concepts loaded")
    else:
        print("\n[2/5] Pass 1 — Proposing themes...")
        raw_themes = pass1_propose_themes(client, memories)

        print("\n[3/5] Pass 2 — Consolidating into canonical concepts...")
        canonical_names = pass2_consolidate(client, raw_themes)

        print("\n[4/5] Pass 3 — Generating eval prompts...")
        concept_defs = pass3_eval_prompts(client, canonical_names)

        with open(CONCEPTS_CACHE, "w") as f:
            json.dump(concept_defs, f, indent=2)
        print(f"  Cached → {CONCEPTS_CACHE}")

    print("\n[5/5] Scoring with LLooM...")
    l, score_df = await score_with_lloom(df, concept_defs)

    print("\nSaving results...")
    save_results(l, score_df, df)


if __name__ == "__main__":
    asyncio.run(main())
