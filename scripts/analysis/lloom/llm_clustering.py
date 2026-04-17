"""
LLM-driven clustering of memory bank content using GPT-4o.

Three-pass approach:
  Pass 1 — Send batches of memory descriptors, ask GPT-4o to propose themes
  Pass 2 — Consolidate proposed themes into canonical cluster taxonomy
  Pass 3 — Assign each memory to canonical clusters (batched)

Then compute: purity, intra-cos, silhouette, emergent flag — same as hierarchical_clustering.py.
Saves: user_study/llm_clustering.json  +  user_study/llm_cluster_labels_cache.json

Usage:
    .venv-analysis/bin/python scripts/analysis/llm_clustering.py
"""

import json
import os
import textwrap
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

load_dotenv(dotenv_path=".env")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
OUT_PATH = USER_STUDY_DIR / "llm_clustering.json"
LABELS_CACHE = USER_STUDY_DIR / "llm_cluster_labels_cache.json"

BATCH_SIZE = 120          # memories per Pass 1 batch
ASSIGN_BATCH_SIZE = 50    # memories per Pass 3 assignment batch


# ── Load ──────────────────────────────────────────────────────────────────────

def load_topics():
    with open(TOPICS_PATH) as f:
        topics = json.load(f)
    subtopics = {}
    for i, t in enumerate(topics):
        for j, desc in enumerate(t["subtopics"]):
            sid = f"{i+1}.{j+1}"
            subtopics[sid] = {"topic_idx": i, "topic_name": t["topic"], "desc": desc}
    return topics, subtopics


def load_memories_and_embeddings():
    memories, embeddings = [], []
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        emb_path = USER_STUDY_DIR / pid / "memory_bank_embeddings.json"
        if not mem_path.is_file() or not emb_path.is_file():
            continue
        with open(mem_path) as f:
            mem_data = json.load(f)
        with open(emb_path) as f:
            emb_data = json.load(f)
        emb_lookup = {e["id"]: e["embedding"] for e in emb_data.get("embeddings", [])}
        for m in mem_data.get("memories", []):
            if m.get("id") in emb_lookup:
                memories.append({"pid": pid, **m})
                embeddings.append(emb_lookup[m["id"]])
    return memories, np.array(embeddings, dtype=np.float32)


# ── Helpers ───────────────────────────────────────────────────────────────────

def memory_descriptor(m, subtopics, short=False):
    """Compact one-line descriptor for a memory."""
    links = m.get("subtopic_links", [])
    top = max(links, key=lambda l: l.get("importance", 0)) if links else None
    sub_str = top["subtopic_id"] if top else "?"
    if short:
        return f"{m['id'][:8]}: {m['title'][:60]}"
    return f"[{m['id'][:8]}] {m['title']} | subtopic: {sub_str}"


def gpt(client, messages, temperature=0.3, json_mode=False):
    kwargs = dict(
        model="gpt-4o",
        messages=messages,
        temperature=temperature,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ── Pass 1: Propose themes from batches ───────────────────────────────────────

def pass1_propose_themes(client, memories, subtopics):
    print(f"  Pass 1: proposing themes from {len(memories)} memories in batches of {BATCH_SIZE}...")
    all_themes = []

    for batch_start in range(0, len(memories), BATCH_SIZE):
        batch = memories[batch_start:batch_start + BATCH_SIZE]
        descriptors = "\n".join(
            f"{i+1}. {memory_descriptor(m, subtopics)}"
            for i, m in enumerate(batch)
        )
        prompt = (
            "You are analyzing interview memories from a user study about AI adoption at work.\n"
            "Below is a batch of memory entries (title + interview subtopic tag).\n\n"
            "Identify the distinct THEMES you see in this batch. Each theme should:\n"
            "- Capture a coherent concept that groups multiple memories\n"
            "- Be described in 3-6 words (noun/gerund phrase)\n"
            "- Be specific enough to be distinct from adjacent themes\n\n"
            "Return ONLY a JSON object with key \"themes\": list of theme name strings.\n"
            "Propose as many themes as the data warrants — don't force a fixed number.\n\n"
            f"Memories:\n{descriptors}"
        )
        raw = gpt(client, [{"role": "user", "content": prompt}], json_mode=True)
        result = json.loads(raw)
        themes = result.get("themes", [])
        all_themes.extend(themes)
        print(f"    Batch {batch_start//BATCH_SIZE + 1}/{-(-len(memories)//BATCH_SIZE)}: {len(themes)} themes proposed")

    print(f"  Total raw themes: {len(all_themes)}")
    return all_themes


# ── Pass 2: Consolidate into canonical taxonomy ───────────────────────────────

CONSOLIDATE_BATCH = 60   # themes per intermediate consolidation batch

def consolidate_batch(client, themes, context=""):
    themes_str = "\n".join(f"- {t}" for t in themes)
    prompt = (
        "You are building a cluster taxonomy for a user study about AI adoption at work.\n"
        f"{context}"
        "Below are theme proposals — many are overlapping or redundant.\n\n"
        "Merge overlapping themes, split compound ones, remove near-duplicates.\n"
        "Keep only genuinely distinct, coherent concepts.\n"
        "Name each with a clear 3-6 word phrase.\n"
        "Decide the count yourself.\n\n"
        "Return ONLY JSON: {\"clusters\": [\"name1\", \"name2\", ...]}\n\n"
        f"Themes:\n{themes_str}"
    )
    raw = gpt(client, [{"role": "user", "content": prompt}], json_mode=True)
    return json.loads(raw).get("clusters", [])


def pass2_consolidate(client, raw_themes):
    print(f"  Pass 2: consolidating {len(raw_themes)} raw themes...")

    # Step 1: consolidate in batches of CONSOLIDATE_BATCH
    intermediate = []
    for i in range(0, len(raw_themes), CONSOLIDATE_BATCH):
        batch = raw_themes[i:i + CONSOLIDATE_BATCH]
        merged = consolidate_batch(client, batch)
        intermediate.extend(merged)
        print(f"    Batch {i//CONSOLIDATE_BATCH + 1}: {len(batch)} → {len(merged)} themes")

    print(f"  Intermediate: {len(intermediate)} themes — final merge...")

    # Step 2: final merge of intermediate themes
    canonical = consolidate_batch(
        client, intermediate,
        context="These are already partially consolidated themes. Do a final merge.\n"
    )
    print(f"  Canonical clusters: {len(canonical)}")
    for i, c in enumerate(canonical):
        print(f"    {i:3d}: {c}")
    return canonical


# ── Pass 3: Assign memories to clusters ───────────────────────────────────────

def assign_batch(client, batch, canonical_clusters):
    """Assign a batch of memories using sequential numbering to avoid ID confusion."""
    clusters_str = "\n".join(f"{i}: {c}" for i, c in enumerate(canonical_clusters))
    # Number each memory 0..n-1 within the batch
    lines = "\n".join(f"{i}: {m['title'][:70]}" for i, m in enumerate(batch))
    prompt = (
        "Assign each numbered memory to the best-fitting cluster index.\n"
        "Return ONLY JSON: {\"0\": <cluster_idx>, \"1\": <cluster_idx>, ...}\n"
        "Use the memory's sequential number (0-based) as key, cluster index as integer value.\n"
        f"Every number 0 to {len(batch)-1} must appear.\n\n"
        f"Clusters:\n{clusters_str}\n\n"
        f"Memories:\n{lines}"
    )
    raw = gpt(client, [{"role": "user", "content": prompt}], json_mode=True)
    result = json.loads(raw)
    # Map sequential index → memory id
    assignments = {}
    for seq_str, cid in result.items():
        try:
            seq = int(seq_str)
            if 0 <= seq < len(batch):
                assignments[batch[seq]["id"]] = int(cid)
        except (ValueError, TypeError):
            pass
    return assignments


def pass3_assign(client, memories, canonical_clusters, subtopics):
    print(f"  Pass 3: assigning {len(memories)} memories to {len(canonical_clusters)} clusters...")
    id_to_cluster = {}
    n_batches = -(-len(memories) // ASSIGN_BATCH_SIZE)

    for batch_start in range(0, len(memories), ASSIGN_BATCH_SIZE):
        batch = memories[batch_start:batch_start + ASSIGN_BATCH_SIZE]
        batch_num = batch_start // ASSIGN_BATCH_SIZE + 1
        try:
            assigned = assign_batch(client, batch, canonical_clusters)
            id_to_cluster.update(assigned)
            print(f"    Batch {batch_num}/{n_batches}: assigned {len(assigned)}/{len(batch)}")
        except Exception as e:
            print(f"    Batch {batch_num}/{n_batches}: ERROR {e} — retrying in halves")
            for sub_start in range(0, len(batch), 25):
                sub = batch[sub_start:sub_start + 25]
                try:
                    assigned2 = assign_batch(client, sub, canonical_clusters)
                    id_to_cluster.update(assigned2)
                    print(f"      Sub-batch: assigned {len(assigned2)}/{len(sub)}")
                except Exception as e2:
                    print(f"      Sub-batch: FAILED {e2}, skipping")

    missing = [m for m in memories if m["id"] not in id_to_cluster]
    if missing:
        print(f"  Warning: {len(missing)} memories unassigned, defaulting to cluster 0")
        for m in missing:
            id_to_cluster[m["id"]] = 0

    return id_to_cluster


# ── Compute cluster stats ─────────────────────────────────────────────────────

def compute_stats(cid, idxs, memories, embeddings, subtopics):
    n = len(idxs)
    n_part = len(set(memories[i]["pid"] for i in idxs))

    # Purity
    topic_counts = Counter()
    for i in idxs:
        links = memories[i].get("subtopic_links", [])
        if links:
            best = max(links, key=lambda l: l.get("importance", 0))
            tid = int(best["subtopic_id"].split(".")[0]) - 1
            topic_counts[tid] += 1
    dom_topic = topic_counts.most_common(1)[0][0] if topic_counts else 0
    purity = topic_counts[dom_topic] / n if topic_counts else 0.0

    # Intra-cluster cosine
    sample = list(idxs) if n <= 150 else list(np.random.choice(idxs, 150, replace=False))
    emb_slice = embeddings[sample]
    sims = cosine_similarity(emb_slice)
    mask = np.triu(np.ones(sims.shape, dtype=bool), k=1)
    intra_cos = float(sims[mask].mean()) if mask.sum() > 0 else 0.0

    is_emergent = purity < 0.60 and intra_cos >= 0.55

    sample_titles = [memories[i]["title"] for i in list(idxs)[:5]]
    memory_ids = [memories[i]["id"] for i in idxs]

    return {
        "n": n,
        "n_part": n_part,
        "dom_topic": dom_topic,
        "purity": round(purity, 4),
        "cos": round(intra_cos, 4),
        "is_emergent": is_emergent,
        "sample_titles": sample_titles,
        "memory_ids": memory_ids,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("LLM CLUSTERING (GPT-4o)")
    print("=" * 55)

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print("\n[1/5] Loading data...")
    topics, subtopics = load_topics()
    memories, embeddings = load_memories_and_embeddings()
    print(f"  {len(memories)} memories, {len(set(m['pid'] for m in memories))} participants")

    print("\n[2/5] Pass 1 — Propose themes from batches...")
    raw_themes = pass1_propose_themes(client, memories, subtopics)

    print("\n[3/5] Pass 2 — Consolidate into canonical taxonomy...")
    canonical_clusters = pass2_consolidate(client, raw_themes)

    print("\n[4/5] Pass 3 — Assign memories to clusters...")
    id_to_cluster = pass3_assign(client, memories, canonical_clusters, subtopics)

    # Build label array aligned to memories list
    labels = np.array([id_to_cluster.get(m["id"], 0) for m in memories])

    print("\n[5/5] Computing metrics...")

    # Silhouette on embeddings (sample for speed)
    sample_size = min(1000, len(memories))
    sample_idxs = np.random.choice(len(memories), sample_size, replace=False)
    sil = silhouette_score(embeddings[sample_idxs], labels[sample_idxs], metric="cosine")
    print(f"  Silhouette score (cosine, n={sample_size}): {sil:.4f}")

    # Per-cluster stats
    clusters = {}
    unique_cids = sorted(set(labels))
    for cid in unique_cids:
        idxs = np.where(labels == cid)[0]
        if cid < len(canonical_clusters):
            label = canonical_clusters[cid]
        else:
            label = f"Cluster {cid}"
        stats = compute_stats(cid, idxs, memories, embeddings, subtopics)
        stats["label"] = label
        clusters[str(cid)] = stats

    emergent = [k for k, v in clusters.items() if v["is_emergent"]]
    print(f"  {len(clusters)} clusters, {len(emergent)} emergent")

    # Save labels cache (for subtopic_scatter.py)
    with open(LABELS_CACHE, "w") as f:
        json.dump(id_to_cluster, f)
    print(f"  Labels cache → {LABELS_CACHE}")

    # Save full results
    out = {
        "n_clusters": len(canonical_clusters),
        "silhouette_score": round(float(sil), 4),
        "canonical_clusters": canonical_clusters,
        "clusters": clusters,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results     → {OUT_PATH}")

    print("\nAll clusters:")
    for k in sorted(clusters.keys(), key=int):
        v = clusters[k]
        flag = " ⭐" if v["is_emergent"] else ""
        print(f"  C{k:>3s}  n={v['n']:3d}  part={v['n_part']:2d}  "
              f"purity={v['purity']:.2f}  cos={v['cos']:.3f}  "
              f"\"{v['label']}\"{flag}")

    print("\nEmergent clusters:")
    for k in sorted(emergent, key=lambda k: clusters[k]["purity"]):
        v = clusters[k]
        print(f"  C{k:>3s}  n={v['n']:3d}  part={v['n_part']:2d}  "
              f"purity={v['purity']:.2f}  cos={v['cos']:.3f}  \"{v['label']}\"")


if __name__ == "__main__":
    main()
