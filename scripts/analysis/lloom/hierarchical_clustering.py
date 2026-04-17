"""
Agglomerative clustering on memory embeddings.

Dynamically selects n_clusters by maximizing silhouette score over a sweep,
then saves hierarchical_clustering.json with per-cluster stats AND memory IDs
so the result is reproducible without re-running clustering.

Usage:
    .venv-analysis/bin/python scripts/analysis/hierarchical_clustering.py
"""

import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

from umap import UMAP
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import openai

load_dotenv(dotenv_path=".env")

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
OUT_PATH = USER_STUDY_DIR / "hierarchical_clustering.json"
LABELS_CACHE = USER_STUDY_DIR / "cluster_labels_cache.json"

N_CLUSTERS_MIN = 10
N_CLUSTERS_MAX = 80
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_RANDOM_STATE = 42


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


# ── UMAP reduction ────────────────────────────────────────────────────────────

def reduce(embeddings, n_components):
    print(f"  UMAP {embeddings.shape[1]}D → {n_components}D ...")
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.0,
        metric="cosine",
        random_state=UMAP_RANDOM_STATE,
    )
    return reducer.fit_transform(embeddings)


# ── Silhouette sweep ──────────────────────────────────────────────────────────

def find_optimal_n_clusters(reduced, n_min, n_max):
    print(f"  Sweeping n_clusters {n_min}–{n_max} ...")
    best_n, best_score = n_min, -1
    scores = {}
    for n in range(n_min, n_max + 1, 2):  # step 2 for speed
        labels = AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(reduced)
        score = silhouette_score(reduced, labels, sample_size=500, random_state=42)
        scores[n] = round(float(score), 4)
        if score > best_score:
            best_score = score
            best_n = n
        print(f"    n={n:3d}  sil={score:.4f}" + (" ← best" if n == best_n else ""))
    return best_n, best_score, scores


# ── Per-cluster stats ─────────────────────────────────────────────────────────

def cluster_stats(cid, idxs, memories, embeddings, subtopics, topics):
    n = len(idxs)
    n_part = len(set(memories[i]["pid"] for i in idxs))

    # Purity (dominant existing topic)
    topic_counts = Counter()
    for i in idxs:
        links = memories[i].get("subtopic_links", [])
        if links:
            best = max(links, key=lambda l: l.get("importance", 0))
            tid = int(best["subtopic_id"].split(".")[0]) - 1
            topic_counts[tid] += 1
    dom_topic = topic_counts.most_common(1)[0][0] if topic_counts else 0
    purity = topic_counts[dom_topic] / n if topic_counts else 0.0

    # Intra-cluster cosine (sample up to 150)
    sample = list(idxs) if n <= 150 else list(np.random.choice(idxs, 150, replace=False))
    emb_slice = embeddings[sample]
    sims = cosine_similarity(emb_slice)
    mask = np.triu(np.ones(sims.shape, dtype=bool), k=1)
    intra_cos = float(sims[mask].mean()) if mask.sum() > 0 else 0.0

    # Emergent flag
    is_emergent = purity < 0.60 and intra_cos >= 0.55

    # Top subtopic anchors (closest existing subtopics by centroid similarity)
    centroid = embeddings[list(idxs)].mean(axis=0, keepdims=True)
    top_subtopics = []
    for sid, info in subtopics.items():
        # Anchor = mean embedding of memories linked to this subtopic (importance >= 6)
        anchor_idxs = [
            j for j in range(len(memories))
            for lnk in memories[j].get("subtopic_links", [])
            if lnk["subtopic_id"] == sid and lnk.get("importance", 0) >= 6
        ]
        if not anchor_idxs:
            continue
        anchor = embeddings[anchor_idxs].mean(axis=0, keepdims=True)
        sim = float(cosine_similarity(centroid, anchor)[0, 0])
        top_subtopics.append((sid, info["desc"], info["topic_idx"], sim))
    top_subtopics.sort(key=lambda x: -x[3])
    top_subtopics = [[s[0], s[1], s[2], round(s[3], 4)] for s in top_subtopics[:3]]

    sample_titles = [memories[i]["title"] for i in list(idxs)[:5]]
    memory_ids = [memories[i]["id"] for i in idxs]

    return {
        "n": n,
        "n_part": n_part,
        "dom_topic": dom_topic,
        "purity": round(purity, 4),
        "cos": round(intra_cos, 4),
        "is_emergent": is_emergent,
        "top_subtopics": top_subtopics,
        "sample_titles": sample_titles,
        "memory_ids": memory_ids,
    }


# ── LLM labeling ─────────────────────────────────────────────────────────────

def generate_labels(clusters):
    """Call GPT-4o once to label all clusters. Returns dict cid_str → label."""
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    blocks = []
    for cid_str, meta in sorted(clusters.items(), key=lambda x: int(x[0])):
        titles = "; ".join(meta["sample_titles"][:6])
        top_subs = ", ".join(f"{s[0]} ({s[1][:35]})" for s in meta["top_subtopics"][:2])
        emergent = " [EMERGENT - spans multiple topics]" if meta["is_emergent"] else ""
        blocks.append(
            f"Cluster {cid_str}{emergent}\n"
            f"  n={meta['n']} memories, {meta['n_part']} participants\n"
            f"  purity={meta['purity']:.2f}, intra-cos={meta['cos']:.3f}\n"
            f"  Nearest subtopics: {top_subs}\n"
            f"  Sample titles: {titles}"
        )

    prompt = (
        "You are analyzing a user study about AI tool adoption at work. "
        "Below are agglomerative clusters of ~2,200 participant memories.\n\n"
        "For each cluster, generate a SHORT (3-6 word) human-readable label capturing its core theme.\n"
        "Rules:\n"
        "- Natural noun or gerund phrase (e.g. 'Verifying AI Outputs', 'Self-Directed AI Learning')\n"
        "- Distinct from all other labels\n"
        "- Avoid generic phrases like 'AI tools usage' or 'user experience'\n"
        "- For EMERGENT clusters, the label should reflect what's new/cross-cutting\n\n"
        "Return ONLY a JSON object: {\"0\": \"label\", \"1\": \"label\", ...}\n\n"
        "Clusters:\n\n" + "\n\n".join(blocks)
    )

    print(f"  Calling GPT-4o to label {len(clusters)} clusters...", end=" ", flush=True)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    labels = json.loads(response.choices[0].message.content)
    print("done")
    return {str(k): v for k, v in labels.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 55)

    print("\n[1/4] Loading data...")
    topics, subtopics = load_topics()
    memories, embeddings = load_memories_and_embeddings()
    print(f"  {len(memories)} memories, {len(set(m['pid'] for m in memories))} participants")

    print("\n[2/4] UMAP reduction...")
    reduced = reduce(embeddings, UMAP_N_COMPONENTS)

    print("\n[3/4] Silhouette sweep to find optimal n_clusters...")
    best_n, best_score, all_scores = find_optimal_n_clusters(reduced, N_CLUSTERS_MIN, N_CLUSTERS_MAX)
    print(f"\n  ✓ Optimal: n_clusters={best_n}  silhouette={best_score:.4f}")

    print(f"\n[4/4] Final clustering with n_clusters={best_n} ...")
    labels = AgglomerativeClustering(n_clusters=best_n, linkage="ward").fit_predict(reduced)

    clusters = {}
    for cid in sorted(set(labels)):
        idxs = np.where(labels == cid)[0]
        clusters[str(cid)] = cluster_stats(cid, idxs, memories, embeddings, subtopics, topics)

    emergent = [k for k, v in clusters.items() if v["is_emergent"]]
    print(f"  {len(clusters)} clusters, {len(emergent)} emergent")

    print("\n[5/5] LLM cluster labeling...")
    llm_labels = generate_labels(clusters)
    for cid_str, label in llm_labels.items():
        if cid_str in clusters:
            clusters[cid_str]["label"] = label

    # Save cluster_labels_cache.json for subtopic_scatter.py
    id_to_label = {memories[i]["id"]: int(labels[i]) for i in range(len(memories))}
    with open(LABELS_CACHE, "w") as f:
        json.dump(id_to_label, f)
    print(f"  Labels cache → {LABELS_CACHE}")

    # Save hierarchical_clustering.json
    out = {
        "n_clusters": best_n,
        "silhouette_score": round(best_score, 4),
        "silhouette_sweep": all_scores,
        "fine": clusters,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  Results     → {OUT_PATH}")

    print("\nTop emergent clusters:")
    for k in sorted(emergent, key=lambda k: clusters[k]["purity"]):
        v = clusters[k]
        print(f"  C{k:>3s}  n={v['n']:3d}  part={v['n_part']:2d}  "
              f"purity={v['purity']:.2f}  cos={v['cos']:.3f}  "
              f"\"{v.get('label', '?')}\"")

    print("\nAll clusters:")
    for k in sorted(clusters.keys(), key=int):
        v = clusters[k]
        flag = " ⭐" if v["is_emergent"] else ""
        print(f"  C{k:>3s}  n={v['n']:3d}  \"{v.get('label', '?')}\"{flag}")


if __name__ == "__main__":
    main()
