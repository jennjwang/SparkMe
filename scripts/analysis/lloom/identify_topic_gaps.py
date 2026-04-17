"""
Topic Gap Identification Script

Systematically identifies gaps in the existing 10-topic interview framework by:
1. Finding "orphaned" memories that don't link well to any existing subtopic
2. Clustering all memories using their embeddings to discover natural groupings
3. Measuring how well each cluster aligns with existing subtopic descriptions
4. Surfacing clusters with poor alignment as candidate new topics/subtopics

Usage:
    python scripts/identify_topic_gaps.py

Outputs:
    - user_study/topic_gap_analysis.md  (readable report)
    - user_study/topic_gap_analysis.json (structured data for downstream use)
"""

import json
import os
import numpy as np
import faiss
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
EXISTING_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
REPORT_PATH = USER_STUDY_DIR / "topic_gap_analysis.md"
JSON_PATH = USER_STUDY_DIR / "topic_gap_analysis.json"

# Clustering parameters
N_CLUSTERS = 40           # Number of clusters for K-Means (intentionally over-segments)
MIN_CLUSTER_SIZE = 10     # Ignore clusters smaller than this
MIN_PARTICIPANTS = 5      # A cluster must span this many participants to be interesting
GAP_THRESHOLD = 0.90      # Max cosine similarity to existing subtopics to count as a "gap"


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_existing_topics():
    """Load existing topics and build a flat list of subtopic descriptions."""
    with open(EXISTING_TOPICS_PATH) as f:
        topics = json.load(f)

    subtopics = []
    for i, topic in enumerate(topics):
        topic_id = str(i + 1)
        for j, sub_desc in enumerate(topic["subtopics"]):
            subtopics.append({
                "subtopic_id": f"{topic_id}.{j+1}",
                "topic_id": topic_id,
                "topic_name": topic["topic"],
                "description": sub_desc,
                # We'll build a richer text representation for embedding comparison
                "full_text": f"{topic['topic']}: {sub_desc}",
            })
    return topics, subtopics


def load_all_memories_with_embeddings():
    """Load all memories and their embeddings from all participants."""
    memories = []
    embeddings = []

    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        emb_path = USER_STUDY_DIR / pid / "memory_bank_embeddings.json"
        if not mem_path.is_file() or not emb_path.is_file():
            continue

        with open(mem_path) as f:
            mem_data = json.load(f)
        with open(emb_path) as f:
            emb_data = json.load(f)

        # Build embedding lookup by ID
        emb_lookup = {e["id"]: e["embedding"] for e in emb_data.get("embeddings", [])}

        for m in mem_data.get("memories", []):
            mid = m.get("id", "")
            if mid not in emb_lookup:
                continue
            memories.append({"pid": pid, **m})
            embeddings.append(emb_lookup[mid])

    return memories, np.array(embeddings, dtype=np.float32)


# ─── Analysis 1: Poorly-Linked Memories ─────────────────────────────────────

def find_poorly_linked_memories(memories):
    """
    Find memories with weak links to existing subtopics.
    These are memories where the system struggled to place them.
    """
    orphaned = []       # No subtopic links at all
    weak_fit = []       # Max importance <= 5
    cross_cutting = []  # Spans 3+ different top-level topics

    for m in memories:
        links = m.get("subtopic_links", [])
        if not links:
            orphaned.append(m)
            continue

        importances = [l["importance"] for l in links]
        max_imp = max(importances)
        topic_ids = set(l["subtopic_id"].split(".")[0] for l in links)

        if max_imp <= 5:
            weak_fit.append({"memory": m, "max_importance": max_imp})
        if len(topic_ids) >= 3:
            cross_cutting.append({"memory": m, "topics_spanned": len(topic_ids)})

    return orphaned, weak_fit, cross_cutting


# ─── Analysis 2: Embedding-Based Clustering ─────────────────────────────────

def run_clustering(embeddings, n_clusters=N_CLUSTERS):
    """Cluster all memory embeddings using K-Means."""
    # Normalize embeddings for cosine-like behavior with L2-based KMeans
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(normalized)
    centroids = kmeans.cluster_centers_

    return labels, centroids, normalized


def compute_subtopic_embeddings(subtopics, all_memories, all_embeddings):
    """
    Compute a representative embedding for each existing subtopic by averaging
    the embeddings of memories linked to it with high importance.
    """
    subtopic_embs = {}

    for sub in subtopics:
        sid = sub["subtopic_id"]
        linked_indices = []
        for i, m in enumerate(all_memories):
            for link in m.get("subtopic_links", []):
                if link["subtopic_id"] == sid and link["importance"] >= 7:
                    linked_indices.append(i)
                    break

        if linked_indices:
            emb_matrix = all_embeddings[linked_indices]
            subtopic_embs[sid] = emb_matrix.mean(axis=0)

    return subtopic_embs


def measure_cluster_gap(centroids, subtopic_embeddings, normalized_embeddings):
    """
    For each cluster centroid, compute its max cosine similarity to any
    existing subtopic embedding. Low similarity = potential gap.
    """
    if not subtopic_embeddings:
        return [0.0] * len(centroids)

    sub_matrix = np.array(list(subtopic_embeddings.values()), dtype=np.float32)
    # Normalize
    sub_norms = np.linalg.norm(sub_matrix, axis=1, keepdims=True)
    sub_norms[sub_norms == 0] = 1
    sub_matrix_norm = sub_matrix / sub_norms

    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroid_norms[centroid_norms == 0] = 1
    centroids_norm = centroids / centroid_norms

    # Cosine similarity: each centroid vs all subtopic embeddings
    sim_matrix = cosine_similarity(centroids_norm, sub_matrix_norm)
    max_sims = sim_matrix.max(axis=1)  # Best match per cluster

    sub_ids = list(subtopic_embeddings.keys())
    best_match_ids = [sub_ids[idx] for idx in sim_matrix.argmax(axis=1)]

    return max_sims, best_match_ids


def describe_cluster(cluster_memories):
    """
    Generate a description of a cluster from its memories using
    the most common title words and metadata.
    """
    # Title word frequency
    word_counts = Counter()
    for m in cluster_memories:
        words = m["title"].lower().split()
        for w in words:
            if len(w) > 3 and w not in {"user", "user's", "with", "that", "this",
                                         "from", "about", "their", "they", "have",
                                         "been", "more", "also", "does", "into"}:
                word_counts[w] += 1

    top_words = [w for w, _ in word_counts.most_common(8)]

    # Metadata key frequency
    meta_keys = Counter()
    meta_vals = Counter()
    for m in cluster_memories:
        md = m.get("metadata", {})
        for k, v in md.items():
            meta_keys[k] += 1
            if isinstance(v, str) and len(v) < 50:
                meta_vals[f"{k}={v}"] += 1

    top_meta = [kv for kv, _ in meta_vals.most_common(5)]

    return {
        "top_keywords": top_words,
        "top_metadata": top_meta,
        "sample_titles": [m["title"] for m in cluster_memories[:5]],
        "sample_texts": [m["text"][:150] for m in cluster_memories[:3]],
    }


# ─── Report Generation ──────────────────────────────────────────────────────

def generate_report(orphaned, weak_fit, cross_cutting, cluster_analysis, subtopics_flat):
    """Generate the markdown gap analysis report."""
    lines = [
        "# Topic Gap Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "This report identifies gaps in the existing 10-topic interview framework",
        "using two complementary methods:",
        "1. **Link quality analysis** — memories that the system struggled to place",
        "2. **Embedding-based clustering** — natural groupings that don't align with existing subtopics",
        "",
        "---",
        "",
        "## Method 1: Poorly-Linked Memories",
        "",
        f"### Orphaned memories (no subtopic links): {len(orphaned)}",
        "",
    ]

    if orphaned:
        for m in orphaned[:10]:
            lines.append(f"- **{m['title']}** [{m['pid'][:8]}...]")
            lines.append(f"  > {m['text'][:150]}...")
            lines.append("")

    lines.append(f"### Weak-fit memories (max importance <= 5): {len(weak_fit)}")
    lines.append("")
    if weak_fit:
        for item in weak_fit[:10]:
            m = item["memory"]
            lines.append(f"- **{m['title']}** (max_importance={item['max_importance']}) [{m['pid'][:8]}...]")
            lines.append(f"  > {m['text'][:150]}...")
            lines.append("")

    lines.append(f"### Cross-cutting memories (span 3+ topics): {len(cross_cutting)}")
    lines.append("")
    lines.append("These memories touch so many topics they may represent themes that")
    lines.append("don't fit neatly into the existing framework.")
    lines.append("")
    if cross_cutting:
        for item in cross_cutting[:10]:
            m = item["memory"]
            topic_ids = sorted(set(l["subtopic_id"].split(".")[0] for l in m.get("subtopic_links", [])))
            lines.append(f"- **{m['title']}** (topics: {', '.join(topic_ids)}) [{m['pid'][:8]}...]")
            lines.append(f"  > {m['text'][:150]}...")
            lines.append("")

    # Thematic summary of poorly-linked memories
    all_poor = [m for m in orphaned] + [item["memory"] for item in weak_fit]
    if all_poor:
        lines.append("### Themes among poorly-linked memories")
        lines.append("")
        word_counts = Counter()
        for m in all_poor:
            for w in m["title"].lower().split():
                if len(w) > 3 and w not in {"user", "user's", "with", "that", "this", "from", "about"}:
                    word_counts[w] += 1
        lines.append("Top keywords: " + ", ".join(f"**{w}** ({c})" for w, c in word_counts.most_common(15)))
        lines.append("")

    lines.extend([
        "---",
        "",
        "## Method 2: Embedding-Based Cluster Gap Analysis",
        "",
        f"All memories were clustered into {N_CLUSTERS} groups using K-Means on their",
        "1536-dim embeddings. Each cluster centroid was compared against existing subtopic",
        f"embeddings. Clusters with max cosine similarity < {GAP_THRESHOLD} to any existing",
        "subtopic are flagged as **gaps**.",
        "",
    ])

    # Build subtopic lookup
    sub_lookup = {s["subtopic_id"]: s for s in subtopics_flat}

    # Sort clusters: gaps first (lowest similarity), then by size
    gap_clusters = [c for c in cluster_analysis if c["is_gap"]]
    aligned_clusters = [c for c in cluster_analysis if not c["is_gap"]]

    lines.append(f"### Gap clusters (similarity < {GAP_THRESHOLD}): {len(gap_clusters)}")
    lines.append("")

    for c in sorted(gap_clusters, key=lambda x: x["max_similarity"]):
        lines.append(f"#### Cluster {c['cluster_id']} — similarity: {c['max_similarity']:.3f}")
        lines.append(f"- **Size:** {c['n_memories']} memories, {c['n_participants']} participants")
        best = c["best_match_subtopic"]
        sub_info = sub_lookup.get(best, {})
        lines.append(f"- **Nearest existing subtopic:** {best} ({sub_info.get('full_text', '?')})")
        lines.append(f"- **Top keywords:** {', '.join(c['description']['top_keywords'])}")
        if c["description"]["top_metadata"]:
            lines.append(f"- **Common metadata:** {', '.join(c['description']['top_metadata'][:3])}")
        lines.append(f"- **Sample titles:**")
        for t in c["description"]["sample_titles"]:
            lines.append(f"  - {t}")
        lines.append(f"- **Sample text:**")
        for t in c["description"]["sample_texts"]:
            lines.append(f"  > {t}...")
        lines.append("")

    lines.append(f"### Well-aligned clusters (similarity >= {GAP_THRESHOLD}): {len(aligned_clusters)}")
    lines.append("")
    lines.append("| Cluster | Size | Participants | Similarity | Nearest Subtopic |")
    lines.append("|---------|------|-------------|------------|------------------|")
    for c in sorted(aligned_clusters, key=lambda x: -x["max_similarity"]):
        best = c["best_match_subtopic"]
        sub_info = sub_lookup.get(best, {})
        kw = ", ".join(c["description"]["top_keywords"][:4])
        lines.append(f"| {c['cluster_id']} | {c['n_memories']} | {c['n_participants']} | "
                      f"{c['max_similarity']:.3f} | {best}: {sub_info.get('description', '?')[:50]} |")
    lines.append("")

    # Summary: suggested new themes from gaps
    lines.extend([
        "---",
        "",
        "## Summary: Candidate New Topics from Gap Analysis",
        "",
        "The following gap clusters represent themes with sufficient evidence",
        f"(>= {MIN_PARTICIPANTS} participants) that are semantically distant from",
        "existing subtopics:",
        "",
    ])

    candidate_count = 0
    for c in sorted(gap_clusters, key=lambda x: x["max_similarity"]):
        if c["n_participants"] >= MIN_PARTICIPANTS:
            candidate_count += 1
            lines.append(f"### Candidate {candidate_count}: Cluster {c['cluster_id']}")
            lines.append(f"- **Evidence:** {c['n_memories']} memories, {c['n_participants']} participants")
            lines.append(f"- **Distance from existing framework:** {1 - c['max_similarity']:.3f}")
            lines.append(f"- **Keywords:** {', '.join(c['description']['top_keywords'])}")
            lines.append(f"- **Representative titles:**")
            for t in c["description"]["sample_titles"]:
                lines.append(f"  - {t}")
            lines.append("")

    if candidate_count == 0:
        lines.append("No gap clusters met the minimum participant threshold.")
        lines.append("Consider lowering MIN_PARTICIPANTS or GAP_THRESHOLD.")
        lines.append("")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("TOPIC GAP IDENTIFICATION")
    print("=" * 60)

    # Load data
    print("\n[1/6] Loading existing topics...")
    topics, subtopics_flat = load_existing_topics()
    print(f"  {len(topics)} topics, {len(subtopics_flat)} subtopics")

    print("[2/6] Loading memories and embeddings...")
    memories, embeddings = load_all_memories_with_embeddings()
    print(f"  {len(memories)} memories with embeddings from "
          f"{len(set(m['pid'] for m in memories))} participants")

    # Analysis 1: Link quality
    print("[3/6] Analyzing subtopic link quality...")
    orphaned, weak_fit, cross_cutting = find_poorly_linked_memories(memories)
    print(f"  Orphaned: {len(orphaned)}, Weak-fit: {len(weak_fit)}, "
          f"Cross-cutting: {len(cross_cutting)}")

    # Analysis 2: Clustering
    print(f"[4/6] Clustering memories (K-Means, k={N_CLUSTERS})...")
    labels, centroids, normalized = run_clustering(embeddings, N_CLUSTERS)

    # Compute subtopic reference embeddings from high-importance linked memories
    print("[5/6] Computing subtopic embeddings and measuring gaps...")
    subtopic_embs = compute_subtopic_embeddings(subtopics_flat, memories, embeddings)
    print(f"  Computed embeddings for {len(subtopic_embs)}/{len(subtopics_flat)} subtopics")

    max_sims, best_matches = measure_cluster_gap(centroids, subtopic_embs, normalized)

    # Build cluster analysis
    cluster_analysis = []
    for cid in range(N_CLUSTERS):
        cluster_mask = labels == cid
        cluster_memories = [memories[i] for i in range(len(memories)) if cluster_mask[i]]
        if len(cluster_memories) < MIN_CLUSTER_SIZE:
            continue

        n_participants = len(set(m["pid"] for m in cluster_memories))
        desc = describe_cluster(cluster_memories)

        cluster_analysis.append({
            "cluster_id": cid,
            "n_memories": len(cluster_memories),
            "n_participants": n_participants,
            "max_similarity": float(max_sims[cid]),
            "best_match_subtopic": best_matches[cid],
            "is_gap": max_sims[cid] < GAP_THRESHOLD,
            "description": desc,
        })

    gap_count = sum(1 for c in cluster_analysis if c["is_gap"])
    sims = [c["max_similarity"] for c in cluster_analysis]
    print(f"  {len(cluster_analysis)} clusters above min size, {gap_count} flagged as gaps")
    if sims:
        print(f"  Similarity range: {min(sims):.4f} — {max(sims):.4f} "
              f"(mean={sum(sims)/len(sims):.4f}, threshold={GAP_THRESHOLD})")

    # Generate outputs
    print("[6/6] Generating reports...")

    report = generate_report(orphaned, weak_fit, cross_cutting,
                             cluster_analysis, subtopics_flat)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"  Written: {REPORT_PATH}")

    # JSON output
    json_output = {
        "generated": datetime.now().isoformat(),
        "config": {
            "n_clusters": N_CLUSTERS,
            "min_cluster_size": MIN_CLUSTER_SIZE,
            "min_participants": MIN_PARTICIPANTS,
            "gap_threshold": GAP_THRESHOLD,
        },
        "poorly_linked": {
            "orphaned": [{"pid": m["pid"], "id": m["id"], "title": m["title"],
                          "text": m["text"]} for m in orphaned],
            "weak_fit": [{"pid": item["memory"]["pid"], "id": item["memory"]["id"],
                          "title": item["memory"]["title"],
                          "max_importance": item["max_importance"]}
                         for item in weak_fit],
            "cross_cutting_count": len(cross_cutting),
        },
        "clusters": [{
            "cluster_id": c["cluster_id"],
            "n_memories": c["n_memories"],
            "n_participants": c["n_participants"],
            "max_similarity_to_existing": c["max_similarity"],
            "best_match_subtopic": c["best_match_subtopic"],
            "is_gap": bool(c["is_gap"]),
            "top_keywords": c["description"]["top_keywords"],
            "sample_titles": c["description"]["sample_titles"],
        } for c in cluster_analysis],
    }
    with open(JSON_PATH, "w") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Written: {JSON_PATH}")

    # Console summary
    print("\n" + "=" * 60)
    print("GAP CLUSTERS (candidate new topics/subtopics)")
    print("=" * 60)
    for c in sorted(cluster_analysis, key=lambda x: x["max_similarity"]):
        if not c["is_gap"]:
            continue
        marker = "*" if c["n_participants"] >= MIN_PARTICIPANTS else " "
        print(f"\n{marker} Cluster {c['cluster_id']}: sim={c['max_similarity']:.3f}, "
              f"{c['n_memories']} mem, {c['n_participants']} part")
        print(f"  Keywords: {', '.join(c['description']['top_keywords'])}")
        print(f"  Titles: {c['description']['sample_titles'][:3]}")

    print("\n" + "=" * 60)
    print("Done!")


if __name__ == "__main__":
    main()
