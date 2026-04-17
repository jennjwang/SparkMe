"""
Cluster Visualization

Reduces all memory embeddings to 2D using UMAP, then produces an interactive
Plotly HTML visualization with:
  - Points colored by K-Means cluster
  - Gap clusters highlighted (those most distant from existing subtopics)
  - A second view colored by existing topic assignment
  - Hover tooltips: memory title, participant, subtopic links, cluster ID

Usage:
    python scripts/visualize_clusters.py

Output:
    user_study/cluster_viz.html
"""

import json
import os
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")
from umap import UMAP
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
EXISTING_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
OUTPUT_PATH = USER_STUDY_DIR / "cluster_viz.html"

N_CLUSTERS = 40
GAP_THRESHOLD = 0.90
UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.1
RANDOM_STATE = 42

# ─── Load data ────────────────────────────────────────────────────────────────

def load_topics():
    with open(EXISTING_TOPICS_PATH) as f:
        topics = json.load(f)
    subtopics = {}
    for i, t in enumerate(topics):
        tid = str(i + 1)
        for j, desc in enumerate(t["subtopics"]):
            subtopics[f"{tid}.{j+1}"] = {"topic": t["topic"], "desc": desc, "topic_id": tid}
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


# ─── Clustering + gap detection ──────────────────────────────────────────────

def run_clustering(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normalized = embeddings / norms
    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(normalized)
    return labels, km.cluster_centers_, normalized


def compute_subtopic_embeddings(subtopics, memories, embeddings):
    sub_embs = {}
    for sid in subtopics:
        indices = [
            i for i, m in enumerate(memories)
            for link in m.get("subtopic_links", [])
            if link["subtopic_id"] == sid and link["importance"] >= 7
        ]
        if indices:
            sub_embs[sid] = embeddings[indices].mean(axis=0)
    return sub_embs


def measure_gaps(centroids, sub_embs):
    if not sub_embs:
        return np.zeros(len(centroids)), [""] * len(centroids)
    sub_matrix = np.array(list(sub_embs.values()), dtype=np.float32)
    sub_ids = list(sub_embs.keys())
    # Normalize
    cn = centroids / (np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-8)
    sn = sub_matrix / (np.linalg.norm(sub_matrix, axis=1, keepdims=True) + 1e-8)
    sim = cosine_similarity(cn, sn)
    best_idx = sim.argmax(axis=1)
    return sim.max(axis=1), [sub_ids[i] for i in best_idx]


# ─── UMAP reduction ──────────────────────────────────────────────────────────

def reduce_umap(embeddings):
    print("  Running UMAP...", end=" ", flush=True)
    reducer = UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric="cosine",
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(embeddings)
    print("done")
    return coords


# ─── Build cluster summaries ─────────────────────────────────────────────────

def top_keywords(cluster_memories, n=6):
    from collections import Counter
    stop = {"user", "user's", "with", "that", "this", "from", "about", "their",
            "they", "have", "been", "more", "also", "does", "into", "work", "role"}
    cnt = Counter()
    for m in cluster_memories:
        for w in m["title"].lower().split():
            if len(w) > 3 and w not in stop:
                cnt[w] += 1
    return [w for w, _ in cnt.most_common(n)]


# ─── Build visualization ─────────────────────────────────────────────────────

def build_figure(memories, coords, labels, max_sims, best_matches, subtopics, topics):
    n = len(memories)

    # Assign primary topic from highest-importance subtopic link
    def primary_topic(m):
        links = m.get("subtopic_links", [])
        if not links:
            return "0"
        best = max(links, key=lambda l: l["importance"])
        return best["subtopic_id"].split(".")[0]

    topic_ids = [primary_topic(m) for m in memories]

    # Cluster keywords
    cluster_kw = {}
    for cid in range(N_CLUSTERS):
        mems = [memories[i] for i in range(n) if labels[i] == cid]
        cluster_kw[cid] = ", ".join(top_keywords(mems))

    # Color palettes
    topic_colors = px.colors.qualitative.D3 + px.colors.qualitative.Set2
    cluster_colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24

    # Build hover text
    def hover(i):
        m = memories[i]
        cid = labels[i]
        sim = max_sims[cid]
        gap_flag = " ⚠️ GAP" if sim < GAP_THRESHOLD else ""
        sub_links = ", ".join(
            f"{l['subtopic_id']}({l['importance']})"
            for l in m.get("subtopic_links", [])[:3]
        )
        return (
            f"<b>{m['title']}</b><br>"
            f"Participant: {m['pid'][:10]}...<br>"
            f"Cluster: {cid} [{cluster_kw.get(cid, '')}]<br>"
            f"Cluster sim to existing: {sim:.3f}{gap_flag}<br>"
            f"Best match: {best_matches[cid]}<br>"
            f"Subtopic links: {sub_links or 'none'}<br>"
            f"<i>{m.get('text', '')[:120]}...</i>"
        )

    hovers = [hover(i) for i in range(n)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Colored by K-Means Cluster (k={N_CLUSTERS}) — ⚠️ = gap cluster",
            "Colored by Primary Existing Topic"
        ),
        horizontal_spacing=0.06,
    )

    # ── Left: colored by cluster, gap clusters get star marker ──
    gap_cids = {cid for cid in range(N_CLUSTERS) if max_sims[cid] < GAP_THRESHOLD}

    for cid in range(N_CLUSTERS):
        mask = np.where(labels == cid)[0]
        if len(mask) == 0:
            continue
        is_gap = cid in gap_cids
        n_part = len(set(memories[i]["pid"] for i in mask))
        kw = cluster_kw.get(cid, "")
        cluster_label = f"C{cid} {kw[:25]}"
        if is_gap:
            cluster_label = f"⚠️ C{cid} {kw[:20]}"

        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            marker=dict(
                size=7 if not is_gap else 10,
                color=cluster_colors[cid % len(cluster_colors)],
                symbol="circle" if not is_gap else "star",
                opacity=0.7 if not is_gap else 1.0,
                line=dict(width=1.5 if is_gap else 0,
                          color="black" if is_gap else "white"),
            ),
            name=cluster_label,
            text=[hovers[i] for i in mask],
            hovertemplate="%{text}<extra></extra>",
            legendgroup=f"cluster_{cid}",
            showlegend=True,
        ), row=1, col=1)

    # ── Right: colored by existing topic ──
    topic_labels = {
        str(i + 1): f"T{i+1}: {t['topic'][:30]}"
        for i, t in enumerate(topics)
    }
    topic_labels["0"] = "T0: Unlinked"

    seen_topics = set()
    for tid in sorted(set(topic_ids)):
        mask = np.where(np.array(topic_ids) == tid)[0]
        if len(mask) == 0:
            continue
        t_idx = int(tid) if tid.isdigit() else 0
        label = topic_labels.get(tid, f"T{tid}")
        fig.add_trace(go.Scatter(
            x=coords[mask, 0],
            y=coords[mask, 1],
            mode="markers",
            marker=dict(
                size=6,
                color=topic_colors[t_idx % len(topic_colors)],
                opacity=0.65,
            ),
            name=label,
            text=[hovers[i] for i in mask],
            hovertemplate="%{text}<extra></extra>",
            legendgroup=f"topic_{tid}",
            showlegend=(tid not in seen_topics),
        ), row=1, col=2)
        seen_topics.add(tid)

    fig.update_layout(
        title=dict(
            text=f"Memory Embedding Space — {n} memories, {len(set(m['pid'] for m in memories))} participants",
            font=dict(size=16),
        ),
        height=800,
        width=1800,
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            font=dict(size=9),
        ),
        hovermode="closest",
        template="plotly_white",
    )

    # Remove axis labels
    for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2"]:
        fig.update_layout(**{ax: dict(showticklabels=False, showgrid=False,
                                      zeroline=False, title="")})

    return fig


# ─── Cluster-to-topic heatmap ────────────────────────────────────────────────

def build_heatmap(memories, labels, max_sims, cluster_kw, topics):
    """
    Heatmap: rows = clusters, columns = existing topics.
    Cell = % of memories in that cluster whose highest-importance subtopic link
    belongs to that topic. Gap clusters are shown with a red row label.
    """
    n_topics = len(topics)
    topic_names = [f"T{i+1}: {t['topic'][:22]}" for i, t in enumerate(topics)]
    topic_names_short = [f"T{i+1}" for i in range(n_topics)]

    # Count: cluster × topic
    counts = np.zeros((N_CLUSTERS, n_topics), dtype=float)
    unlinked = np.zeros(N_CLUSTERS, dtype=float)

    for i, m in enumerate(memories):
        cid = labels[i]
        links = m.get("subtopic_links", [])
        if not links:
            unlinked[cid] += 1
            continue
        best = max(links, key=lambda l: l["importance"])
        tid = int(best["subtopic_id"].split(".")[0]) - 1
        if 0 <= tid < n_topics:
            counts[cid, tid] += 1
        else:
            unlinked[cid] += 1

    # Row totals for normalisation
    row_totals = counts.sum(axis=1) + unlinked
    row_totals[row_totals == 0] = 1
    pct = counts / row_totals[:, None] * 100

    # Sort rows: gap clusters first (by ascending similarity), then by size
    cluster_sizes = np.array([(labels == c).sum() for c in range(N_CLUSTERS)])
    gap_mask = np.array([max_sims[c] < GAP_THRESHOLD for c in range(N_CLUSTERS)])
    sort_key = np.where(gap_mask, max_sims, 1 + (1 / (cluster_sizes + 1)))
    row_order = np.argsort(sort_key)

    pct_sorted = pct[row_order]
    sims_sorted = max_sims[row_order]
    sizes_sorted = cluster_sizes[row_order]
    gap_sorted = gap_mask[row_order]

    # Row labels
    row_labels = []
    for rank, cid in enumerate(row_order):
        kw = cluster_kw.get(cid, "")[:28]
        sim = max_sims[cid]
        size = cluster_sizes[cid]
        n_part = len(set(memories[i]["pid"] for i in range(len(memories)) if labels[i] == cid))
        gap = "⚠️ " if gap_sorted[rank] else "   "
        row_labels.append(f"{gap}C{cid} [{size}m,{n_part}p] sim={sim:.2f} — {kw}")

    # Build hover text matrix
    hover_text = []
    for rank, cid in enumerate(row_order):
        row_hover = []
        for tid in range(n_topics):
            n_mem = int(counts[cid, tid])
            p = pct_sorted[rank, tid]
            row_hover.append(
                f"Cluster C{cid}<br>"
                f"Topic: {topic_names[tid]}<br>"
                f"{n_mem} memories ({p:.1f}%)<br>"
                f"Cluster sim={sims_sorted[rank]:.3f}"
                + (" ⚠️ GAP" if gap_sorted[rank] else "")
            )
        hover_text.append(row_hover)

    # Row color for y-axis labels: red for gaps
    ticktext = [
        f'<span style="color:{"crimson" if gap_sorted[r] else "black"}">{lbl}</span>'
        for r, lbl in enumerate(row_labels)
    ]

    fig = go.Figure(go.Heatmap(
        z=pct_sorted,
        x=topic_names,
        y=row_labels,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale="Blues",
        zmin=0,
        zmax=100,
        colorbar=dict(title="% of cluster<br>memories", thickness=15),
    ))

    # Highlight gap cluster rows with a faint red background via shapes
    n_rows = len(row_order)
    for rank in range(n_rows):
        if gap_sorted[rank]:
            fig.add_shape(
                type="rect",
                x0=-0.5, x1=n_topics - 0.5,
                y0=rank - 0.5, y1=rank + 0.5,
                line=dict(color="crimson", width=1),
                fillcolor="rgba(220,50,50,0.06)",
                layer="below",
            )

    fig.update_layout(
        title=dict(
            text=(
                "Cluster → Existing Topic Mapping<br>"
                "<sup>Cell = % of cluster memories linked to that topic  |  "
                "<span style='color:crimson'>⚠️ red rows = gap clusters</span></sup>"
            ),
            font=dict(size=15),
        ),
        height=max(600, 20 * N_CLUSTERS),
        width=1300,
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=ticktext,
            tickfont=dict(size=9, family="monospace"),
            autorange="reversed",
        ),
        xaxis=dict(
            tickfont=dict(size=10),
            tickangle=-35,
            side="top",
        ),
        margin=dict(l=420, r=80, t=120, b=40),
        template="plotly_white",
    )

    return fig


def add_gap_annotations(fig, coords, labels, max_sims, cluster_kw):
    """Add text labels for gap clusters at their centroid positions."""
    gap_cids = {cid for cid in range(N_CLUSTERS) if max_sims[cid] < GAP_THRESHOLD}
    for cid in gap_cids:
        mask = np.where(labels == cid)[0]
        if len(mask) == 0:
            continue
        cx, cy = coords[mask, 0].mean(), coords[mask, 1].mean()
        kw = cluster_kw.get(cid, "")
        fig.add_annotation(
            x=cx, y=cy,
            text=f"C{cid}<br><i>{kw[:20]}</i>",
            showarrow=False,
            font=dict(size=9, color="black"),
            bgcolor="rgba(255,220,100,0.8)",
            bordercolor="orange",
            borderwidth=1,
            row=1, col=1,
        )
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("CLUSTER VISUALIZATION")
    print("=" * 55)

    print("\n[1/5] Loading data...")
    topics, subtopics = load_topics()
    memories, embeddings = load_memories_and_embeddings()
    print(f"  {len(memories)} memories, {len(set(m['pid'] for m in memories))} participants")

    print("[2/5] Clustering...")
    labels, centroids, normalized = run_clustering(embeddings)

    print("[3/5] Measuring gaps...")
    sub_embs = compute_subtopic_embeddings(subtopics, memories, embeddings)
    max_sims, best_matches = measure_gaps(centroids, sub_embs)
    gap_cids = {cid for cid in range(N_CLUSTERS) if max_sims[cid] < GAP_THRESHOLD}
    print(f"  {len(gap_cids)} gap clusters (sim < {GAP_THRESHOLD})")

    print("[4/5] Running UMAP reduction...")
    coords = reduce_umap(embeddings)

    print("[5/5] Building visualization...")
    cluster_kw = {
        cid: ", ".join(top_keywords([memories[i] for i in range(len(memories)) if labels[i] == cid]))
        for cid in range(N_CLUSTERS)
    }
    fig = build_figure(memories, coords, labels, max_sims, best_matches, subtopics, topics)
    fig = add_gap_annotations(fig, coords, labels, max_sims, cluster_kw)

    fig.write_html(str(OUTPUT_PATH), include_plotlyjs="cdn")
    print(f"  Written: {OUTPUT_PATH}")

    heatmap_path = USER_STUDY_DIR / "cluster_topic_heatmap.html"
    hm = build_heatmap(memories, labels, max_sims, cluster_kw, topics)
    hm.write_html(str(heatmap_path), include_plotlyjs="cdn")
    print(f"  Written: {heatmap_path}")

    # Print gap cluster summary
    print("\nGap clusters (⚠️):")
    for cid in sorted(gap_cids, key=lambda c: max_sims[c]):
        mask = np.where(labels == cid)[0]
        n_part = len(set(memories[i]["pid"] for i in mask))
        print(f"  C{cid}: sim={max_sims[cid]:.3f}, {len(mask)} mem, "
              f"{n_part} part — {cluster_kw[cid]}")

    print("\nDone! Open in browser:")
    print(f"  open {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
