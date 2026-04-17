"""
BERTopic-style Topic Modeling on Memory Embeddings

Uses pre-computed embeddings + HDBSCAN + c-TF-IDF to discover topics without
requiring a fixed k. Produces:
  - Console summary of discovered topics
  - user_study/topic_modeling_results.json  (topic assignments + labels)
  - user_study/topic_modeling_viz.html       (interactive UMAP scatter)
  - user_study/topic_modeling_heatmap.html   (topic × existing-topic heatmap)

Usage:
    .venv-analysis/bin/python scripts/analysis/topic_modeling.py
"""

import json
import os
import warnings
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")

from umap import UMAP
import hdbscan
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

from dotenv import load_dotenv
load_dotenv()
import openai

# ─── Config ──────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
EXISTING_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"

# HDBSCAN: min memories to form a topic cluster
HDBSCAN_MIN_CLUSTER_SIZE = 15
HDBSCAN_MIN_SAMPLES = 5

# UMAP reduction params
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS_CLUSTER = 10   # higher-dim reduction for clustering
UMAP_N_COMPONENTS_VIZ = 2        # 2D for visualization

# Gap threshold: topic is a gap if max similarity to existing subtopics < this
GAP_THRESHOLD = 0.82

# ─── Load ────────────────────────────────────────────────────────────────────

def load_topics():
    with open(EXISTING_TOPICS_PATH) as f:
        topics = json.load(f)
    subtopics = {}
    for i, t in enumerate(topics):
        tid = str(i + 1)
        for j, desc in enumerate(t["subtopics"]):
            subtopics[f"{tid}.{j+1}"] = {
                "topic_id": tid, "topic_name": t["topic"], "desc": desc
            }
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


# ─── Dimensionality reduction ─────────────────────────────────────────────────

def reduce_for_clustering(embeddings):
    """Reduce to 10D for HDBSCAN (better than raw 1536D)."""
    print("  UMAP 1536D → 10D for clustering...", end=" ", flush=True)
    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS_CLUSTER,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.0,        # tighter clusters
        metric="cosine",
        random_state=42,
    )
    reduced = reducer.fit_transform(embeddings)
    print("done")
    return reduced


def reduce_for_viz(embeddings):
    """Reduce to 2D for visualization."""
    print("  UMAP 1536D → 2D for visualization...", end=" ", flush=True)
    reducer = UMAP(
        n_components=UMAP_N_COMPONENTS_VIZ,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords = reducer.fit_transform(embeddings)
    print("done")
    return coords


# ─── HDBSCAN clustering ───────────────────────────────────────────────────────

def cluster_hdbscan(reduced):
    """Density-based clustering — no fixed k, handles noise as topic -1."""
    print(f"  HDBSCAN (min_cluster={HDBSCAN_MIN_CLUSTER_SIZE}, "
          f"min_samples={HDBSCAN_MIN_SAMPLES})...", end=" ", flush=True)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(reduced)
    n_topics = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    print(f"done — {n_topics} topics, {n_noise} noise points")
    return labels


# ─── c-TF-IDF topic labels ────────────────────────────────────────────────────

def compute_ctfidf_labels(memories, labels, top_n=8):
    """
    c-TF-IDF: treats each topic as a single 'document' (all memory text
    concatenated), then finds terms most distinctive to each topic vs. all others.
    Returns a dict: topic_id → list of top keywords.
    """
    topic_docs = defaultdict(list)
    for i, m in enumerate(memories):
        tid = labels[i]
        if tid == -1:
            continue
        text = " ".join([
            m.get("title", ""),
            m.get("text", ""),
            m.get("source_interview_response", "")[:300],
        ])
        topic_docs[tid].append(text)

    topic_ids = sorted(topic_docs.keys())
    corpus = [" ".join(topic_docs[tid]) for tid in topic_ids]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    topic_keywords = {}
    for i, tid in enumerate(topic_ids):
        scores = tfidf_matrix[i].toarray().flatten()
        top_indices = scores.argsort()[::-1][:top_n]
        topic_keywords[tid] = [feature_names[j] for j in top_indices]

    return topic_keywords


def make_topic_label(keywords, max_words=5):
    """Turn keyword list into a short readable label."""
    return " · ".join(keywords[:max_words])


def generate_llm_labels(memories, labels, topic_keywords, max_sims, n_examples=6):
    """
    Call GPT-4o once with all topics to generate short, intuitive human-readable names.
    Returns dict: topic_id → label string.
    """
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    topic_ids = sorted(topic_keywords.keys())

    # Build a compact descriptor for each topic
    topic_blocks = []
    for tid in topic_ids:
        mask = [i for i, l in enumerate(labels) if l == tid]
        # Sample representative memory titles
        sample_titles = [memories[i]["title"] for i in mask[:n_examples]]
        kw_str = ", ".join(topic_keywords[tid][:8])
        is_gap = max_sims.get(tid, 0) < GAP_THRESHOLD
        gap_flag = " [GAP - not covered by existing framework]" if is_gap else ""
        block = (
            f"Topic {tid}{gap_flag}\n"
            f"  Keywords: {kw_str}\n"
            f"  Example memory titles: {'; '.join(sample_titles)}"
        )
        topic_blocks.append(block)

    prompt = (
        "You are analyzing a user study about AI tool adoption at work. "
        "Below are discovered topic clusters from ~2,200 participant memories, "
        "described by c-TF-IDF keywords and example memory titles.\n\n"
        "For each topic, generate a SHORT (3-6 word) human-readable label that "
        "captures the core theme. The label should:\n"
        "- Be natural and intuitive (not just keywords strung together)\n"
        "- Be distinct from the others\n"
        "- Avoid generic phrases like 'AI tools' or 'user experience'\n"
        "- Prefer gerund or noun phrases (e.g., 'Verifying AI Outputs', 'Team Collaboration with AI')\n\n"
        "Return ONLY a JSON object mapping topic number to label, like:\n"
        '{"0": "Verifying AI Outputs", "1": "Daily Software Tools", ...}\n\n'
        "Topics:\n\n" + "\n\n".join(topic_blocks)
    )

    print(f"  Calling GPT-4o to label {len(topic_ids)} topics...", end=" ", flush=True)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    mapping = json.loads(raw)
    # Normalize keys to int
    llm_labels = {int(k): v for k, v in mapping.items()}
    print("done")
    return llm_labels


# ─── Gap detection ────────────────────────────────────────────────────────────

def compute_subtopic_ref_embeddings(subtopics, memories, embeddings):
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


def measure_topic_gaps(topic_ids, memories, embeddings, labels, sub_embs):
    """For each topic, compute max cosine similarity to any existing subtopic."""
    if not sub_embs:
        return {tid: 0.0 for tid in topic_ids}, {tid: "" for tid in topic_ids}

    sub_matrix = np.array(list(sub_embs.values()), dtype=np.float32)
    sub_ids = list(sub_embs.keys())

    # Build topic centroid embeddings from member memories
    topic_centroids = {}
    for tid in topic_ids:
        indices = [i for i, l in enumerate(labels) if l == tid]
        topic_centroids[tid] = embeddings[indices].mean(axis=0)

    centroid_matrix = np.array([topic_centroids[tid] for tid in topic_ids], dtype=np.float32)

    # Normalize
    cn = centroid_matrix / (np.linalg.norm(centroid_matrix, axis=1, keepdims=True) + 1e-8)
    sn = sub_matrix / (np.linalg.norm(sub_matrix, axis=1, keepdims=True) + 1e-8)

    sim = cosine_similarity(cn, sn)
    max_sims = sim.max(axis=1)
    best_ids = [sub_ids[j] for j in sim.argmax(axis=1)]

    return (
        {tid: float(max_sims[i]) for i, tid in enumerate(topic_ids)},
        {tid: best_ids[i] for i, tid in enumerate(topic_ids)},
    )


# ─── Visualization ────────────────────────────────────────────────────────────

def classify_verdicts(memories, embeddings, labels, is_gap_map, min_participants=5):
    """Return dict: topic_id → verdict (CLEAN/CROSS-CUTTING/TOO COARSE/ARTIFACT)."""
    from sklearn.metrics import silhouette_samples
    from collections import defaultdict

    def dominant_topic(m):
        links = m.get("subtopic_links", [])
        if not links: return -1
        best = max(links, key=lambda l: l.get("importance", 0))
        return int(best["subtopic_id"].split(".")[0]) - 1

    assigned_mask = np.array(labels) != -1
    assigned_indices = np.where(assigned_mask)[0]
    # Use raw embeddings projected to lower dim via mean — approximate sil via cosine
    # Compute intra-cluster cosine per topic
    dom = [dominant_topic(m) for m in memories]
    topic_ids = sorted(set(labels) - {-1})
    verdicts = {}
    for cid in topic_ids:
        idxs = np.array([i for i, l in enumerate(labels) if l == cid])
        n_part = len(set(memories[i]["pid"] for i in idxs))
        counts = defaultdict(int)
        for i in idxs:
            d = dom[i]
            if d >= 0: counts[d] += 1
        total_linked = sum(counts.values())
        purity = max(counts.values()) / total_linked if total_linked else 0
        vecs = embeddings[idxs]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vn = vecs / (norms + 1e-8)
        sim_mat = vn @ vn.T
        np.fill_diagonal(sim_mat, np.nan)
        cos = float(np.nanmean(sim_mat))

        if n_part < min_participants:
            verdicts[cid] = "ARTIFACT"
        elif purity >= 0.60:
            verdicts[cid] = "CLEAN"
        elif cos >= 0.55:
            verdicts[cid] = "CROSS-CUTTING"
        else:
            verdicts[cid] = "TOO COARSE"
    return verdicts


def build_scatter(memories, coords, labels, topic_keywords, max_sims, topics,
                  llm_labels=None, verdicts=None):
    topic_ids_set = set(labels) - {-1}
    colors = px.colors.qualitative.Alphabet + px.colors.qualitative.Dark24 + px.colors.qualitative.Set2
    if llm_labels is None:
        llm_labels = {}
    if verdicts is None:
        verdicts = {}

    # marker style per verdict
    STYLES = {
        "CLEAN":          dict(symbol="circle",        size=6,  opacity=0.65, line_width=0,   line_color="white"),
        "CROSS-CUTTING":  dict(symbol="diamond",       size=9,  opacity=0.90, line_width=1.5, line_color="darkorange"),
        "TOO COARSE":     dict(symbol="x",             size=9,  opacity=0.80, line_width=1.5, line_color="crimson"),
        "ARTIFACT":       dict(symbol="circle-open",   size=6,  opacity=0.50, line_width=1,   line_color="gray"),
        "GAP":            dict(symbol="star",          size=10, opacity=0.90, line_width=1.5, line_color="black"),
    }
    PREFIXES = {
        "CLEAN": "", "CROSS-CUTTING": "◆ ", "TOO COARSE": "✗ ", "ARTIFACT": "○ ", "GAP": "⭐ "
    }

    def topic_name(tid):
        return llm_labels.get(tid, make_topic_label(topic_keywords.get(tid, []), max_words=4))

    def hover(i):
        m = memories[i]
        tid = labels[i]
        if tid == -1:
            topic_str = "noise (unassigned)"
        else:
            name = topic_name(tid)
            sim = max_sims.get(tid, 0)
            v = verdicts.get(tid, "")
            verdict_str = f" [{v}]" if v else ""
            gap = " ⚠️ GAP" if sim < GAP_THRESHOLD else ""
            topic_str = f"Topic {tid}: {name} (sim={sim:.2f}{gap}{verdict_str})"
        links = ", ".join(
            f"{l['subtopic_id']}({l['importance']})"
            for l in m.get("subtopic_links", [])[:3]
        )
        return (
            f"<b>{m['title']}</b><br>"
            f"{topic_str}<br>"
            f"Participant: {m['pid'][:10]}...<br>"
            f"Subtopic links: {links or 'none'}<br>"
            f"<i>{m.get('text', '')[:120]}...</i>"
        )

    hovers = [hover(i) for i in range(len(memories))]
    fig = go.Figure()

    # Noise points
    noise_mask = np.where(np.array(labels) == -1)[0]
    if len(noise_mask):
        fig.add_trace(go.Scatter(
            x=coords[noise_mask, 0], y=coords[noise_mask, 1],
            mode="markers",
            marker=dict(size=4, color="lightgray", opacity=0.4),
            name="noise (unassigned)",
            text=[hovers[i] for i in noise_mask],
            hovertemplate="%{text}<extra></extra>",
        ))

    # Topic clusters
    for tid in sorted(topic_ids_set):
        mask = np.where(np.array(labels) == tid)[0]
        name = topic_name(tid)
        sim = max_sims.get(tid, 0)
        is_gap = sim < GAP_THRESHOLD
        v = "GAP" if is_gap else verdicts.get(tid, "CLEAN")
        style = STYLES.get(v, STYLES["CLEAN"])
        prefix = PREFIXES.get(v, "")
        n_part = len(set(memories[i]["pid"] for i in mask))
        legend_label = f"{prefix}T{tid}: {name} [{len(mask)}m,{n_part}p]"
        fig.add_trace(go.Scatter(
            x=coords[mask, 0], y=coords[mask, 1],
            mode="markers",
            marker=dict(
                size=style["size"],
                color=colors[tid % len(colors)],
                symbol=style["symbol"],
                opacity=style["opacity"],
                line=dict(width=style["line_width"], color=style["line_color"]),
            ),
            name=legend_label,
            text=[hovers[i] for i in mask],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=(
                "Memory Topic Modeling (HDBSCAN + c-TF-IDF)<br>"
                "<sup>⭐ star=gap | ◆ diamond=cross-cutting emergent subtopic | "
                "✗ x=too coarse | ● circle=clean | ○ open=artifact (&lt;5p)</sup>"
            ),
            font=dict(size=15),
        ),
        height=800, width=1400,
        template="plotly_white",
        hovermode="closest",
        legend=dict(font=dict(size=9), x=1.01, y=1),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def build_scatter_by_existing_topic(memories, coords, labels, topics):
    """
    Color each dot by the highest-importance existing topic link on that memory.
    Noise points (label=-1) and unlinked memories shown in gray.
    """
    topic_colors = px.colors.qualitative.Bold + px.colors.qualitative.Pastel
    topic_names = [t["topic"] for t in topics]
    n_existing = len(topics)

    # Assign each memory its dominant existing topic (highest importance link)
    def dominant_topic(m):
        links = m.get("subtopic_links", [])
        if not links:
            return -1
        best = max(links, key=lambda l: l.get("importance", 0))
        tid_str = best["subtopic_id"].split(".")[0]
        return int(tid_str) - 1  # 0-indexed

    dom = [dominant_topic(m) for m in memories]

    def hover(i):
        m = memories[i]
        d = dom[i]
        topic_str = topic_names[d] if d >= 0 else "unlinked"
        links = ", ".join(
            f"{l['subtopic_id']}(imp={l['importance']})"
            for l in m.get("subtopic_links", [])[:3]
        )
        cluster = labels[i]
        cluster_str = f"Cluster {cluster}" if cluster != -1 else "noise"
        return (
            f"<b>{m['title']}</b><br>"
            f"Existing topic: {topic_str}<br>"
            f"HDBSCAN cluster: {cluster_str}<br>"
            f"Links: {links or 'none'}<br>"
            f"<i>{m.get('text', '')[:120]}...</i>"
        )

    hovers = [hover(i) for i in range(len(memories))]
    fig = go.Figure()

    # Unlinked / noise
    unlinked = [i for i, d in enumerate(dom) if d < 0]
    if unlinked:
        fig.add_trace(go.Scatter(
            x=coords[unlinked, 0], y=coords[unlinked, 1],
            mode="markers",
            marker=dict(size=4, color="lightgray", opacity=0.35),
            name="unlinked",
            text=[hovers[i] for i in unlinked],
            hovertemplate="%{text}<extra></extra>",
        ))

    # One trace per existing topic
    for tid in range(n_existing):
        idxs = [i for i, d in enumerate(dom) if d == tid]
        if not idxs:
            continue
        color = topic_colors[tid % len(topic_colors)]
        fig.add_trace(go.Scatter(
            x=coords[idxs, 0], y=coords[idxs, 1],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.75,
                        line=dict(width=0.5, color="white")),
            name=f"T{tid+1}: {topic_names[tid][:30]}",
            text=[hovers[i] for i in idxs],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=(
                "Memories Colored by Existing Topic Assignment<br>"
                "<sup>Color = highest-importance subtopic link | gray = unlinked</sup>"
            ),
            font=dict(size=15),
        ),
        height=800, width=1400,
        template="plotly_white",
        hovermode="closest",
        legend=dict(font=dict(size=10), x=1.01, y=1),
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig


def build_heatmap(memories, labels, topic_keywords, max_sims, best_matches,
                  subtopics, topics, llm_labels=None):
    n_existing = len(topics)
    topic_names = [f"T{i+1}: {t['topic'][:22]}" for i, t in enumerate(topics)]
    topic_ids_list = sorted(set(labels) - {-1})
    if llm_labels is None:
        llm_labels = {}

    def topic_name(tid):
        return llm_labels.get(tid, make_topic_label(topic_keywords.get(tid, []), max_words=5))

    # Count: discovered_topic × existing_topic
    counts = np.zeros((len(topic_ids_list), n_existing), dtype=float)
    for i, m in enumerate(memories):
        did = labels[i]
        if did == -1:
            continue
        row = topic_ids_list.index(did)
        for link in m.get("subtopic_links", []):
            eid = int(link["subtopic_id"].split(".")[0]) - 1
            if 0 <= eid < n_existing:
                counts[row, eid] += link["importance"]

    row_totals = counts.sum(axis=1, keepdims=True)
    row_totals[row_totals == 0] = 1
    pct = counts / row_totals * 100

    # Sort: gaps first by sim, then by cluster size
    sizes = [int((np.array(labels) == tid).sum()) for tid in topic_ids_list]
    sims = [max_sims.get(tid, 0) for tid in topic_ids_list]
    is_gap = [s < GAP_THRESHOLD for s in sims]
    sort_key = [s if g else 1 + 1/(sz+1) for s, g, sz in zip(sims, is_gap, sizes)]
    order = sorted(range(len(topic_ids_list)), key=lambda x: sort_key[x])

    pct_sorted = pct[order]
    row_labels, hover_text = [], []
    for rank in order:
        tid = topic_ids_list[rank]
        name = topic_name(tid)
        sz = sizes[rank]
        n_part = len(set(memories[i]["pid"] for i in range(len(memories)) if labels[i] == tid))
        sim = sims[rank]
        gap = "⚠️ " if is_gap[rank] else "   "
        best = best_matches.get(tid, "")
        best_desc = subtopics.get(best, {}).get("desc", "")[:40] if best else ""
        row_labels.append(
            f"{gap}T{tid} [{sz}m,{n_part}p] sim={sim:.2f} — {name}"
        )
        row_hover = []
        for eid in range(n_existing):
            row_hover.append(
                f"Discovered topic T{tid}<br>"
                f"Label: {name}<br>"
                f"Existing topic: {topic_names[eid]}<br>"
                f"Weighted overlap: {pct_sorted[list(order).index(rank), eid]:.1f}%<br>"
                f"Best match: {best} ({best_desc})"
                + (" ⚠️ GAP" if is_gap[rank] else "")
            )
        hover_text.append(row_hover)

    ticktext = [
        f'<span style="color:{"crimson" if is_gap[order[r]] else "black"}">{lbl}</span>'
        for r, lbl in enumerate(row_labels)
    ]

    fig = go.Figure(go.Heatmap(
        z=pct_sorted,
        x=topic_names,
        y=row_labels,
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
        colorscale="Blues",
        zmin=0, zmax=100,
        colorbar=dict(title="Weighted<br>overlap %", thickness=15),
    ))

    for r in range(len(order)):
        if is_gap[order[r]]:
            fig.add_shape(
                type="rect",
                x0=-0.5, x1=n_existing - 0.5,
                y0=r - 0.5, y1=r + 0.5,
                line=dict(color="crimson", width=1),
                fillcolor="rgba(220,50,50,0.06)",
                layer="below",
            )

    fig.update_layout(
        title=dict(
            text=(
                "Discovered Topics → Existing Topic Overlap<br>"
                "<sup>Cell = weighted % of subtopic links to that existing topic  |  "
                "<span style='color:crimson'>⚠️ red = gap topics</span></sup>"
            ),
            font=dict(size=14),
        ),
        height=max(500, 22 * len(topic_ids_list)),
        width=1300,
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(len(order))),
            ticktext=ticktext,
            tickfont=dict(size=9, family="monospace"),
            autorange="reversed",
        ),
        xaxis=dict(tickfont=dict(size=10), tickangle=-35, side="top"),
        margin=dict(l=450, r=80, t=120, b=40),
        template="plotly_white",
    )
    return fig


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("TOPIC MODELING (HDBSCAN + c-TF-IDF)")
    print("=" * 55)

    print("\n[1/6] Loading data...")
    topics, subtopics = load_topics()
    memories, embeddings = load_memories_and_embeddings()
    n_participants = len(set(m["pid"] for m in memories))
    print(f"  {len(memories)} memories, {n_participants} participants")

    print("\n[2/6] Dimensionality reduction...")
    reduced_cluster = reduce_for_clustering(embeddings)
    coords_viz = reduce_for_viz(embeddings)

    print("\n[3/6] HDBSCAN clustering...")
    labels = cluster_hdbscan(reduced_cluster)
    topic_ids = sorted(set(labels) - {-1})

    print("\n[4/6] Computing c-TF-IDF topic labels...")
    topic_keywords = compute_ctfidf_labels(memories, labels)

    print("\n[5/6] Measuring gaps against existing subtopics...")
    sub_embs = compute_subtopic_ref_embeddings(subtopics, memories, embeddings)
    max_sims, best_matches = measure_topic_gaps(
        topic_ids, memories, embeddings, labels, sub_embs
    )

    print("\n[5b] Generating intuitive LLM labels...")
    llm_labels = generate_llm_labels(memories, labels, topic_keywords, max_sims)

    print("\n[5c] Classifying cluster verdicts...")
    verdicts = classify_verdicts(memories, embeddings, labels, {})

    # ── Console summary ──
    print("\n" + "=" * 55)
    print(f"DISCOVERED TOPICS ({len(topic_ids)} topics, "
          f"{(np.array(labels)==-1).sum()} noise)")
    print("=" * 55)
    gap_topics = [tid for tid in topic_ids if max_sims[tid] < GAP_THRESHOLD]
    aligned_topics = [tid for tid in topic_ids if max_sims[tid] >= GAP_THRESHOLD]

    print(f"\n⚠️  GAP topics ({len(gap_topics)}) — not well covered by existing framework:")
    for tid in sorted(gap_topics, key=lambda t: max_sims[t]):
        mask = np.array(labels) == tid
        n_part = len(set(memories[i]["pid"] for i in range(len(memories)) if mask[i]))
        name = llm_labels.get(tid, make_topic_label(topic_keywords.get(tid, [])))
        print(f"  T{tid:2d}: sim={max_sims[tid]:.3f}  {mask.sum():3d}m {n_part:2d}p  {name}")

    print(f"\n✓  Well-aligned topics ({len(aligned_topics)}):")
    for tid in sorted(aligned_topics, key=lambda t: -max_sims[t]):
        mask = np.array(labels) == tid
        n_part = len(set(memories[i]["pid"] for i in range(len(memories)) if mask[i]))
        name = llm_labels.get(tid, make_topic_label(topic_keywords.get(tid, [])))
        best = best_matches.get(tid, "")
        best_desc = subtopics.get(best, {}).get("desc", "")[:45]
        print(f"  T{tid:2d}: sim={max_sims[tid]:.3f}  {mask.sum():3d}m {n_part:2d}p  "
              f"{name}  → {best}: {best_desc}")

    print("\n[6/6] Writing outputs...")

    # JSON results
    results = {
        "n_memories": len(memories),
        "n_participants": n_participants,
        "n_topics": len(topic_ids),
        "n_noise": int((np.array(labels) == -1).sum()),
        "topics": [
            {
                "topic_id": int(tid),
                "label": llm_labels.get(tid, make_topic_label(topic_keywords.get(tid, []))),
                "keywords": topic_keywords.get(tid, []),
                "n_memories": int((np.array(labels) == tid).sum()),
                "n_participants": len(set(
                    memories[i]["pid"] for i in range(len(memories))
                    if labels[i] == tid
                )),
                "max_sim_to_existing": round(max_sims.get(tid, 0), 4),
                "best_match_subtopic": best_matches.get(tid, ""),
                "is_gap": max_sims.get(tid, 0) < GAP_THRESHOLD,
                "verdict": verdicts.get(tid, "CLEAN"),
            }
            for tid in topic_ids
        ],
    }
    json_path = USER_STUDY_DIR / "topic_modeling_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Written: {json_path}")

    scatter_path = USER_STUDY_DIR / "topic_modeling_viz.html"
    fig_scatter = build_scatter(
        memories, coords_viz, labels, topic_keywords, max_sims, topics, llm_labels, verdicts
    )
    fig_scatter.write_html(str(scatter_path), include_plotlyjs="cdn")
    print(f"  Written: {scatter_path}")

    heatmap_path = USER_STUDY_DIR / "topic_modeling_heatmap.html"
    fig_heatmap = build_heatmap(
        memories, labels, topic_keywords, max_sims, best_matches, subtopics, topics, llm_labels
    )
    fig_heatmap.write_html(str(heatmap_path), include_plotlyjs="cdn")
    print(f"  Written: {heatmap_path}")

    existing_topic_path = USER_STUDY_DIR / "topic_modeling_by_existing.html"
    fig_existing = build_scatter_by_existing_topic(memories, coords_viz, labels, topics)
    fig_existing.write_html(str(existing_topic_path), include_plotlyjs="cdn")
    print(f"  Written: {existing_topic_path}")

    print("\nDone! Open visualizations:")
    print(f"  open {scatter_path}")
    print(f"  open {heatmap_path}")
    print(f"  open {existing_topic_path}")


if __name__ == "__main__":
    main()
