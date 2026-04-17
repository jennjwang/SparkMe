"""
Embed all tasks and visualize in 2D with UMAP + Plotly.

Usage:
    python analysis/task_clustering/embed_viz.py \
        --input analysis/task_clustering/input_cirs.json \
        --output analysis/task_clustering/output/embed_viz.html

Optionally overlay cluster assignments from a state file:
    python analysis/task_clustering/embed_viz.py \
        --input analysis/task_clustering/input_cirs.json \
        --state  analysis/task_clustering/output/cirs_clusters.json \
        --output analysis/task_clustering/output/embed_viz.html
"""

import argparse
import json
import os
import sys

import numpy as np
import plotly.graph_objects as go
import umap
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBED_MODEL = "text-embedding-3-large"
COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF", "#AEC7E8",
]


def get_embeddings(texts: list[str], client: OpenAI) -> np.ndarray:
    print(f"Embedding {len(texts)} tasks with {EMBED_MODEL}...")
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [r.embedding for r in sorted(response.data, key=lambda x: x.index)]
    return np.array(vecs)


def load_tasks(input_path: str) -> list[dict]:
    with open(input_path) as f:
        data = json.load(f)

    tasks = []
    if isinstance(data, list) and data and "tasks" in data[0]:
        # Nested layout
        for record in data:
            for t in record.get("tasks", []):
                tasks.append({
                    "text": t.get("task_statement") or t.get("text", ""),
                    "source": record.get("user_id", "")[:8],
                    "occupation": record.get("occupation", ""),
                })
    else:
        # Flat layout
        for t in data:
            tasks.append({
                "text": t.get("task_statement") or t.get("text", ""),
                "source": str(t.get("source", t.get("user_id", ""))),
                "occupation": t.get("occupation", ""),
            })
    return tasks


def load_cluster_labels(state_path: str) -> dict[str, str]:
    """Map item_id → cluster label from a saved state JSON."""
    with open(state_path) as f:
        state = json.load(f)

    clusters = state.get("clusters", {})
    items = state.get("items", {})

    item_to_cluster: dict[str, str] = {}
    for cid, cluster in clusters.items():
        label = cluster.get("leader", cid[:8])
        for mid in cluster.get("members", []):
            item_to_cluster[mid] = label
    return item_to_cluster


def run_umap(embeddings: np.ndarray, n_neighbors: int = 5, min_dist: float = 0.3) -> np.ndarray:
    print("Running UMAP...")
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    return reducer.fit_transform(embeddings)


def build_figure(
    tasks: list[dict],
    coords: np.ndarray,
    item_to_cluster: dict[str, str] | None,
) -> go.Figure:
    # Determine color grouping: clusters if available, else occupation, else source
    if item_to_cluster:
        # Build item_id lookup by text (tasks loaded from input may not have ids)
        # We'll use index order as a proxy — both loaded in same order
        groups = []
        for i, task in enumerate(tasks):
            label = item_to_cluster.get(f"item_{i}", None)
            if label is None:
                # Try matching by text across state items
                label = "Unassigned"
            groups.append(label)
        color_key = "cluster"
    elif any(t["occupation"] for t in tasks):
        groups = [t["occupation"] for t in tasks]
        color_key = "occupation"
    else:
        groups = [t["source"] for t in tasks]
        color_key = "source"

    unique_groups = list(dict.fromkeys(groups))
    color_map = {g: COLORS[i % len(COLORS)] for i, g in enumerate(unique_groups)}

    fig = go.Figure()

    for group in unique_groups:
        indices = [i for i, g in enumerate(groups) if g == group]
        fig.add_trace(go.Scatter(
            x=coords[indices, 0],
            y=coords[indices, 1],
            mode="markers",
            name=group,
            marker=dict(size=10, color=color_map[group], opacity=0.85),
            text=[
                f"<b>{tasks[i]['source']}</b><br>{tasks[i]['text']}"
                for i in indices
            ],
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=f"Task Embeddings (UMAP 2D) — colored by {color_key}",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            title=color_key.capitalize(),
            itemsizing="constant",
            font=dict(size=11),
        ),
        width=960,
        height=700,
        margin=dict(l=20, r=20, t=50, b=20),
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="analysis/task_clustering/output/screened_study_tasks.json")
    parser.add_argument("--state",  default=None, help="Optional state JSON to overlay cluster labels")
    parser.add_argument("--output", default="analysis/task_clustering/output/embed_viz.html")
    args = parser.parse_args()

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    tasks = load_tasks(args.input)
    print(f"Loaded {len(tasks)} tasks from {args.input}")

    texts = [t["text"] for t in tasks]
    embeddings = get_embeddings(texts, client)

    coords = run_umap(embeddings, n_neighbors=min(5, len(tasks) - 1))

    item_to_cluster = None
    if args.state:
        # Match state items by text
        item_to_cluster_raw = load_cluster_labels(args.state)
        state_items = json.load(open(args.state)).get("items", {})
        text_to_cluster = {}
        for iid, item in state_items.items():
            label = item_to_cluster_raw.get(iid)
            if label:
                text_to_cluster[item["text"]] = label
        item_to_cluster = {str(i): text_to_cluster.get(t["text"], "Unassigned") for i, t in enumerate(tasks)}

    fig = build_figure(tasks, coords, item_to_cluster)
    fig.write_html(args.output, include_plotlyjs="cdn")
    print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
