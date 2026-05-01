"""
Visualize task generator output vs. participant tasks in 2D embedding space.

Each point is one task. Color/shape encodes origin and participant response:
  - Generated & confirmed  (●, green)
  - Generated & removed    (●, red)
  - Generated & unreviewed (●, gray)
  - Generated & edited     (●, orange)
  - Participant-added      (★, purple)  — extra tasks from brain-dump

Points are labeled with short task names on hover (interactive) or annotated
in the static PNG output.

Usage:
    # single user — saves plot to evaluations/task_diversity_plot.png
    python evaluation/plot_task_diversity.py --base-path logs/ --user-id 1241161077

    # multiple users — one subplot per user (up to --max-users)
    python evaluation/plot_task_diversity.py --base-path logs/ \
        --sample-users-path analysis/sample_users_50.json --max-users 6

    # interactive HTML (requires plotly)
    python evaluation/plot_task_diversity.py --base-path logs/ --user-id 1241161077 --interactive
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI

OPENAI_CLIENT = OpenAI()
EMBED_MODEL = "text-embedding-3-small"


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 unit-normalised embeddings."""
    BATCH = 100
    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = OPENAI_CLIENT.embeddings.create(input=batch, model=EMBED_MODEL)
        vecs.append(np.array([r.embedding for r in resp.data], dtype=np.float32))
    mat = np.vstack(vecs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def reduce_2d(embeddings: np.ndarray, method: str = "umap") -> np.ndarray:
    """Reduce to 2D. Uses UMAP if available, falls back to PCA."""
    if method == "umap":
        try:
            import umap as umap_module
            reducer = umap_module.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(embeddings) - 1))
            return reducer.fit_transform(embeddings)
        except Exception as e:
            print(f"UMAP failed ({e}), falling back to PCA")
    from sklearn.decomposition import PCA
    return PCA(n_components=2).fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

STATUS_ORDER = ["confirmed", "edited", "unreviewed", "removed"]

STYLE = {
    "confirmed":  {"color": "#2ecc71", "marker": "o", "label": "Generated — confirmed",  "zorder": 4},
    "edited":     {"color": "#f39c12", "marker": "o", "label": "Generated — edited",     "zorder": 4},
    "unreviewed": {"color": "#95a5a6", "marker": "o", "label": "Generated — unreviewed", "zorder": 2},
    "removed":    {"color": "#e74c3c", "marker": "o", "label": "Generated — removed",    "zorder": 3},
    "participant":{"color": "#8e44ad", "marker": "*", "label": "Participant-added",       "zorder": 5},
}


def load_user_tasks(user_id: str, base_path: str) -> Optional[list[dict]]:
    """Return list of {text, status} dicts, or None if data missing."""
    widget_path = Path(base_path) / user_id / "evaluations" / "task_widget_data.json"
    if not widget_path.exists():
        return None
    try:
        data = json.loads(widget_path.read_text())
    except Exception:
        return None

    tasks = []
    seen = set()

    for t in (data.get("listed") or []):
        text = (t.get("text") or t.get("name") or "").strip() if isinstance(t, dict) else str(t).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        status = (t.get("status") or "unreviewed").strip() if isinstance(t, dict) else "unreviewed"
        if status not in STYLE:
            status = "unreviewed"
        tasks.append({"text": text, "status": status})

    # Extra tasks added by participant during brain-dump
    for t in (data.get("extra_tasks") or []):
        text = (t.get("name") or t.get("text") or str(t)).strip() if isinstance(t, dict) else str(t).strip()
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        tasks.append({"text": text, "status": "participant"})

    return tasks if tasks else None


# ---------------------------------------------------------------------------
# Single-user plot
# ---------------------------------------------------------------------------

def plot_user(ax, tasks: list[dict], user_id: str, title: str = None, annotate: bool = True):
    import matplotlib.pyplot as plt

    texts = [t["text"] for t in tasks]
    statuses = [t["status"] for t in tasks]

    embeddings = embed_texts(texts)
    coords = reduce_2d(embeddings)

    # Group by status for legend de-duplication
    plotted_labels = set()
    for i, (x, y) in enumerate(coords):
        st = statuses[i]
        s = STYLE[st]
        label = s["label"] if s["label"] not in plotted_labels else None
        if label:
            plotted_labels.add(s["label"])
        size = 180 if st == "participant" else 80
        ax.scatter(x, y, c=s["color"], marker=s["marker"], s=size,
                   zorder=s["zorder"], label=label,
                   edgecolors="white" if st != "participant" else s["color"],
                   linewidths=0.5)

    if annotate:
        for i, (x, y) in enumerate(coords):
            short = texts[i][:32] + ("…" if len(texts[i]) > 32 else "")
            ax.annotate(short, (x, y), fontsize=5.5, alpha=0.75,
                        xytext=(4, 4), textcoords="offset points")

    ax.set_title(title or f"User {user_id}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return coords


def plot_single(user_id: str, base_path: str, out_path: Path, annotate: bool = True):
    import matplotlib.pyplot as plt

    tasks = load_user_tasks(user_id, base_path)
    if not tasks:
        print(f"No task data for user {user_id}")
        sys.exit(1)

    print(f"Plotting {len(tasks)} tasks for user {user_id}…")
    fig, ax = plt.subplots(figsize=(10, 7))
    plot_user(ax, tasks, user_id, annotate=annotate)
    ax.legend(loc="best", fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Multi-user grid plot
# ---------------------------------------------------------------------------

def plot_grid(user_ids: list[str], base_path: str, out_path: Path, annotate: bool = False):
    import matplotlib.pyplot as plt

    n = len(user_ids)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    legend_handles = {}

    for i, uid in enumerate(user_ids):
        tasks = load_user_tasks(uid, base_path)
        ax = axes[i]
        if not tasks:
            ax.set_visible(False)
            continue
        print(f"  [{i+1}/{n}] user {uid}: {len(tasks)} tasks")
        plot_user(ax, tasks, uid, annotate=annotate)
        for patch in ax.get_legend_handles_labels()[0]:
            label = patch.get_label()
            if label and label not in legend_handles:
                legend_handles[label] = patch

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    if legend_handles:
        fig.legend(legend_handles.values(), legend_handles.keys(),
                   loc="lower center", ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, 0.01), framealpha=0.9)

    fig.suptitle("Task Generator Output — UMAP Projection", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Interactive (plotly)
# ---------------------------------------------------------------------------

def plot_interactive(user_id: str, base_path: str, out_path: Path):
    try:
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        sys.exit(1)

    tasks = load_user_tasks(user_id, base_path)
    if not tasks:
        print(f"No task data for user {user_id}")
        sys.exit(1)

    texts = [t["text"] for t in tasks]
    statuses = [t["status"] for t in tasks]
    embeddings = embed_texts(texts)
    coords = reduce_2d(embeddings)

    traces = []
    for st in STATUS_ORDER + ["participant"]:
        s = STYLE[st]
        idx = [i for i, v in enumerate(statuses) if v == st]
        if not idx:
            continue
        xs = [coords[i, 0] for i in idx]
        ys = [coords[i, 1] for i in idx]
        labels = [texts[i] for i in idx]
        symbol = "star" if st == "participant" else "circle"
        traces.append(go.Scatter(
            x=xs, y=ys, mode="markers+text",
            text=[t[:40] + ("…" if len(t) > 40 else "") for t in labels],
            textposition="top center", textfont=dict(size=9),
            hovertext=labels, hoverinfo="text",
            marker=dict(color=s["color"], size=12 if st == "participant" else 8,
                        symbol=symbol, line=dict(width=0.5, color="white")),
            name=s["label"]
        ))

    fig = go.Figure(traces)
    fig.update_layout(
        title=f"Task diversity — user {user_id}",
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15),
        width=1000, height=700,
    )
    fig.write_html(str(out_path))
    print(f"Saved interactive plot → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot task diversity in 2D embedding space")
    parser.add_argument("--base-path", required=True, help="Logs directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user-id", help="Single user ID")
    group.add_argument("--sample-users-path", help="JSON list of users for grid plot")
    parser.add_argument("--max-users", type=int, default=6, help="Max users in grid (default 6)")
    parser.add_argument("--out", help="Output file path (PNG or HTML)")
    parser.add_argument("--interactive", action="store_true", help="Produce interactive HTML via plotly")
    parser.add_argument("--no-annotate", action="store_true", help="Skip task label annotations")
    args = parser.parse_args()

    annotate = not args.no_annotate

    if args.user_id:
        out = Path(args.out) if args.out else Path(args.base_path) / args.user_id / "evaluations" / (
            "task_diversity_plot.html" if args.interactive else "task_diversity_plot.png"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        if args.interactive:
            plot_interactive(args.user_id, args.base_path, out)
        else:
            plot_single(args.user_id, args.base_path, out, annotate=annotate)
    else:
        with open(args.sample_users_path) as f:
            sample = json.load(f)
        user_ids = [u["User ID"] for u in sample][: args.max_users]
        out = Path(args.out) if args.out else Path(args.base_path) / "task_diversity_grid.png"
        print(f"Plotting grid for {len(user_ids)} users…")
        plot_grid(user_ids, args.base_path, out, annotate=annotate)


if __name__ == "__main__":
    main()
