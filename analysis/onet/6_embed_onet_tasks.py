"""
Embed O*NET task statements for study participant occupations and visualize
as a 2D UMAP scatter plot colored by occupation.

Usage:
    python analysis/onet/6_embed_onet_tasks.py
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
import plotly.express as px
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).parent.parent.parent
STUDY_TASKS = ROOT / "analysis/onet/data/study_tasks.json"
ONET_TASKS_XLSX = ROOT / "analysis/onet/Task Statements O*NET 30.2.xlsx"
OUT_HTML = ROOT / "analysis/onet/data/onet_task_embedding.html"


def load_study_occupations(path: Path) -> dict[str, str]:
    """Returns {onet_code: onet_title} for all study participants."""
    with open(path) as f:
        data = json.load(f)
    occupations = {}
    for r in data:
        code = r.get("onet_code", "").strip()
        title = r.get("onet_title", "").strip()
        if code and title:
            occupations[code] = title
    return occupations


def load_onet_tasks(xlsx_path: Path, onet_codes: set[str]) -> pd.DataFrame:
    """Load task statements for the given O*NET codes."""
    df = pd.read_excel(xlsx_path)
    # Filter to core tasks for our occupations
    mask = df["O*NET-SOC Code"].isin(onet_codes)
    sub = df[mask][["O*NET-SOC Code", "Title", "Task", "Task Type"]].copy()
    sub.columns = ["onet_code", "onet_title", "task", "task_type"]
    return sub.reset_index(drop=True)


def embed_tasks(tasks: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    print(f"Embedding {len(tasks)} task statements with {model_name}...")
    model = SentenceTransformer(model_name)
    return model.encode(tasks, show_progress_bar=True, batch_size=64)


def reduce_umap(embeddings: np.ndarray, n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=42)
    return reducer.fit_transform(embeddings)


def build_plot(df: pd.DataFrame, xy: np.ndarray) -> go.Figure:
    df = df.copy()
    df["x"] = xy[:, 0]
    df["y"] = xy[:, 1]

    # Assign a consistent color per occupation title
    titles = sorted(df["onet_title"].unique())
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(titles)}

    fig = go.Figure()
    for title in titles:
        sub = df[df["onet_title"] == title]
        fig.add_trace(go.Scatter(
            x=sub["x"],
            y=sub["y"],
            mode="markers",
            name=title,
            marker=dict(color=color_map[title], size=6, opacity=0.75),
            text=sub["task"],
            customdata=np.stack([sub["onet_code"], sub["task_type"]], axis=1),
            hovertemplate=(
                "<b>%{fullData.name}</b><br>"
                "Code: %{customdata[0]}<br>"
                "Type: %{customdata[1]}<br>"
                "<br>%{text}<extra></extra>"
            ),
        ))

    fig.update_layout(
        title=dict(
            text="O*NET Task Statements — Embedding by Occupation",
            font=dict(size=18),
        ),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        legend=dict(
            title="Occupation",
            font=dict(size=11),
            itemsizing="constant",
            tracegroupgap=2,
        ),
        width=1200,
        height=800,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def main():
    occupations = load_study_occupations(STUDY_TASKS)
    print(f"Study occupations: {len(occupations)}")
    for code, title in sorted(occupations.items()):
        print(f"  {code}  {title}")

    df = load_onet_tasks(ONET_TASKS_XLSX, set(occupations.keys()))
    print(f"\nLoaded {len(df)} task statements across {df['onet_code'].nunique()} occupations")

    embeddings = embed_tasks(df["task"].tolist())
    xy = reduce_umap(embeddings)

    fig = build_plot(df, xy)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    print(f"\nSaved → {OUT_HTML}")


if __name__ == "__main__":
    main()
