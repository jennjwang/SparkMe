"""
Embed both study-derived task statements and O*NET task statements together,
then visualize in a shared 2D UMAP space.

- Color = occupation (onet_title)
- Shape = source: circle = O*NET, diamond = study interview

Usage:
    python analysis/onet/7_compare_study_vs_onet_embedding.py
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
STUDY_TASKS_JSON = ROOT / "analysis/onet/data/study_tasks.json"
ONET_TASKS_XLSX  = ROOT / "analysis/onet/Task Statements O*NET 30.2.xlsx"
OUT_HTML         = ROOT / "analysis/onet/data/study_vs_onet_embedding.html"


# ── data loading ────────────────────────────────────────────────────────────

def load_study_tasks(path: Path) -> pd.DataFrame:
    with open(path) as f:
        data = json.load(f)
    rows = []
    for r in data:
        title = r.get("onet_title", "").strip()
        code  = r.get("onet_code",  "").strip()
        if not title or not code:
            continue
        for t in r.get("tasks", []):
            stmt = t.get("task_statement", "").strip()
            if stmt:
                rows.append({
                    "source":     "Study (interview)",
                    "onet_code":  code,
                    "onet_title": title,
                    "task":       stmt,
                    "task_type":  "",
                    "user_id":    r["user_id"],
                    "occupation": r.get("occupation", ""),
                })
    return pd.DataFrame(rows)


def load_onet_tasks(xlsx_path: Path, onet_codes: set) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path)
    sub = df[df["O*NET-SOC Code"].isin(onet_codes)].copy()
    sub = sub[["O*NET-SOC Code", "Title", "Task", "Task Type"]]
    sub.columns = ["onet_code", "onet_title", "task", "task_type"]
    sub["source"]     = "O*NET"
    sub["user_id"]    = ""
    sub["occupation"] = ""
    return sub.reset_index(drop=True)


# ── embedding + UMAP ────────────────────────────────────────────────────────

def embed(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    print(f"Embedding {len(texts)} statements with {model_name}...")
    model = SentenceTransformer(model_name)
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def run_umap(embeddings: np.ndarray) -> np.ndarray:
    print("Running UMAP...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42)
    return reducer.fit_transform(embeddings)


# ── plotting ─────────────────────────────────────────────────────────────────

MARKER_SYMBOL = {
    "O*NET":             "circle",
    "Study (interview)": "diamond",
}
MARKER_SIZE = {
    "O*NET":             6,
    "Study (interview)": 10,
}
MARKER_OPACITY = {
    "O*NET":             0.55,
    "Study (interview)": 0.90,
}


def build_figure(df: pd.DataFrame) -> go.Figure:
    titles  = sorted(df["onet_title"].unique())
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(titles)}

    fig = go.Figure()

    # One trace per (occupation, source) pair so legend groups work cleanly
    for source in ["O*NET", "Study (interview)"]:
        for title in titles:
            sub = df[(df["onet_title"] == title) & (df["source"] == source)]
            if sub.empty:
                continue

            # Build hover text
            if source == "Study (interview)":
                customdata = np.stack(
                    [sub["onet_code"], sub["occupation"], sub["source"]], axis=1
                )
                hovertemplate = (
                    "<b>%{fullData.name}</b><br>"
                    "Source: %{customdata[2]}<br>"
                    "Stated occupation: %{customdata[1]}<br>"
                    "Code: %{customdata[0]}<br><br>"
                    "%{text}<extra></extra>"
                )
            else:
                customdata = np.stack(
                    [sub["onet_code"], sub["task_type"], sub["source"]], axis=1
                )
                hovertemplate = (
                    "<b>%{fullData.name}</b><br>"
                    "Source: %{customdata[2]}<br>"
                    "Task type: %{customdata[1]}<br>"
                    "Code: %{customdata[0]}<br><br>"
                    "%{text}<extra></extra>"
                )

            legend_label = title if source == "O*NET" else f"{title} ★"

            fig.add_trace(go.Scatter(
                x=sub["x"],
                y=sub["y"],
                mode="markers",
                name=legend_label,
                legendgroup=title,
                showlegend=True,
                marker=dict(
                    symbol=MARKER_SYMBOL[source],
                    color=color_map[title],
                    size=MARKER_SIZE[source],
                    opacity=MARKER_OPACITY[source],
                    line=dict(
                        width=1 if source == "Study (interview)" else 0,
                        color="black",
                    ),
                ),
                text=sub["task"],
                customdata=customdata,
                hovertemplate=hovertemplate,
            ))

    fig.update_layout(
        title=dict(
            text=(
                "O*NET vs Study-Derived Task Statements — UMAP Embedding<br>"
                "<sup>Color = occupation · Circle = O*NET · Diamond (★) = interview</sup>"
            ),
            font=dict(size=17),
        ),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        legend=dict(
            title="Occupation  (★ = interview)",
            font=dict(size=10),
            itemsizing="constant",
            tracegroupgap=1,
        ),
        width=1300,
        height=850,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    study_df = load_study_tasks(STUDY_TASKS_JSON)
    print(f"Study tasks:  {len(study_df)} statements, {study_df['onet_title'].nunique()} occupations")

    onet_codes = set(study_df["onet_code"].unique())
    onet_df  = load_onet_tasks(ONET_TASKS_XLSX, onet_codes)
    print(f"O*NET tasks:  {len(onet_df)} statements, {onet_df['onet_title'].nunique()} occupations")

    combined = pd.concat([onet_df, study_df], ignore_index=True)

    embeddings = embed(combined["task"].tolist())
    xy = run_umap(embeddings)

    combined["x"] = xy[:, 0]
    combined["y"] = xy[:, 1]

    fig = build_figure(combined)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    print(f"\nSaved → {OUT_HTML}")


if __name__ == "__main__":
    main()
