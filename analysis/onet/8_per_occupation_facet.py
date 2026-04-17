"""
Per-occupation small-multiples: one panel per occupation showing global UMAP
with all other points in gray background, then O*NET tasks (circles) and
study interview tasks (diamonds) for the focal occupation highlighted.

Usage:
    python analysis/onet/8_per_occupation_facet.py
"""

import json
import re
import sys
from pathlib import Path
import math
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from sentence_transformers import SentenceTransformer

# Matches AI tool mentions used purely as an instrument ("using ChatGPT", etc.)
# Does NOT match when AI is the grammatical object (e.g. "Verify AI-generated answers").
REWRITE_PROMPT = """\
You are editing task statements. For each statement, remove any mention of AI tools \
if the AI tool is only an instrument (e.g. "using AI tools", "using ChatGPT"). \
Keep the action and the real work object intact.

Rules:
- If AI is mentioned as a tool ("using AI tools to X"), remove the tool mention.
- If a non-AI tool is also mentioned ("using AI and Excel"), keep only the non-AI tool.
- If AI IS the actual work object (e.g. "Verify AI-generated answers", \
"Prompt AI tools with questions", "Share AI prompt examples"), leave the statement unchanged.
- Do not change wording beyond removing the AI tool mention.
- Return a JSON array of strings, one rewritten statement per input, in the same order.
- Return ONLY the JSON array. No markdown fences.

Statements:
{statements}
"""


def _rewrite_ai_instruments(statements: list[str], llm) -> list[str]:
    """Use an LLM to strip AI-as-instrument mentions from task statements."""
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
    prompt   = REWRITE_PROMPT.format(statements=numbered)
    response = llm.call(prompt, model="gpt-4.1-mini", temperature=0.0, max_tokens=4096).strip()
    for fence in ("```json", "```"):
        if response.startswith(fence):
            response = response[len(fence):]
    if response.endswith("```"):
        response = response[:-3]
    result = json.loads(response.strip())
    if len(result) != len(statements):
        raise ValueError(f"Expected {len(statements)} results, got {len(result)}")
    return result

ROOT            = Path(__file__).parent.parent.parent
STUDY_TASKS_JSON     = ROOT / "analysis/onet/data/study_tasks.json"
CANONICAL_TASKS_JSON = ROOT / "analysis/onet/data/canonical_tasks.json"
CLUSTER_TRACE_JSON   = ROOT / "analysis/task_clustering/output/onet_clusters_trace.json"
ONET_TASKS_XLSX  = ROOT / "analysis/onet/Task Statements O*NET 30.2.xlsx"
OUT_HTML         = ROOT / "analysis/onet/data/per_occupation_facet.html"
OUT_OVERVIEW     = ROOT / "analysis/onet/data/occupation_overview.html"
OUT_CLUSTER      = ROOT / "analysis/onet/data/cluster_overview.html"
OUT_MAPPING      = ROOT / "analysis/onet/data/occupation_mapping.html"

ONET_COLOR    = "#2166ac"   # blue
STUDY_COLOR   = "#d6604d"   # red-orange
BG_COLOR      = "#cccccc"   # light gray for context
ONET_SYMBOL   = "circle"
STUDY_SYMBOL  = "diamond"
NCOLS         = 5


# ── loaders ──────────────────────────────────────────────────────────────────

def load_cluster_assignments(trace_path: Path) -> dict[str, str]:
    """Read the clustering trace and return {task_text: cluster_label}.
    Uses the last recorded assignment for each task (handles re-assignments after splits)."""
    if not trace_path.exists():
        return {}
    with open(trace_path) as f:
        trace = json.load(f)
    assignments: dict[str, str] = {}
    for event in trace:
        label = event.get("assigned_label", "")
        if label:
            assignments[event["task_text"]] = label
    return assignments


def load_study_tasks(
    study_path: Path,
    canonical_path: Path,
    llm,
    cluster_assignments: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load screened + rewritten canonical tasks, joined with onet codes from study_tasks.
    Uses an LLM to strip AI-as-instrument mentions from task statements.
    If cluster_assignments is provided, adds a cluster_label column."""
    with open(study_path) as f:
        study = json.load(f)
    code_map = {r["onet_title"]: r["onet_code"] for r in study
                if r.get("onet_title") and r.get("onet_code")}

    with open(canonical_path) as f:
        canonical = json.load(f)

    rows = []
    for r in canonical:
        title = r["category"].strip()
        code  = code_map.get(title, "")
        for t in r.get("canonical_tasks", []):
            stmt = t.get("canonical_statement", "").strip()
            if stmt:
                rows.append({
                    "source":     "interview",
                    "onet_code":  code,
                    "onet_title": title,
                    "task":       stmt,
                    "task_type":  "",
                })

    df = pd.DataFrame(rows)

    # Rewrite only the display label — embedding uses the original text so
    # UMAP positions are not affected by the AI-stripping.
    print("Rewriting AI-instrument mentions with LLM (display only)...")
    df["task_display"] = _rewrite_ai_instruments(df["task"].tolist(), llm)

    if cluster_assignments:
        df["cluster_label"] = df["task"].map(cluster_assignments).fillna("Unassigned")
    else:
        df["cluster_label"] = "Unassigned"

    return df


def load_onet_tasks(xlsx_path: Path, onet_codes: set, study_code_to_title: dict) -> pd.DataFrame:
    """Load O*NET task statements for the given codes.
    For codes not found verbatim (e.g. catch-all .00 codes), fall back to any
    sub-codes sharing the same 7-character SOC prefix (e.g. 19-3039)."""
    df = pd.read_excel(xlsx_path)
    available = set(df["O*NET-SOC Code"].dropna().unique())

    rows = []
    for code in onet_codes:
        if code in available:
            match_codes = {code}
        else:
            prefix = code[:7]
            match_codes = {c for c in available if c.startswith(prefix)}
            if match_codes:
                print(f"  {code} not found → using sub-codes: {sorted(match_codes)}")

        for mc in match_codes:
            chunk = df[df["O*NET-SOC Code"] == mc][["O*NET-SOC Code", "Title", "Task", "Task Type"]].copy()
            # Normalise back to the study's code/title so panels align correctly
            chunk.columns = ["onet_code", "onet_title", "task", "task_type"]
            chunk["onet_code"]    = code
            chunk["onet_title"]   = study_code_to_title.get(code, chunk["onet_title"].iloc[0])
            chunk["task_display"] = chunk["task"]
            rows.append(chunk)

    result = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["onet_code", "onet_title", "task", "task_type"]
    )
    result["source"] = "onet"
    result["cluster_label"] = ""
    return result


# ── embed + UMAP ─────────────────────────────────────────────────────────────

def embed(texts: list[str]) -> np.ndarray:
    print(f"Embedding {len(texts)} statements...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=True, batch_size=64)


def run_umap(embs: np.ndarray) -> np.ndarray:
    print("Running UMAP...")
    return umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42
    ).fit_transform(embs)


# ── figure ────────────────────────────────────────────────────────────────────

def wrap(text: str, width: int = 60) -> str:
    """Wrap long task text for hover display."""
    words, lines, line = text.split(), [], []
    for w in words:
        if sum(len(x) + 1 for x in line) + len(w) > width:
            lines.append(" ".join(line))
            line = [w]
        else:
            line.append(w)
    if line:
        lines.append(" ".join(line))
    return "<br>".join(lines)


def short_title(title: str, max_len: int = 38) -> str:
    return title if len(title) <= max_len else title[:max_len - 1] + "…"


def build_figure(df: pd.DataFrame) -> go.Figure:
    titles = sorted(df["onet_title"].unique())
    n      = len(titles)
    ncols  = NCOLS
    nrows  = math.ceil(n / ncols)

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[wrap(t, width=30) for t in titles],
        horizontal_spacing=0.04,
        vertical_spacing=0.10,
    )

    legend_added = {"onet": False, "interview": False, "bg": False}

    for idx, title in enumerate(titles):
        row = idx // ncols + 1
        col = idx % ncols + 1

        is_focal    = df["onet_title"] == title
        bg          = df[df["source"] == "onet"]                        # all O*NET
        focal_onet  = df[is_focal & (df["source"] == "onet")]
        focal_study = df[is_focal & (df["source"] == "interview")]

        # ── background (all O*NET tasks) ─────────────────────────────────
        show_bg_legend = not legend_added["bg"]
        fig.add_trace(go.Scatter(
            x=bg["x"], y=bg["y"],
            mode="markers",
            marker=dict(color=BG_COLOR, size=3, opacity=0.25),
            showlegend=show_bg_legend,
            legendgroup="bg",
            name="All O*NET tasks",
            hoverinfo="skip",
        ), row=row, col=col)
        legend_added["bg"] = True

        # ── focal occupation O*NET tasks (blue) ──────────────────────────
        if not focal_onet.empty:
            show_onet_legend = not legend_added["onet"]
            fig.add_trace(go.Scatter(
                x=focal_onet["x"], y=focal_onet["y"],
                mode="markers",
                marker=dict(
                    symbol=ONET_SYMBOL,
                    color=ONET_COLOR,
                    size=7,
                    opacity=0.85,
                ),
                showlegend=show_onet_legend,
                legendgroup="onet",
                name="O*NET task",
                text=[wrap(t) for t in focal_onet["task_display"]],
                hovertemplate="<b>O*NET</b><br>%{text}<extra></extra>",
            ), row=row, col=col)
            legend_added["onet"] = True

        # ── canonical interview tasks (red diamonds) ──────────────────────
        if not focal_study.empty:
            show_study_legend = not legend_added["interview"]
            hover_texts = [
                f"{wrap(task)}<br><i>Cluster: {cl}</i>"
                for task, cl in zip(focal_study["task_display"], focal_study["cluster_label"])
            ]
            fig.add_trace(go.Scatter(
                x=focal_study["x"], y=focal_study["y"],
                mode="markers",
                marker=dict(
                    symbol=STUDY_SYMBOL,
                    color=STUDY_COLOR,
                    size=10,
                    opacity=0.55,
                    line=dict(width=1, color="black"),
                ),
                showlegend=show_study_legend,
                legendgroup="interview",
                name="Interview task",
                text=hover_texts,
                hovertemplate="<b>Interview</b><br>%{text}<extra></extra>",
            ), row=row, col=col)
            legend_added["interview"] = True

    # hide axes on every subplot
    for i in range(1, nrows * ncols + 1):
        axis_kw = dict(showticklabels=False, showgrid=False,
                       zeroline=False, showline=False)
        fig.update_xaxes(**axis_kw, row=(i - 1) // ncols + 1, col=(i - 1) % ncols + 1)
        fig.update_yaxes(**axis_kw, row=(i - 1) // ncols + 1, col=(i - 1) % ncols + 1)

    # subplot title font size
    for ann in fig.layout.annotations:
        ann.font.size = 10

    fig.update_layout(
        title=dict(
            text=(
                "O*NET vs Interview Task Statements — per Occupation<br>"
                "<sup>Each panel: gray = all O*NET · blue circle = O*NET (focal) · "
                "red diamond = canonical interview</sup>"
            ),
            font=dict(size=16),
            x=0.5,
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.04,
            xanchor="center", x=0.5,
            font=dict(size=12),
            itemsizing="constant",
        ),
        width=1400,
        height=280 * nrows + 80,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ── overview plot ────────────────────────────────────────────────────────────

def build_overview(df: pd.DataFrame) -> go.Figure:
    titles   = sorted(df["onet_title"].unique())
    palette  = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(titles)}

    source_cfg = {
        "onet":      dict(symbol=ONET_SYMBOL,  size=6,  opacity=0.55, line_width=0),
        "interview": dict(symbol=STUDY_SYMBOL, size=10, opacity=0.55, line_width=1),
    }
    source_label = {
        "onet":      "O*NET",
        "interview": "Canonical interview",
    }

    fig = go.Figure()

    # One trace per (source, occupation) — legendgroup = source so clicking
    # a source header in the legend toggles all occupations for that source.
    for source in ["onet", "interview"]:
        cfg   = source_cfg[source]
        label = source_label[source]
        for title in titles:
            sub = df[(df["onet_title"] == title) & (df["source"] == source)]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=title,
                legendgroup=source,
                legendgrouptitle=dict(text=label) if title == titles[0] else {},
                showlegend=True,
                marker=dict(
                    symbol=cfg["symbol"],
                    color=color_map[title],
                    size=cfg["size"],
                    opacity=cfg["opacity"],
                    line=dict(width=cfg["line_width"], color="black"),
                ),
                text=sub["task_display"],
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    f"Source: {label}<br>"
                    "%{text}<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(
            text=(
                "All Task Statements — Colored by Occupation<br>"
                "<sup>Circle = O*NET · Diamond = canonical interview"
                " — click a source header to toggle all its points</sup>"
            ),
            font=dict(size=17),
            x=0.5,
        ),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        legend=dict(
            groupclick="togglegroup",
            font=dict(size=10),
            itemsizing="constant",
            tracegroupgap=8,
        ),
        width=1300,
        height=850,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ── cluster overview ─────────────────────────────────────────────────────────

def build_cluster_overview(study_df: pd.DataFrame) -> go.Figure:
    """UMAP of interview tasks only, colored by cluster label."""
    palette = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    labels = sorted(study_df["cluster_label"].unique())
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(labels)}

    fig = go.Figure()

    for label in labels:
        sub = study_df[study_df["cluster_label"] == label]
        fig.add_trace(go.Scatter(
            x=sub["x"], y=sub["y"],
            mode="markers",
            name=label,
            marker=dict(
                symbol=STUDY_SYMBOL,
                color=color_map[label],
                size=10,
                opacity=0.8,
                line=dict(width=1, color="black"),
            ),
            text=[
                f"{wrap(task)}<br><i>{occ}</i>"
                for task, occ in zip(sub["task_display"], sub["onet_title"])
            ],
            hovertemplate="<b>%{fullData.name}</b><br>%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text=(
                "Interview Tasks — Colored by Cluster<br>"
                "<sup>Diamond = canonical interview task — hover for task text and occupation</sup>"
            ),
            font=dict(size=17),
            x=0.5,
        ),
        xaxis=dict(title="UMAP-1", showgrid=False, zeroline=False),
        yaxis=dict(title="UMAP-2", showgrid=False, zeroline=False),
        legend=dict(
            font=dict(size=10),
            itemsizing="constant",
            title="Cluster",
        ),
        width=1300,
        height=850,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


# ── occupation mapping table ──────────────────────────────────────────────────

def build_mapping_table(study_path: Path) -> go.Figure:
    """Interactive table showing study occupation → O*NET mapping for all participants."""
    with open(study_path) as f:
        data = json.load(f)

    df = pd.DataFrame([
        {
            "occupation":    r.get("occupation", ""),
            "category":      r.get("category", ""),
            "onet_industry": r.get("onet_industry", ""),
            "onet_code":     r.get("onet_code", ""),
            "onet_title":    r.get("onet_title", ""),
            "match_notes":   r.get("onet_match_notes", ""),
        }
        for r in data
    ]).sort_values(["category", "occupation"]).reset_index(drop=True)

    row_colors = ["white" if i % 2 == 0 else "#f0f4f8" for i in range(len(df))]

    fig = go.Figure(go.Table(
        columnwidth=[180, 130, 160, 100, 200, 300],
        header=dict(
            values=[
                "<b>Study Occupation</b>", "<b>Category</b>", "<b>Industry</b>",
                "<b>O*NET Code</b>", "<b>O*NET Title</b>", "<b>Match Notes</b>",
            ],
            fill_color="#2166ac",
            font=dict(color="white", size=12),
            align="left",
            height=34,
        ),
        cells=dict(
            values=[
                df["occupation"], df["category"], df["onet_industry"],
                df["onet_code"], df["onet_title"], df["match_notes"],
            ],
            fill_color=[row_colors] * 6,
            align="left",
            font=dict(size=11),
            height=26,
        ),
    ))

    fig.update_layout(
        title=dict(
            text="Study Occupation → O*NET Mapping",
            font=dict(size=17),
            x=0.5,
        ),
        width=1400,
        height=100 + 28 * len(df),
        margin=dict(t=60, b=20, l=20, r=20),
    )
    return fig


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    sys.path.insert(0, str(ROOT / "dataset_gen"))
    from llm_client import LLMClient
    llm = LLMClient()

    cluster_assignments = load_cluster_assignments(CLUSTER_TRACE_JSON)
    print(f"Cluster assignments: {len(cluster_assignments)} tasks mapped to clusters")

    study_df = load_study_tasks(STUDY_TASKS_JSON, CANONICAL_TASKS_JSON, llm,
                                cluster_assignments=cluster_assignments)
    print(f"Study tasks : {len(study_df)} across {study_df['onet_title'].nunique()} occupations")

    onet_codes        = set(study_df["onet_code"].unique())
    code_to_title     = dict(zip(study_df["onet_code"], study_df["onet_title"]))
    onet_df           = load_onet_tasks(ONET_TASKS_XLSX, onet_codes, code_to_title)
    print(f"O*NET tasks : {len(onet_df)} across {onet_df['onet_title'].nunique()} occupations")

    combined = pd.concat([onet_df, study_df], ignore_index=True)

    embs = embed(combined["task"].tolist())
    xy   = run_umap(embs)

    combined["x"] = xy[:, 0]
    combined["y"] = xy[:, 1]

    fig = build_figure(combined)
    fig.write_html(str(OUT_HTML), include_plotlyjs="cdn")
    print(f"Saved → {OUT_HTML}")

    overview = build_overview(combined)
    overview.write_html(str(OUT_OVERVIEW), include_plotlyjs="cdn")
    print(f"Saved → {OUT_OVERVIEW}")

    cluster_fig = build_cluster_overview(combined[combined["source"] == "interview"].copy())
    cluster_fig.write_html(str(OUT_CLUSTER), include_plotlyjs="cdn")
    print(f"Saved → {OUT_CLUSTER}")

    mapping_fig = build_mapping_table(STUDY_TASKS_JSON)
    mapping_fig.write_html(str(OUT_MAPPING), include_plotlyjs="cdn")
    print(f"Saved → {OUT_MAPPING}")


if __name__ == "__main__":
    main()
