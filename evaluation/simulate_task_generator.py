"""
Simulate and compare the two task generator variants side-by-side,
drawn directly from the real implementation on each branch.

Variants:
  simple      — surface tasks the user explicitly mentioned; fill rest from role
                (simple branch: mentioned_block says "MUST include all of these")
  post_hoc    — anchor on the user's brain dump but prioritize GAPS —
                areas they did NOT mention (interactive branch: typical_week_block)

Both variants use the exact prompts from each branch, run against the same
user's job_description and first chat message (used as brain_dump proxy).

All tasks are embedded jointly and projected into a shared 2D UMAP so
positions are directly comparable across variants.

Usage:
    python evaluation/simulate_task_generator.py \
        --base-path logs/ \
        --user-ids 8071017214 1241161077

    python evaluation/simulate_task_generator.py \
        --base-path logs/ \
        --sample-users-path analysis/sample_users_50.json --n-users 2
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

# ── src on path ──────────────────────────────────────────────────────────────
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.utils.llm.engines import get_engine, invoke_engine

OPENAI_CLIENT = OpenAI()
EMBED_MODEL   = "text-embedding-3-small"
TASK_GEN_MODEL = os.getenv("TASK_GEN_MODEL", "claude-sonnet-4-6")
TASK_BATCH_SIZE = 3
CORE_BATCHES    = 2   # matches both branches

VARIANT_STYLE = {
    "scratch":    {"color": "#95a5a6", "label": "Scratch (role only)"},
    "simple":     {"color": "#3498db", "label": "Simple (explicit categories)"},
    "post_hoc":   {"color": "#e67e22", "label": "Post-hoc (brain-dump gap-fill)"},
    "combined":   {"color": "#8e44ad", "label": "Combined (categories + brain dump)"},
}

# Pulled directly from _SESSION_PERSPECTIVES in main_flask.py
SESSION_PERSPECTIVES = [
    "Emphasize tasks that are recurring and form the backbone of the daily or weekly routine.",
    "Emphasize tasks that involve coordinating with, communicating with, or depending on other people.",
    "Emphasize tasks that are less frequent but carry high stakes or require significant effort.",
    "Emphasize tasks that involve documentation, tracking, or managing information.",
    "Emphasize tasks that involve external parties such as clients, partners, vendors, or the public.",
    "Emphasize tasks that are often invisible or taken for granted but are essential to the role.",
    "Emphasize tasks that require specialized judgment, expertise, or domain knowledge.",
    "Emphasize tasks related to planning, prioritizing, and managing workloads or timelines.",
    "Emphasize tasks that involve using specific tools, systems, or technology.",
    "Emphasize tasks that have direct impact on outcomes, quality, or the work of others.",
    "Emphasize tasks that involve learning, staying current, or improving skills and knowledge.",
    "Emphasize tasks that come up reactively — in response to problems, requests, or unexpected events.",
]


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_user_data(user_id: str, base_path: str) -> dict | None:
    portrait_path = Path(base_path) / user_id / "user_portrait.json"
    if not portrait_path.exists():
        return None
    portrait = json.loads(portrait_path.read_text())
    job_description = portrait.get("Role") or ""

    # Mentioned tasks from portrait Task Inventory (mirrors _extract_user_mentioned_tasks)
    mentioned_tasks = [str(t).strip() for t in (portrait.get("Task Inventory") or []) if str(t).strip()]

    # Brain dump: first user message from chat log (proxy for typical-week answer)
    brain_dump = ""
    chat_log = Path(base_path) / user_id / "execution_logs" / "session_0" / "chat_history.log"
    if chat_log.exists():
        for line in chat_log.read_text().splitlines():
            m = re.search(r" - INFO - User: (.+)", line)
            if m:
                brain_dump = m.group(1).strip()
                break
    if not brain_dump and mentioned_tasks:
        brain_dump = "I typically: " + "; ".join(mentioned_tasks[:6])

    return {
        "user_id":         user_id,
        "job_description": job_description,
        "mentioned_tasks": mentioned_tasks,
        "brain_dump":      brain_dump,
        "role_short":      job_description[:60],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prior_block(prior_tasks: list[str]) -> str:
    if not prior_tasks:
        return ""
    return (
        "\nTasks already shown (STRICT: do NOT repeat or rephrase any of these — "
        "treat semantically similar tasks as duplicates):\n"
        + "\n".join(f"- {t}" for t in prior_tasks) + "\n"
    )

def _coverage_hint(batch_index: int) -> str:
    if batch_index < CORE_BATCHES:
        rank = batch_index + 1
        return (
            f"Return the {TASK_BATCH_SIZE} {'most' if batch_index == 0 else 'next most'} important "
            f"core tasks for this role — the work this person spends the most time on. "
            f"Rank strictly by centrality and frequency (batch {rank} of {CORE_BATCHES} core batches)."
        )
    return (
        f"The first {CORE_BATCHES * TASK_BATCH_SIZE} tasks covered the core of this role. "
        f"Now return {TASK_BATCH_SIZE} tasks from a DIFFERENT area not yet represented — "
        "vary the type of work: coordination, documentation, external-facing, reactive, specialized, etc. "
        "Think about what a complete picture of this occupation looks like across all dimensions."
    )

_RULES = (
    "Rules:\n"
    "- Focus on tasks typical of the general occupation/role category, not details specific to this individual\n"
    "- Tasks must be DISTINCT — no two tasks should describe the same activity even with different wording\n"
    "- Each task: 5–12 words, starts with an action verb, sentence case\n"
    "- Be specific — include object and context where helpful\n"
    "- Each task needs a name (5–12 words) and a description (1 sentence).\n"
    "- Return ONLY valid JSON (no markdown fences):\n"
    '  {"tasks": [{"name": "Task name here", "description": "One sentence."}, ...], "has_more": true}'
)


def _clean_task_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    name = value.strip()
    lower = name.lower()
    if not name:
        return None
    if name.startswith(("{", "}", "[", "]", ":", ",")):
        return None
    if "}, {" in name or "has_more" in lower:
        return None
    if lower in {"true", "false", "null"}:
        return None
    if not re.search(r"[A-Za-z]", name):
        return None
    n_words = len(name.split())
    if n_words < 3 or n_words > 16:
        return None
    return name


# ─────────────────────────────────────────────────────────────────────────────
# Variant prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_prompt_scratch(job_description: str, prior_tasks: list[str], batch_index: int) -> str:
    """No user context — pure role-based generation."""
    return (
        f"Occupation context: \"{job_description}\"\n"
        f"{_prior_block(prior_tasks)}\n"
        f"{_coverage_hint(batch_index)}\n\n"
        f"{_RULES}"
    )


def _build_prompt_simple(
    job_description: str, prior_tasks: list[str], batch_index: int, session_perspective: str
) -> str:
    """Explicit category guidance via session perspective (simple branch)."""
    perspective_line = f"Session focus: {session_perspective}\n" if session_perspective else ""
    return (
        f"Occupation context: \"{job_description}\"\n"
        f"{perspective_line}"
        f"{_prior_block(prior_tasks)}\n"
        f"{_coverage_hint(batch_index)}\n\n"
        f"{_RULES}"
    )


def _build_prompt_post_hoc(
    job_description: str, prior_tasks: list[str], batch_index: int, brain_dump: str
) -> str:
    """Brain-dump anchor — prioritize gaps in what the user described (interactive branch)."""
    typical_week_block = (
        f"\nThis participant described their typical work week as follows:\n"
        f"\"{brain_dump}\"\n\n"
        f"Use this as your primary anchor. Generate tasks that:\n"
        f"1. Prioritize areas of work this person did NOT mention — probe the gaps in their self-description.\n"
        f"2. Fill gaps where their description was high-level (e.g. 'meetings' → break into "
        f"specific meeting types; 'admin work' → specific admin tasks).\n"
        f"3. Maximize breadth of coverage across their full work profile, not just the most salient items.\n"
    ) if brain_dump else ""
    return (
        f"Occupation context: \"{job_description}\"\n"
        f"{typical_week_block}"
        f"{_prior_block(prior_tasks)}\n"
        f"{_coverage_hint(batch_index)}\n\n"
        f"{_RULES}"
    )


def _build_prompt_combined(
    job_description: str, prior_tasks: list[str], batch_index: int,
    brain_dump: str, session_perspective: str,
) -> str:
    """Both signals: brain-dump gap-fill anchor + explicit category emphasis."""
    typical_week_block = (
        f"\nThis participant described their typical work week as follows:\n"
        f"\"{brain_dump}\"\n\n"
        f"Use this as your primary anchor. Generate tasks that:\n"
        f"1. Prioritize areas of work this person did NOT mention — probe the gaps in their self-description.\n"
        f"2. Fill gaps where their description was high-level (e.g. 'meetings' → break into "
        f"specific meeting types; 'admin work' → specific admin tasks).\n"
        f"3. Maximize breadth of coverage across their full work profile, not just the most salient items.\n"
    ) if brain_dump else ""
    perspective_line = f"Session focus: {session_perspective}\n" if session_perspective else ""
    return (
        f"Occupation context: \"{job_description}\"\n"
        f"{perspective_line}"
        f"{typical_week_block}"
        f"{_prior_block(prior_tasks)}\n"
        f"{_coverage_hint(batch_index)}\n\n"
        f"{_RULES}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# LLM call + parse
# ─────────────────────────────────────────────────────────────────────────────

def _call_and_parse(prompt: str) -> tuple[list[str], bool]:
    engine = get_engine(model_name=TASK_GEN_MODEL, temperature=0.9)
    response = invoke_engine(engine, prompt)
    raw = (response.content if hasattr(response, "content") else str(response)).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            tasks_raw = parsed.get("tasks") or []
            has_more = bool(parsed.get("has_more", True))
        elif isinstance(parsed, list):
            tasks_raw = parsed
            has_more = True
        else:
            tasks_raw = []
            has_more = True
        names = []
        for t in tasks_raw:
            if isinstance(t, dict):
                name = _clean_task_name(t.get("name"))
            else:
                name = _clean_task_name(t)
            if name:
                names.append(name)
        return names, has_more
    except Exception:
        names = []
        for match in re.finditer(r'"name"\s*:\s*"((?:\\.|[^"\\])*)"', raw):
            try:
                candidate = json.loads(f'"{match.group(1)}"')
            except Exception:
                candidate = match.group(1)
            name = _clean_task_name(candidate)
            if name:
                names.append(name)
        return names, True


def generate_variant(variant: str, user: dict, target: int) -> list[str]:
    jd = user["job_description"]
    bd = user["brain_dump"]
    tasks: list[str] = []
    b = 0

    while len(tasks) < target:
        prior = tasks[:]
        perspective = SESSION_PERSPECTIVES[b % len(SESSION_PERSPECTIVES)]

        if variant == "scratch":
            prompt = _build_prompt_scratch(jd, prior, b)
        elif variant == "simple":
            prompt = _build_prompt_simple(jd, prior, b, perspective)
        elif variant == "post_hoc":
            prompt = _build_prompt_post_hoc(jd, prior, b, bd)
        elif variant == "combined":
            prompt = _build_prompt_combined(jd, prior, b, bd, perspective)
        else:
            raise ValueError(f"Unknown variant: {variant}")

        new_tasks, _ = _call_and_parse(prompt)
        tasks.extend(new_tasks)
        b += 1

        # Safety cap to avoid infinite loops
        if b >= target:
            break

    return tasks[:target]


# ─────────────────────────────────────────────────────────────────────────────
# Embedding + projection
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> np.ndarray:
    resp = OPENAI_CLIENT.embeddings.create(input=texts, model=EMBED_MODEL)
    mat  = np.array([r.embedding for r in resp.data], dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def reduce_2d(embeddings: np.ndarray) -> np.ndarray:
    try:
        import umap as umap_module
        n = len(embeddings)
        reducer = umap_module.UMAP(
            n_components=2, random_state=42,
            n_neighbors=min(10, n - 1), min_dist=0.25,
        )
        return reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"  UMAP failed ({e}), using PCA")
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(embeddings)


def diversity_metrics(embeddings: np.ndarray) -> dict:
    n = len(embeddings)
    if n < 2:
        return {"mean_pairwise": 0.0, "min_pairwise": 0.0, "centroid_spread": 0.0}
    sim  = embeddings @ embeddings.T
    dist = np.clip(1.0 - sim, 0, 2)
    mask = ~np.eye(n, dtype=bool)
    pw   = dist[mask]
    c    = embeddings.mean(0); c /= (np.linalg.norm(c) or 1.0)
    return {
        "mean_pairwise":  round(float(pw.mean()), 4),
        "min_pairwise":   round(float(pw.min()),  4),
        "centroid_spread": round(float(np.clip(1.0 - (embeddings @ c), 0, 2).mean()), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_user(user: dict, results: dict[str, list[str]], out_path: Path):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    all_texts  = []
    all_labels = []
    for variant, tasks in results.items():
        all_texts.extend(tasks)
        all_labels.extend([variant] * len(tasks))

    if not all_texts:
        print("  Nothing to plot.")
        return

    # Include user's own mentioned tasks in the joint embedding
    user_tasks = user.get("mentioned_tasks", [])
    all_texts_with_user = all_texts + user_tasks

    print(f"  Embedding {len(all_texts_with_user)} tasks jointly ({len(user_tasks)} user-mentioned)…")
    embeddings_all = embed_texts(all_texts_with_user)
    coords_all     = reduce_2d(embeddings_all)

    embeddings = embeddings_all[:len(all_texts)]
    coords     = coords_all[:len(all_texts)]
    user_coords = coords_all[len(all_texts):]

    n_variants = len(results)
    fig = plt.figure(figsize=(22, 9))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2.5, 1.2], wspace=0.3, figure=fig)

    ax_scatter = fig.add_subplot(gs[0])
    offset = 0
    metrics_by_variant = {}

    for variant, tasks in results.items():
        n   = len(tasks)
        emb = embeddings[offset:offset + n]
        xy  = coords[offset:offset + n]
        s   = VARIANT_STYLE[variant]
        metrics_by_variant[variant] = diversity_metrics(emb)

        ax_scatter.scatter(xy[:, 0], xy[:, 1], c=s["color"], s=75, alpha=0.75,
                           edgecolors="white", linewidths=0.4, zorder=3,
                           label=f"{s['label']} (n={n})")
        for i, (x, y) in enumerate(xy):
            short = tasks[i][:40] + ("…" if len(tasks[i]) > 40 else "")
            ax_scatter.annotate(short, (x, y), fontsize=7, alpha=0.80,
                                xytext=(4, 4), textcoords="offset points",
                                color=s["color"])
        offset += n

    # Plot user-mentioned tasks as black stars
    if len(user_coords) > 0:
        ax_scatter.scatter(user_coords[:, 0], user_coords[:, 1],
                           c="black", marker="*", s=200, zorder=6,
                           label=f"User mentioned (n={len(user_tasks)})")
        for i, (x, y) in enumerate(user_coords):
            short = user_tasks[i][:40] + ("…" if len(user_tasks[i]) > 40 else "")
            ax_scatter.annotate(short, (x, y), fontsize=7, alpha=0.9,
                                xytext=(4, 4), textcoords="offset points",
                                color="black", fontweight="bold")

    all_x = coords_all[:, 0]; all_y = coords_all[:, 1]
    xlim  = (all_x.min() - 0.5, all_x.max() + 0.5)
    ylim  = (all_y.min() - 0.5, all_y.max() + 0.5)

    ax_scatter.set_xlim(xlim); ax_scatter.set_ylim(ylim)
    ax_scatter.set_title("All variants — shared UMAP space", fontsize=10)
    ax_scatter.set_xticks([]); ax_scatter.set_yticks([])
    ax_scatter.legend(fontsize=8, loc="best", framealpha=0.85)

    # ── metrics bar chart ────────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[1])
    metric_keys   = ["mean_pairwise", "min_pairwise", "centroid_spread"]
    metric_labels = ["Mean\npairwise", "Min\npairwise", "Centroid\nspread"]
    x     = np.arange(len(metric_keys))
    width = 0.18
    for i, (variant, metrics) in enumerate(metrics_by_variant.items()):
        vals = [metrics[k] for k in metric_keys]
        bars = ax_bar.bar(x + i * width, vals, width,
                          label=VARIANT_STYLE[variant]["label"],
                          color=VARIANT_STYLE[variant]["color"], alpha=0.85)
        for bar, v in zip(bars, vals):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.002,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
    ax_bar.set_xticks(x + width * (n_variants - 1) / 2)
    ax_bar.set_xticklabels(metric_labels, fontsize=8)
    ax_bar.set_ylabel("Cosine distance", fontsize=8)
    ax_bar.set_title("Diversity metrics", fontsize=9)
    ax_bar.legend(fontsize=7, loc="upper right")
    max_val = max(metrics_by_variant[v][k] for v in metrics_by_variant for k in metric_keys)
    ax_bar.set_ylim(0, max(max_val * 1.35, 0.05))

    uid = user["user_id"]
    fig.suptitle(
        f"Task generator variants — {uid}  |  {user['role_short']}",
        fontsize=12, y=1.01,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out_path}")

    # Console table
    print(f"\n  {'Variant':<12}  {'mean_pw':>9}  {'min_pw':>9}  {'spread':>9}  n")
    for variant, m in metrics_by_variant.items():
        n = len(results[variant])
        print(f"  {variant:<12}  {m['mean_pairwise']:>9.4f}  {m['min_pairwise']:>9.4f}  {m['centroid_spread']:>9.4f}  {n}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--user-ids", nargs="+")
    group.add_argument("--sample-users-path")
    parser.add_argument("--n-users",  type=int, default=2)
    parser.add_argument("--n-tasks", type=int, default=12,
                        help="Tasks to generate per variant (default 12)")
    parser.add_argument("--out-dir")
    args = parser.parse_args()

    if args.user_ids:
        user_ids = args.user_ids
    else:
        with open(args.sample_users_path) as f:
            user_ids = [u["User ID"] for u in json.load(f)][: args.n_users]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.base_path)

    for uid in user_ids:
        print(f"\n{'='*60}\nUser: {uid}")
        user = load_user_data(uid, args.base_path)
        if not user or not user["job_description"]:
            print("  No job description — skipping")
            continue
        print(f"  Role: {user['role_short']}")
        print(f"  Brain dump: {user['brain_dump'][:100]}" if user["brain_dump"] else "  Brain dump: (none)")

        results = {}
        for variant in ["scratch", "simple", "post_hoc", "combined"]:
            print(f"\n  Generating [{variant}] (target {args.n_tasks} tasks)…")
            tasks = generate_variant(variant, user, args.n_tasks)
            results[variant] = tasks
            print(f"  → {len(tasks)} tasks: {tasks[:3]}")

        out_path = out_dir / uid / "evaluations" / "task_generator_simulation.png"
        plot_user(user, results, out_path)

        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps({
            "user_id": uid,
            "job_description": user["job_description"],
            "brain_dump": user["brain_dump"],
            "variants": results,
        }, indent=2))
        print(f"  JSON  → {json_path}")


if __name__ == "__main__":
    main()
