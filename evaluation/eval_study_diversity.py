"""
Evaluate and visualize task diversity for real user-study participants,
grouped by occupational category.

Each per-group subplot shows five task sources in a shared UMAP space:
  - Real tasks (gray)           — user study ground truth
  - O*NET reference (navy)      — standard ONET task statements for the occupation
  - Scratch (gray-blue)         — generated from role description only
  - Simple (blue)               — generated with explicit session-perspective hints
  - Post-hoc (orange)           — generated anchored on brain-dump / gap-fill
  - Combined (purple)           — session perspective + brain-dump gap-fill

Usage:
    # Generate tasks + plot (skips users already cached)
    python evaluation/eval_study_diversity.py \
        --input  analysis/task_clustering/output/screened_study_tasks.json \
        --onet   "analysis/onet/Task Statements O*NET 30.2.xlsx" \
        --out    analysis/task_clustering/output/study_diversity_plot.png

    # Force regeneration:
        --surgery

    # Skip generation, just plot from cache:
        --no-generate

    # Metrics table only (no LLM calls, no plot):
        --metrics-only
"""

import argparse
import json
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.utils.llm.engines import get_engine, invoke_engine

OPENAI_CLIENT  = OpenAI()
EMBED_MODEL    = "text-embedding-3-large"
TASK_GEN_MODEL = os.getenv("TASK_GEN_MODEL", "claude-sonnet-4-6")
TASK_BATCH_SIZE = 3
CORE_BATCHES    = 2
ONET_TASKS_PER_CODE = 8   # max ONET task statements to include per O*NET-SOC code
TASK_VARIANTS = ("scratch", "simple", "post_hoc", "combined", "combined_no_gap", "combined_no_gap_participant", "combined_participant", "ai_tasks")
SESSION_FOCUS_VARIANTS = ("simple", "combined", "combined_no_gap", "combined_no_gap_participant", "combined_participant")
BRAIN_DUMP_VARIANTS = ("post_hoc", "combined", "combined_no_gap", "combined_no_gap_participant", "combined_participant")
GAP_FILL_VARIANTS = ("post_hoc", "combined", "combined_participant")
PARTICIPANT_SPECIFIC_VARIANTS = ("combined_no_gap_participant", "combined_participant")
AI_TASK_VARIANTS = ("ai_tasks",)
TASK_GEN_TIMEOUT_SECONDS = float(os.getenv("TASK_GEN_TIMEOUT_SECONDS", "120"))
TASK_GEN_MAX_RETRIES = int(os.getenv("TASK_GEN_MAX_RETRIES", "2"))
_TASK_ENGINE_LOCAL = threading.local()

VARIANT_STYLE = {
    "real":     {"color": "#7f8c8d", "marker": "o", "label": "Real (user study)",    "size": 60,  "zorder": 5, "alpha": 0.9},
    "onet":     {"color": "#1a252f", "marker": "D", "label": "O*NET reference",      "size": 50,  "zorder": 4, "alpha": 0.75},
    "scratch":  {"color": "#95a5a6", "marker": "o", "label": "Scratch",              "size": 40,  "zorder": 3, "alpha": 0.6},
    "simple":   {"color": "#3498db", "marker": "o", "label": "Simple (categories)",  "size": 40,  "zorder": 3, "alpha": 0.6},
    "post_hoc": {"color": "#e67e22", "marker": "o", "label": "Post-hoc (gap-fill)",  "size": 40,  "zorder": 3, "alpha": 0.6},
    "combined": {"color": "#8e44ad", "marker": "o", "label": "Combined",             "size": 40,  "zorder": 3, "alpha": 0.6},
    "combined_no_gap": {"color": "#16a085", "marker": "o", "label": "Combined (no gap-fill)", "size": 40, "zorder": 3, "alpha": 0.6},
    "combined_no_gap_participant": {"color": "#e74c3c", "marker": "o", "label": "Combined (participant-specific)", "size": 40, "zorder": 3, "alpha": 0.6},
    "combined_participant": {"color": "#c0392b", "marker": "s", "label": "Combined (participant+gap-fill)", "size": 40, "zorder": 3, "alpha": 0.6},
    "ai_tasks": {"color": "#f39c12", "marker": "^", "label": "AI tasks", "size": 50, "zorder": 4, "alpha": 0.7},
}

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


# ---------------------------------------------------------------------------
# ONET loader
# ---------------------------------------------------------------------------

def load_onet_tasks(xlsx_path: str) -> dict[str, list[str]]:
    """Return {onet_code: [task_statement, ...]} limited to core tasks."""
    import pandas as pd
    df = pd.read_excel(xlsx_path)
    df = df[df["Task Type"] == "Core"]
    result: dict[str, list[str]] = {}
    for code, grp in df.groupby("O*NET-SOC Code"):
        result[code] = grp["Task"].dropna().tolist()
    return result


# ---------------------------------------------------------------------------
# Embedding & projection
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    BATCH = 100
    vecs = []
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        resp = OPENAI_CLIENT.embeddings.create(input=batch, model=EMBED_MODEL)
        vecs.append(np.array([r.embedding for r in resp.data], dtype=np.float32))
    mat = np.vstack(vecs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1.0, norms)


def reduce_2d(embeddings: np.ndarray) -> np.ndarray:
    try:
        import umap as umap_module
        n = len(embeddings)
        reducer = umap_module.UMAP(n_components=2, random_state=42,
                                    n_neighbors=min(15, n - 1))
        return reducer.fit_transform(embeddings)
    except Exception as e:
        print(f"UMAP failed ({e}), falling back to PCA")
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(embeddings)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def diversity_metrics(embeddings: np.ndarray) -> dict:
    n = len(embeddings)
    if n < 2:
        return {"mean_pairwise_dist": None, "min_pairwise_dist": None,
                "centroid_spread": None, "n_tasks": n}
    sim = embeddings @ embeddings.T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    mask = ~np.eye(n, dtype=bool)
    pairwise = dist[mask]
    centroid = embeddings.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    centroid_dists = np.clip(1.0 - (embeddings @ centroid_norm), 0.0, 2.0)
    return {
        "mean_pairwise_dist": round(float(np.mean(pairwise)), 4),
        "min_pairwise_dist":  round(float(np.min(pairwise)),  4),
        "centroid_spread":    round(float(np.mean(centroid_dists)), 4),
        "n_tasks": n,
    }


# ---------------------------------------------------------------------------
# Task generation (matches simulate_task_generator.py logic)
# ---------------------------------------------------------------------------

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

_RULES_PARTICIPANT = (
    "Rules:\n"
    "- Generate tasks SPECIFIC to this individual — ground each task in their exact domain, tools, industry, "
    "and work context as described above. Do NOT fall back to generic occupational tasks.\n"
    "- Tasks must be DISTINCT — no two tasks should describe the same activity even with different wording\n"
    "- Each task: 5–12 words, starts with an action verb, sentence case\n"
    "- Be specific — reference concrete objects, systems, or workflows the participant mentioned\n"
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


def _prior_block(prior: list[str]) -> str:
    if not prior:
        return ""
    return (
        "\nTasks already shown (STRICT: do NOT repeat or rephrase any of these):\n"
        + "\n".join(f"- {t}" for t in prior) + "\n"
    )


def _coverage_hint(b: int) -> str:
    if b < CORE_BATCHES:
        return (
            f"Return the {TASK_BATCH_SIZE} {'most' if b == 0 else 'next most'} important "
            f"core tasks for this role (batch {b+1} of {CORE_BATCHES} core batches)."
        )
    return (
        f"The first {CORE_BATCHES * TASK_BATCH_SIZE} tasks covered the core. "
        f"Now return {TASK_BATCH_SIZE} tasks from a DIFFERENT area not yet represented."
    )


def _build_prompt(variant: str, jd: str, brain_dump: str, prior: list[str], b: int) -> str:
    perspective = SESSION_PERSPECTIVES[b % len(SESSION_PERSPECTIVES)]
    tw_block = ""
    if brain_dump:
        task_instructions = [
            "Expand high-level mentions (e.g. 'meetings' → specific meeting types).",
            "Maximize breadth across their full work profile.",
        ]
        if variant in GAP_FILL_VARIANTS:
            task_instructions.insert(
                0,
                "Prioritize areas NOT mentioned — probe gaps in their self-description.",
            )
        tw_block = (
            f"\nParticipant's typical work week description:\n\"{brain_dump}\"\n\n"
            "Generate tasks that:\n"
            + "\n".join(
                f"{i}. {instruction}"
                for i, instruction in enumerate(task_instructions, start=1)
            )
            + "\n"
        )
    pline = f"Session focus: {perspective}\n" if variant in SESSION_FOCUS_VARIANTS else ""
    tw = tw_block if variant in BRAIN_DUMP_VARIANTS else ""
    rules = _RULES_PARTICIPANT if variant in PARTICIPANT_SPECIFIC_VARIANTS else _RULES
    return (
        f"Occupation context: \"{jd}\"\n"
        f"{pline}{tw}"
        f"{_prior_block(prior)}\n"
        f"{_coverage_hint(b)}\n\n{rules}"
    )


def _call_and_parse(prompt: str) -> list[str]:
    engine = getattr(_TASK_ENGINE_LOCAL, "engine", None)
    if engine is None:
        engine = get_engine(
            model_name=TASK_GEN_MODEL,
            temperature=0.9,
            timeout=TASK_GEN_TIMEOUT_SECONDS,
            max_retries=TASK_GEN_MAX_RETRIES,
        )
        _TASK_ENGINE_LOCAL.engine = engine
    response = invoke_engine(engine, prompt)
    raw = (response.content if hasattr(response, "content") else str(response)).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            task_items = parsed.get("tasks") or []
        elif isinstance(parsed, list):
            task_items = parsed
        else:
            task_items = []
        names = []
        for task in task_items:
            if isinstance(task, dict):
                name = _clean_task_name(task.get("name"))
            else:
                name = _clean_task_name(task)
            if name:
                names.append(name)
        return names
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
        return names


_AI_BATCH_FOCUS = [
    "Generate the {n} most common new capabilities AI has given people in this role — "
    "things they can now do that weren't possible or practical before.",
    "Generate the {n} most common new oversight or quality-control responsibilities "
    "this role now has because AI is in the workflow — reviewing, verifying, or taking ownership of AI outputs.",
    "Generate {n} AI-related tasks specifically around writing, drafting, editing, or communicating — "
    "not yet listed above. Focus on how AI changes how people in this role produce or polish text.",
    "Generate {n} AI-related tasks specifically around data analysis, research, summarization, or decision-making — "
    "not yet listed above. Focus on how AI helps people in this role make sense of information.",
    "Generate {n} AI-related tasks specifically around automating repetitive work, writing code, or streamlining workflows — "
    "not yet listed above. Focus on how AI saves time or handles mechanical parts of the job.",
    "Generate {n} AI-related tasks specifically around learning new things, onboarding others, training, or building skills — "
    "not yet listed above. Focus on how AI accelerates or supports knowledge transfer in this role.",
]

_AI_RULES = (
    "Rules:\n"
    "- The task NAME must make it explicit that AI is involved — include 'using AI', 'with AI', or a specific AI tool\n"
    "- Tasks should apply broadly to the occupational category, not be hyper-specific to this individual\n"
    "- Tasks must be DISTINCT — no two tasks should describe the same activity\n"
    "- Each task: 5–12 words, starts with an action verb, sentence case\n"
    "- Use plain, everyday language\n"
    "- Label each with ai_type: 'capability' or 'governance'\n"
    "- Each task needs a name (5–12 words) and a description (1 sentence).\n"
    "- Return ONLY valid JSON (no markdown fences):\n"
    '  {"tasks": [{"name": "Task name here", "description": "One sentence.", "ai_type": "capability"}, ...], "has_more": true}'
)


def _build_ai_prompt(jd: str, prior: list[str], b: int) -> str:
    focus = _AI_BATCH_FOCUS[b % len(_AI_BATCH_FOCUS)].format(n=TASK_BATCH_SIZE)
    prior_block = _prior_block(prior) if prior else ""
    return (
        f"Occupation context: \"{jd}\"\n"
        f"{prior_block}\n"
        f"{focus}\n\n"
        f"{_AI_RULES}"
    )


def generate_variant(variant: str, jd: str, brain_dump: str, target: int) -> list[str]:
    tasks: list[str] = []
    b = 0
    if variant in AI_TASK_VARIANTS:
        while len(tasks) < target and b < target:
            tasks.extend(_call_and_parse(_build_ai_prompt(jd, tasks[:], b)))
            b += 1
        return tasks[:target]
    while len(tasks) < target and b < target:
        new = _call_and_parse(_build_prompt(variant, jd, brain_dump, tasks[:], b))
        tasks.extend(new)
        b += 1
    return tasks[:target]


def build_brain_dump(user: dict) -> str:
    seen: set[str] = set()
    lines: list[str] = []
    for t in user["tasks"]:
        for v in (t.get("sources") or {}).values():
            v = (v or "").strip()
            if len(v.split()) >= 8 and v not in seen:
                seen.add(v)
                lines.append(v)
    return " ".join(lines[:6])


def generate_user(user: dict, target: int, cache_dir: Path, surgery: bool,
                  variants_to_generate: tuple[str, ...],
                  sample_idx: int = 0) -> dict:
    uid = user["user_id"]
    cache_path = cache_dir / f"{uid}_s{sample_idx}.json"
    # Backward-compat: sample 0 can fall back to legacy filename
    legacy_path = cache_dir / f"{uid}.json"
    cached: dict = {}
    source_path: Path | None = None
    if not surgery:
        for p in (cache_path, legacy_path if sample_idx == 0 else None):
            if p and p.exists():
                try:
                    cached = json.loads(p.read_text())
                    source_path = p
                    break
                except Exception:
                    pass
        if cached and all(v in cached for v in variants_to_generate):
            if source_path == legacy_path:
                # Migrate to new name
                cache_path.write_text(source_path.read_text())
            return cached

    jd = user["occupation"]
    brain_dump = build_brain_dump(user)
    result = {
        "user_id": uid,
        "brain_dump": cached.get("brain_dump", brain_dump),
        "sample": sample_idx,
        **{k: v for k, v in cached.items() if k not in ("user_id", "brain_dump", "sample")},
    }
    for variant in variants_to_generate:
        if not surgery and variant in result:
            continue
        print(f"  [{uid[:12]}] s{sample_idx} {variant}…", flush=True)
        result[variant] = generate_variant(variant, jd, brain_dump, target)

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(result, indent=2))
    tmp_path.replace(cache_path)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="screened_study_tasks.json")
    parser.add_argument("--onet",  default=None,  help="ONET Task Statements xlsx")
    parser.add_argument("--out",   default=None,  help="Output PNG path")
    parser.add_argument("--cache-dir", default=None, help="Dir for per-user generation cache")
    parser.add_argument("--n-tasks",   type=int, default=10)
    parser.add_argument("--n-samples", type=int, default=1, help="Independent generation runs per user")
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(TASK_VARIANTS),
        choices=list(TASK_VARIANTS),
        help="Only generate/evaluate these generated variants",
    )
    parser.add_argument("--no-generate", action="store_true", help="Skip generation, plot from cache only")
    parser.add_argument("--surgery",     action="store_true", help="Regenerate even if cached")
    parser.add_argument("--metrics-only", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())

    input_dir = Path(args.input).parent
    cache_dir  = Path(args.cache_dir) if args.cache_dir else input_dir / "sim_cache"

    # Load ONET tasks
    onet_db: dict[str, list[str]] = {}
    if args.onet:
        print("Loading O*NET task statements…", flush=True)
        onet_db = load_onet_tasks(args.onet)

    # ── Generation ────────────────────────────────────────────────────────────
    # generated[uid] = list of per-sample dicts (one per sample_idx)
    generated: dict[str, list[dict]] = {}
    n_samples = args.n_samples
    variants_to_generate = tuple(args.variants)

    if not args.no_generate and not args.metrics_only:
        total = len(data) * n_samples
        calls_per_variant = (args.n_tasks + TASK_BATCH_SIZE - 1) // TASK_BATCH_SIZE
        est_llm_calls = total * len(variants_to_generate) * calls_per_variant
        print(f"\nGenerating tasks for {len(data)} users × {n_samples} samples "
              f"(target={args.n_tasks} per variant, {total} sample runs total, "
              f"{len(variants_to_generate)} variant(s), "
              f"~{est_llm_calls} LLM calls if batches fill)…",
              flush=True)
        failures: list[tuple[str, int, str]] = []
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = {
                pool.submit(generate_user, user, args.n_tasks, cache_dir,
                            args.surgery, variants_to_generate, si): (user["user_id"], si)
                for user in data for si in range(n_samples)
            }
            for completed, fut in enumerate(as_completed(futures), start=1):
                uid, si = futures[fut]
                try:
                    result = fut.result()
                    generated.setdefault(uid, [None] * n_samples)
                    generated[uid][si] = result
                except Exception as e:
                    failures.append((uid, si, str(e)))
                    print(f"  ERROR [{uid} s{si}]: {e}", flush=True)
                else:
                    print(f"  DONE [{uid[:12]}] s{si} ({completed}/{total})",
                          flush=True)
        if failures:
            print(f"\nCompleted with {len(failures)} failed sample run(s).",
                  flush=True)
    else:
        # Load all cached samples
        for user in data:
            uid = user["user_id"]
            samples = []
            for si in range(n_samples):
                for p in (cache_dir / f"{uid}_s{si}.json",
                          cache_dir / f"{uid}.json" if si == 0 else None):
                    if p and p.exists():
                        try:
                            samples.append(json.loads(p.read_text()))
                            break
                        except Exception:
                            pass
            if samples:
                generated[uid] = samples

    # ── Build flat task list with metadata ───────────────────────────────────
    # Each entry: (source, category, user_id, text)
    # source ∈ {"real", "onet", "scratch", "simple", "post_hoc", "combined"}
    all_texts: list[str] = []
    task_meta: list[tuple[str, str, str]] = []  # (source, category, user_id)

    for user in data:
        uid = user["user_id"]
        cat = user.get("category", "Unknown")
        onet_code = user.get("onet_code", "")

        # Real tasks
        for t in user["tasks"]:
            if t.get("_screen_status") != "fail":
                all_texts.append(t["task_statement"])
                task_meta.append(("real", cat, uid))

        # O*NET tasks for this user's occupation code
        if onet_code and onet_code in onet_db:
            for stmt in onet_db[onet_code][:ONET_TASKS_PER_CODE]:
                all_texts.append(stmt)
                task_meta.append(("onet", cat, uid))

        # Generated variants — use sample 0 for the plot
        if uid in generated:
            samples = generated[uid] if isinstance(generated[uid], list) else [generated[uid]]
            gen = next((sample for sample in samples if sample), None)
            if gen:
                for variant in variants_to_generate:
                    for task in (gen.get(variant) or []):
                        all_texts.append(task)
                        task_meta.append((variant, cat, uid))

    print(f"\nEmbedding {len(all_texts)} tasks total…", flush=True)
    embeddings = embed_texts(all_texts)

    # ── Per-user diversity on real tasks ──────────────────────────────────────
    user_real_embs: dict[str, np.ndarray] = {}
    for i, (src, cat, uid) in enumerate(task_meta):
        if src == "real":
            user_real_embs.setdefault(uid, []).append(i)

    user_metrics: dict[str, dict] = {}
    for user in data:
        uid = user["user_id"]
        idxs = user_real_embs.get(uid, [])
        if len(idxs) < 2:
            continue
        m = diversity_metrics(embeddings[idxs])
        user_metrics[uid] = {
            "occupation": user["occupation"],
            "category": user.get("category", "Unknown"),
            **m,
        }

    # ── Per-category summary ──────────────────────────────────────────────────
    cat_groups: dict[str, list[dict]] = {}
    for m in user_metrics.values():
        cat_groups.setdefault(m["category"], []).append(m)

    cat_summary: dict[str, dict] = {}
    print(f"\n{'Category':<38} {'users':>5} {'mean_pw':>8} {'min_pw':>8}")
    print("-" * 60)
    for cat in sorted(cat_groups):
        rows = [r for r in cat_groups[cat] if r["mean_pairwise_dist"] is not None]
        if not rows:
            continue
        mpw  = np.mean([r["mean_pairwise_dist"] for r in rows])
        mnpw = np.mean([r["min_pairwise_dist"]  for r in rows])
        cs   = np.mean([r["centroid_spread"]     for r in rows])
        cat_summary[cat] = {"mean_pairwise_dist": mpw, "min_pairwise_dist": mnpw,
                             "centroid_spread": cs, "n_users": len(rows)}
        print(f"{cat:<38} {len(rows):>5} {mpw:>8.4f} {mnpw:>8.4f}")

    all_valid = [m for m in user_metrics.values() if m["mean_pairwise_dist"] is not None]
    overall_mean = np.mean([m["mean_pairwise_dist"] for m in all_valid])
    print(f"\nOverall mean pairwise (real tasks): {overall_mean:.4f}  ({len(all_valid)} users)")

    if args.metrics_only:
        return

    # ── 2-D projection ────────────────────────────────────────────────────────
    coords = reduce_2d(embeddings)

    # ── Index tasks by category ───────────────────────────────────────────────
    cat_task_idx: dict[str, list[tuple[int, str, str]]] = {}
    for i, (src, cat, uid) in enumerate(task_meta):
        cat_task_idx.setdefault(cat, []).append((i, src, uid))

    categories = sorted(cat_summary.keys())
    n_cats = len(categories)
    source_order = ["real", "onet", *variants_to_generate]

    # ── Per-strategy diversity metrics (across all users, then per category) ──
    # strategy_embs[source] = list of per-user embedding arrays
    strategy_global: dict[str, list[int]] = {}   # source → global task indices
    cat_strategy_embs: dict[str, dict[str, list[int]]] = {}  # cat → source → indices
    for i, (src, cat, uid) in enumerate(task_meta):
        strategy_global.setdefault(src, []).append(i)
        cat_strategy_embs.setdefault(cat, {}).setdefault(src, []).append(i)

    def _mean_pw(idxs: list[int]) -> float | None:
        if len(idxs) < 2:
            return None
        emb = embeddings[idxs]
        sim = emb @ emb.T
        dist = np.clip(1.0 - sim, 0.0, 2.0)
        mask = ~np.eye(len(idxs), dtype=bool)
        return round(float(dist[mask].mean()), 4)

    # Aggregate per-strategy (global)
    print(f"\n{'Strategy':<18} {'n_tasks':>8} {'mean_pw':>9} {'min_pw':>9}")
    print("-" * 50)
    strategy_global_metrics: dict[str, dict] = {}
    for src in source_order:
        idxs = strategy_global.get(src, [])
        mpw = _mean_pw(idxs)
        # min pairwise: compute properly
        if len(idxs) >= 2:
            emb = embeddings[idxs]
            sim = emb @ emb.T
            dist = np.clip(1.0 - sim, 0.0, 2.0)
            mask = ~np.eye(len(idxs), dtype=bool)
            mnpw = round(float(dist[mask].min()), 4)
        else:
            mnpw = None
        strategy_global_metrics[src] = {"mean_pw": mpw, "min_pw": mnpw, "n": len(idxs)}
        label = VARIANT_STYLE[src]["label"]
        print(f"{label:<18} {len(idxs):>8} {(mpw if mpw else 0):>9.4f} {(mnpw if mnpw else 0):>9.4f}")

    # Per-category per-strategy mean pairwise
    cat_src_mpw: dict[str, dict[str, float | None]] = {}
    for cat in categories:
        cat_src_mpw[cat] = {}
        for src in source_order:
            idxs = cat_strategy_embs.get(cat, {}).get(src, [])
            cat_src_mpw[cat][src] = _mean_pw(idxs)

    # ── Interactive HTML via Plotly ───────────────────────────────────────────
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError:
        print("plotly not installed. Run: pip install plotly")
        sys.exit(1)

    PLOTLY_STYLE = {
        "real":     {"color": "#5d6d7e", "symbol": "circle",      "size": 10, "opacity": 0.9,  "label": "Real (user study)"},
        "onet":     {"color": "#1a252f", "symbol": "diamond",     "size": 9,  "opacity": 0.8,  "label": "O*NET reference"},
        "scratch":  {"color": "#aab7b8", "symbol": "circle-open", "size": 8,  "opacity": 0.7,  "label": "Scratch"},
        "simple":   {"color": "#3498db", "symbol": "circle",      "size": 7,  "opacity": 0.65, "label": "Simple (categories)"},
        "post_hoc": {"color": "#e67e22", "symbol": "circle",      "size": 7,  "opacity": 0.65, "label": "Post-hoc (gap-fill)"},
        "combined": {"color": "#8e44ad", "symbol": "circle",      "size": 7,  "opacity": 0.65, "label": "Combined"},
        "combined_no_gap": {"color": "#16a085", "symbol": "circle", "size": 7, "opacity": 0.65, "label": "Combined (no gap-fill)"},
        "combined_no_gap_participant": {"color": "#e74c3c", "symbol": "circle", "size": 7, "opacity": 0.65, "label": "Combined (participant-specific)"},
        "combined_participant": {"color": "#c0392b", "symbol": "square", "size": 7, "opacity": 0.65, "label": "Combined (participant+gap-fill)"},
        "ai_tasks": {"color": "#f39c12", "symbol": "triangle-up", "size": 9, "opacity": 0.75, "label": "AI tasks"},
    }

    ncols = 4
    nrows = (n_cats + ncols - 1) // ncols

    src_abbrev = {"real": "real", "onet": "onet", "scratch": "scr",
                  "simple": "sim", "post_hoc": "ph", "combined": "comb",
                  "combined_no_gap": "nogap", "combined_no_gap_participant": "part",
                  "combined_participant": "cp", "ai_tasks": "ai"}

    subplot_titles = []
    for cat in categories:
        m = cat_summary.get(cat)
        color = "green" if m and m["mean_pairwise_dist"] >= overall_mean else "crimson"
        metrics_line = "  ".join(
            f"{src_abbrev[s]}={cat_src_mpw[cat].get(s):.3f}"
            for s in source_order
            if cat_src_mpw[cat].get(s) is not None
        )
        if m:
            subplot_titles.append(
                f"<span style='color:{color}'><b>{cat}</b></span>"
                f"  ({m['n_users']} users)<br>"
                f"<sup>{metrics_line}</sup>"
            )
        else:
            subplot_titles.append(cat)
    # Pad empty cells
    subplot_titles += [""] * (nrows * ncols - n_cats)

    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.04,
        vertical_spacing=0.12,
    )

    legend_shown: set[str] = set()
    uid_to_occ = {u["user_id"]: u["occupation"] for u in data}

    for cat_i, cat in enumerate(categories):
        row = cat_i // ncols + 1
        col = cat_i % ncols + 1
        entries = cat_task_idx.get(cat, [])

        for src in source_order:
            st = PLOTLY_STYLE[src]
            src_entries = [(idx, uid) for idx, s, uid in entries if s == src]
            if not src_entries:
                continue

            idxs = [idx for idx, _ in src_entries]
            uids = [uid for _, uid in src_entries]
            xs   = [float(coords[idx, 0]) for idx in idxs]
            ys   = [float(coords[idx, 1]) for idx in idxs]

            hover = [
                f"<b>{all_texts[idx]}</b><br>"
                f"<i>{uid_to_occ.get(uid, uid)}</i><br>"
                f"<span style='color:gray'>source: {src}</span>"
                for idx, uid in zip(idxs, uids)
            ]

            show_legend = src not in legend_shown
            if show_legend:
                legend_shown.add(src)

            fig.add_trace(
                go.Scatter(
                    x=xs, y=ys,
                    mode="markers",
                    marker=dict(
                        color=st["color"],
                        symbol=st["symbol"],
                        size=st["size"],
                        opacity=st["opacity"],
                        line=dict(width=0.5, color="white"),
                    ),
                    name=st["label"],
                    legendgroup=src,
                    legendgrouptitle=None,
                    showlegend=show_legend,
                    hovertext=hover,
                    hoverinfo="text",
                    hoverlabel=dict(bgcolor="white", font_size=12),
                ),
                row=row, col=col,
            )

    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False)

    fig.update_layout(
        title=dict(
            text=(
                f"Task Diversity per Occupational Group — UMAP "
                f"({len(all_texts)} tasks, {len(data)} users)<br>"
                f"<sup>Overall mean pairwise (real tasks) = {overall_mean:.3f} | "
                f"Click legend to show/hide strategies</sup>"
            ),
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.06,
            xanchor="center",
            x=0.5,
            itemsizing="constant",
            font=dict(size=12),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#ccc",
            borderwidth=1,
        ),
        width=1600,
        height=350 * nrows + 100,
        paper_bgcolor="white",
        plot_bgcolor="white",
        hoverdistance=8,
    )

    out = Path(args.out) if args.out else input_dir / "study_diversity_plot.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
