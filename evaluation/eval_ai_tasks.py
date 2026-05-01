"""
eval_ai_tasks.py — Evaluate the AI task generator against real participant tasks.

Generates AI-related tasks (capability + governance) for each study participant
and scores them using the same LLM-judged metrics as eval_task_diversity_llm.py:

  redundancy_rate  — cross-sample redundancy within the ai_tasks strategy
  recall           — fraction of participant's real tasks matched by ai_tasks
  novelty_rate     — fraction of ai_tasks not matched by any real task

Recall and novelty are computed three ways:
  all       — against all participant tasks
  ai_only   — against participant tasks classified as AI-related
  non_ai    — against participant tasks classified as NOT AI-related

Matched pairs are identified by explicit LLM judgment (paraphrase/rewording only,
no subtask grouping). Results are compared against the other strategies from
eval_task_diversity_llm.py's cached output.

Usage:
    python evaluation/eval_ai_tasks.py \\
        --input   analysis/task_clustering/output/screened_study_tasks.json \\
        --cache-dir analysis/task_clustering/output/sim_cache \\
        --llm-cache-dir analysis/task_clustering/output/llm_diversity_cache
"""

import argparse
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

OPENAI_CLIENT = OpenAI()
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3
COMPARE_STRATEGIES = ["scratch", "simple", "post_hoc", "combined", "combined_no_gap", "combined_no_gap_participant", "combined_participant"]
# Strategies to include in the AI vs non-AI split comparison table
SPLIT_STRATEGIES = ["combined_no_gap_participant", "combined_participant"]


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _call_llm(messages: list[dict], model: str = DEFAULT_MODEL) -> str | None:
    for attempt in range(MAX_RETRIES):
        try:
            resp = OPENAI_CLIENT.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                response_format={"type": "json_object"},
            )
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"LLM call failed (attempt {attempt + 1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def _parse_json(text: str | None) -> dict | None:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Task classification: AI-related vs. not
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = """\
You are classifying work tasks for a {occupation}.

Below are tasks this person reported doing in their job. For each task, decide whether it is \
AI-related — meaning it involves directly using, developing, evaluating, or governing AI/ML tools, \
models, or systems as a meaningful part of the activity. Mark a task as AI-related only if AI is a \
central part of the task, not just incidentally mentioned.

Tasks:
{task_list}

Return JSON with a list of booleans, one per task, in the same order:
{{"ai_related": [true, false, true, ...]}}"""


def classify_real_tasks(
    uid: str,
    occupation: str,
    real_tasks: list[str],
    class_cache_dir: Path,
    model: str = DEFAULT_MODEL,
    surgery: bool = False,
) -> list[bool]:
    cache_path = class_cache_dir / f"{uid}_task_classes.json"
    if not surgery and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if isinstance(cached.get("ai_related"), list) and len(cached["ai_related"]) == len(real_tasks):
                return [bool(x) for x in cached["ai_related"]]
        except Exception:
            pass

    task_list_str = "\n".join(f"{i}: {t}" for i, t in enumerate(real_tasks))
    prompt = CLASSIFY_PROMPT.format(occupation=occupation, task_list=task_list_str)
    raw = _call_llm([{"role": "user", "content": prompt}], model=model)
    parsed = _parse_json(raw)
    if not parsed or "ai_related" not in parsed:
        logging.warning(f"[{uid}] Task classification failed; defaulting all to non-AI")
        return [False] * len(real_tasks)

    labels = [bool(x) for x in parsed["ai_related"]]
    if len(labels) != len(real_tasks):
        labels = (labels + [False] * len(real_tasks))[:len(real_tasks)]

    class_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"ai_related": labels}, indent=2))
    return labels


# ---------------------------------------------------------------------------
# Redundancy across samples
# ---------------------------------------------------------------------------

REDUNDANCY_PROMPT = """\
You are evaluating task diversity for a {occupation}.

Below are AI-related work tasks generated across {n_samples} different samples (labeled [sampleN.taskM]).
Your job: group tasks that represent the same underlying work activity.

Two tasks belong in the same group ONLY if they are paraphrases or rewordings of each other \
(the exact same activity). Do NOT group tasks just because one is a subtask or supertask of another.

Tasks:
{task_list}

Return JSON with one group per unique conceptual task. Every index must appear in exactly one group.
{{"groups": [{{"tasks": ["s0.t0", "s2.t3"], "label": "brief description"}}, ...]}}"""


def compute_redundancy(occupation: str, samples: list[list[str]], model: str = DEFAULT_MODEL) -> dict | None:
    entries: list[tuple[str, str]] = []
    for si, sample_tasks in enumerate(samples):
        for ti, task in enumerate(sample_tasks):
            entries.append((f"s{si}.t{ti}", task))
    if not entries:
        return None

    task_list_str = "\n".join(f"[{label}] {text}" for label, text in entries)
    prompt = REDUNDANCY_PROMPT.format(
        occupation=occupation,
        n_samples=len(samples),
        task_list=task_list_str,
    )
    raw = _call_llm([{"role": "user", "content": prompt}], model=model)
    parsed = _parse_json(raw)
    if not parsed or "groups" not in parsed:
        return None

    n_total = len(entries)
    n_unique = len(parsed["groups"])
    unique_rate = n_unique / n_total if n_total > 0 else 0.0
    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "unique_rate": round(unique_rate, 4),
        "redundancy_rate": round(1.0 - unique_rate, 4),
    }


# ---------------------------------------------------------------------------
# Recall + Novelty via explicit matched pairs
# ---------------------------------------------------------------------------

RECALL_NOVELTY_PROMPT = """\
You are comparing work tasks for a {occupation}.

REAL TASKS (the participant's actual tasks, labeled A0, A1, ...):
{real_tasks}

AI-GENERATED TASKS (labeled B0, B1, ...):
{gen_tasks}

Two tasks "match" ONLY if they describe the exact same underlying work activity \
(paraphrases or rewordings). Do NOT match tasks just because one is a subtask or \
supertask of another.

List every matched pair. A single B may match at most one A, and vice versa.

Return JSON:
{{"matches": [{{"a": 0, "b": 3}}, {{"a": 2, "b": 7}}]}}

Return an empty list if nothing matches: {{"matches": []}}"""


def compute_recall_novelty(
    occupation: str, real_tasks: list[str], gen_tasks: list[str], model: str = DEFAULT_MODEL
) -> dict | None:
    if not real_tasks or not gen_tasks:
        return None

    real_str = "\n".join(f"A{i}: {t}" for i, t in enumerate(real_tasks))
    gen_str  = "\n".join(f"B{i}: {t}" for i, t in enumerate(gen_tasks))
    prompt = RECALL_NOVELTY_PROMPT.format(
        occupation=occupation, real_tasks=real_str, gen_tasks=gen_str
    )
    raw = _call_llm([{"role": "user", "content": prompt}], model=model)
    parsed = _parse_json(raw)
    if not parsed or "matches" not in parsed:
        return None

    seen_a: set[int] = set()
    seen_b: set[int] = set()
    for m in parsed["matches"]:
        a, b = m.get("a"), m.get("b")
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        if not (0 <= a < len(real_tasks) and 0 <= b < len(gen_tasks)):
            continue
        if a in seen_a or b in seen_b:
            continue
        seen_a.add(a)
        seen_b.add(b)

    return {
        "recall": round(len(seen_a) / len(real_tasks), 4),
        "novelty_rate": round((len(gen_tasks) - len(seen_b)) / len(gen_tasks), 4),
        "n_real": len(real_tasks),
        "n_gen": len(gen_tasks),
        "n_real_covered": len(seen_a),
        "n_gen_novel": len(gen_tasks) - len(seen_b),
    }


# ---------------------------------------------------------------------------
# Per-user processing
# ---------------------------------------------------------------------------

def _avg_recall_novelty(
    occupation: str, task_subset: list[str], ai_samples: list[list[str]], model: str
) -> tuple[float | None, float | None]:
    if not task_subset:
        return None, None
    rows = []
    for gen_tasks in ai_samples:
        rn = compute_recall_novelty(occupation, task_subset, gen_tasks, model=model)
        if rn:
            rows.append(rn)
    if not rows:
        return None, None
    return (
        round(float(np.mean([r["recall"] for r in rows])), 4),
        round(float(np.mean([r["novelty_rate"] for r in rows])), 4),
    )


def process_user(
    user: dict,
    samples: list[dict],
    llm_cache_dir: Path,
    model: str = DEFAULT_MODEL,
    surgery: bool = False,
) -> dict | None:
    uid = user["user_id"]
    occupation = user.get("occupation", uid)
    cache_path = llm_cache_dir / f"{uid}_ai.json"

    # Load existing cache
    cached: dict | None = None
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached.get("error"):
                cached = None
        except Exception:
            cached = None

    # Skip entirely if cache is complete and not in surgery mode
    if not surgery and cached and "recall_ai" in cached:
        return cached

    real_tasks = [
        t["task_statement"]
        for t in user.get("tasks", [])
        if t.get("_screen_status") != "fail"
    ]
    if not real_tasks:
        return None

    ai_samples: list[list[str]] = [s.get("ai_tasks") or [] for s in samples]
    ai_samples = [s for s in ai_samples if s]
    if not ai_samples:
        logging.warning(f"[{uid}] No ai_tasks found in cache — skipping")
        return None

    # Reuse cached redundancy (expensive); recompute if missing
    redundancy = cached.get("redundancy") if cached else None
    if redundancy is None:
        redundancy = compute_redundancy(occupation, ai_samples, model=model)

    # Classify real tasks
    labels = classify_real_tasks(uid, occupation, real_tasks, llm_cache_dir, model=model, surgery=surgery)
    ai_real    = [t for t, is_ai in zip(real_tasks, labels) if is_ai]
    non_ai_real = [t for t, is_ai in zip(real_tasks, labels) if not is_ai]

    # Recall/novelty against all three subsets
    recall_all,    novelty_all    = _avg_recall_novelty(occupation, real_tasks,  ai_samples, model)
    recall_ai,     novelty_ai     = _avg_recall_novelty(occupation, ai_real,     ai_samples, model)
    recall_non_ai, novelty_non_ai = _avg_recall_novelty(occupation, non_ai_real, ai_samples, model)

    result = {
        "user_id": uid,
        "occupation": occupation,
        "category": user.get("category", ""),
        "redundancy": redundancy,
        "n_real": len(real_tasks),
        "n_ai_real": len(ai_real),
        "n_non_ai_real": len(non_ai_real),
        # all tasks
        "recall": recall_all,
        "novelty_rate": novelty_all,
        # AI-related tasks only
        "recall_ai": recall_ai,
        "novelty_ai": novelty_ai,
        # non-AI tasks only
        "recall_non_ai": recall_non_ai,
        "novelty_non_ai": novelty_non_ai,
        "n_samples": len(ai_samples),
    }

    llm_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Load samples
# ---------------------------------------------------------------------------

def load_samples(uid: str, cache_dir: Path, n_samples: int | None = None) -> list[dict]:
    sample_paths: dict[int, Path] = {}
    legacy = cache_dir / f"{uid}.json"
    if legacy.exists():
        sample_paths[0] = legacy
    sample_re = re.compile(rf"^{re.escape(uid)}_s(\d+)\.json$")
    for p in cache_dir.glob(f"{uid}_s*.json"):
        match = sample_re.match(p.name)
        if not match:
            continue
        si = int(match.group(1))
        if n_samples is not None and si >= n_samples:
            continue
        sample_paths[si] = p
    samples = []
    for si in sorted(sample_paths):
        try:
            samples.append(json.loads(sample_paths[si].read_text()))
        except Exception:
            pass
    return samples


# ---------------------------------------------------------------------------
# Per-strategy AI/non-AI split
# ---------------------------------------------------------------------------

def load_strategy_task_samples(uid: str, strategy: str, cache_dir: Path, n_samples: int | None = None) -> list[list[str]]:
    """Return list-of-task-lists (one per sample) for a given strategy from sim_cache."""
    samples = load_samples(uid, cache_dir, n_samples)
    result = []
    for s in samples:
        tasks = s.get(strategy) or []
        if tasks:
            result.append(tasks)
    return result


def process_user_split(
    user: dict,
    split_strategies: list[str],
    cache_dir: Path,
    llm_cache_dir: Path,
    model: str = DEFAULT_MODEL,
    surgery: bool = False,
) -> dict:
    uid = user["user_id"]
    occupation = user.get("occupation", uid)
    cache_path = llm_cache_dir / f"{uid}_split.json"

    split_cached: dict = {}
    if cache_path.exists():
        try:
            split_cached = json.loads(cache_path.read_text())
        except Exception:
            pass

    if not surgery and all(s in split_cached for s in split_strategies):
        return split_cached

    real_tasks = [
        t["task_statement"]
        for t in user.get("tasks", [])
        if t.get("_screen_status") != "fail"
    ]
    if not real_tasks:
        return split_cached

    labels = classify_real_tasks(uid, occupation, real_tasks, llm_cache_dir, model=model)
    ai_real    = [t for t, is_ai in zip(real_tasks, labels) if is_ai]
    non_ai_real = [t for t, is_ai in zip(real_tasks, labels) if not is_ai]

    for strategy in split_strategies:
        if not surgery and strategy in split_cached:
            continue
        strat_samples = load_strategy_task_samples(uid, strategy, cache_dir)
        if not strat_samples:
            continue
        recall_ai,     novelty_ai     = _avg_recall_novelty(occupation, ai_real,     strat_samples, model)
        recall_non_ai, novelty_non_ai = _avg_recall_novelty(occupation, non_ai_real, strat_samples, model)
        split_cached[strategy] = {
            "recall_ai":     recall_ai,
            "novelty_ai":    novelty_ai,
            "recall_non_ai": recall_non_ai,
            "novelty_non_ai": novelty_non_ai,
        }

    llm_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(split_cached, indent=2))
    return split_cached


# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------

def print_results(all_results: list[dict], other_cache_dir: Path | None, split_by_uid: dict[str, dict] | None = None) -> None:
    valid = [r for r in all_results if r and r.get("recall") is not None]

    def _agg(vals):
        v = [x for x in vals if x is not None]
        return (float(np.mean(v)), float(np.std(v))) if v else (None, None)

    ai_recall      = _agg([r["recall"] for r in valid])
    ai_novelty     = _agg([r["novelty_rate"] for r in valid])
    ai_redundancy  = _agg([r["redundancy"]["redundancy_rate"] for r in valid if r.get("redundancy")])
    ai_recall_ai   = _agg([r["recall_ai"] for r in valid if r.get("recall_ai") is not None])
    ai_recall_nai  = _agg([r["recall_non_ai"] for r in valid if r.get("recall_non_ai") is not None])
    ai_novelty_ai  = _agg([r["novelty_ai"] for r in valid if r.get("novelty_ai") is not None])
    ai_novelty_nai = _agg([r["novelty_non_ai"] for r in valid if r.get("novelty_non_ai") is not None])

    n_ai_tasks     = [r["n_ai_real"] for r in valid if "n_ai_real" in r]
    n_non_ai_tasks = [r["n_non_ai_real"] for r in valid if "n_non_ai_real" in r]

    print(f"\n{'='*60}")
    print("AI TASK GENERATOR — AGGREGATE (n={})".format(len(valid)))
    print(f"{'='*60}")
    fmt = lambda pair, label: f"  {label:<22}  {pair[0]:.3f} ± {pair[1]:.3f}" if pair[0] is not None else f"  {label:<22}  —"
    print(fmt(ai_redundancy,  "Redundancy rate↓"))
    print()
    print(f"  {'':22}  {'all tasks':>12}  {'AI tasks only':>13}  {'non-AI tasks':>12}")
    print(f"  {'-'*62}")
    def _row(label, all_v, ai_v, nai_v):
        def _fmt(pair):
            return f"{pair[0]:.3f}±{pair[1]:.3f}" if pair[0] is not None else "    —    "
        print(f"  {label:<22}  {_fmt(all_v):>12}  {_fmt(ai_v):>13}  {_fmt(nai_v):>12}")
    _row("Recall↑",       ai_recall,  ai_recall_ai,  ai_recall_nai)
    _row("Novelty rate↑", ai_novelty, ai_novelty_ai, ai_novelty_nai)

    if n_ai_tasks:
        mean_ai = np.mean(n_ai_tasks)
        mean_nai = np.mean(n_non_ai_tasks) if n_non_ai_tasks else 0
        print(f"\n  Avg AI-related tasks per participant:     {mean_ai:.1f}  ({mean_ai/(mean_ai+mean_nai)*100:.0f}%)")
        print(f"  Avg non-AI tasks per participant:         {mean_nai:.1f}  ({mean_nai/(mean_ai+mean_nai)*100:.0f}%)")

    # Comparison table against other strategies (recall/novelty against ALL tasks)
    if other_cache_dir and other_cache_dir.exists():
        strat_data: dict[str, list[dict]] = {s: [] for s in COMPARE_STRATEGIES}
        for p in other_cache_dir.glob("*.json"):
            if p.name.endswith("_ai.json") or p.name.endswith("_task_classes.json"):
                continue
            try:
                d = json.loads(p.read_text())
                for s in COMPARE_STRATEGIES:
                    if s in d and d[s]:
                        strat_data[s].append(d[s])
            except Exception:
                pass

        if any(strat_data.values()):
            print(f"\n\n{'─'*100}")
            print("STRATEGY COMPARISON (recall/novelty vs ALL participant tasks)")
            print(f"{'─'*100}")
            print(f"{'Metric':<20}", end="")
            cols = COMPARE_STRATEGIES + ["ai_tasks"]
            for c in cols:
                print(f"  {c[:16]:>16}", end="")
            print()
            print("-" * (20 + 18 * len(cols)))

            def _strat_agg(rows, key):
                vals = [r.get(key) for r in rows if r.get(key) is not None]
                return (np.mean(vals), np.std(vals)) if vals else (None, None)

            def _redund_agg(rows):
                vals = []
                for r in rows:
                    redund = r.get("redundancy")
                    if isinstance(redund, dict):
                        v = redund.get("redundancy_rate")
                    else:
                        v = r.get("redundancy_rate")
                    if v is not None:
                        vals.append(v)
                return (np.mean(vals), np.std(vals)) if vals else (None, None)

            for metric, label, ai_pair in [
                ("recall",         "Recall↑",        ai_recall),
                ("novelty_rate",   "Novelty rate↑",  ai_novelty),
                ("redundancy_rate","Redundancy↓",    ai_redundancy),
            ]:
                print(f"{label:<20}", end="")
                for s in COMPARE_STRATEGIES:
                    rows = strat_data[s]
                    if metric == "redundancy_rate":
                        m, sd = _redund_agg(rows)
                    else:
                        m, sd = _strat_agg(rows, metric)
                    if m is not None:
                        print(f"  {m:.3f}±{sd:.3f}".rjust(18), end="")
                    else:
                        print(f"  {'—':>16}", end="")
                m, sd = ai_pair
                if m is not None:
                    print(f"  {m:.3f}±{sd:.3f}".rjust(18), end="")
                print()

    # AI vs non-AI split comparison table
    if split_by_uid:
        # Collect per-strategy split aggregates
        split_strats = sorted({s for uid_data in split_by_uid.values() for s in uid_data})
        if split_strats:
            def _split_agg(strat, key):
                vals = [
                    split_by_uid[uid][strat][key]
                    for uid in split_by_uid
                    if strat in split_by_uid[uid] and split_by_uid[uid][strat].get(key) is not None
                ]
                return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

            cols = split_strats + ["ai_tasks"]
            col_w = 18
            print(f"\n\n{'─'*100}")
            print("RECALL vs AI-RELATED / NON-AI TASK SUBSETS")
            print(f"{'─'*100}")
            print(f"{'Metric':<24}", end="")
            for c in cols:
                print(f"  {c[:16]:>{col_w-2}}", end="")
            print()
            print("─" * (24 + col_w * len(cols)))

            def _fmt_pair(pair):
                return f"{pair[0]:.3f}±{pair[1]:.3f}" if pair[0] is not None else "—"

            for metric, label, ai_fallback in [
                ("recall_ai",      "Recall↑ (AI tasks)",     ai_recall_ai),
                ("recall_non_ai",  "Recall↑ (non-AI tasks)", ai_recall_nai),
                ("novelty_ai",     "Novelty↑ (AI tasks)",    ai_novelty_ai),
                ("novelty_non_ai", "Novelty↑ (non-AI tasks)",ai_novelty_nai),
            ]:
                print(f"{label:<24}", end="")
                for s in split_strats:
                    pair = _split_agg(s, metric)
                    print(f"  {_fmt_pair(pair):>{col_w-2}}", end="")
                # ai_tasks column from already-computed aggregates
                print(f"  {_fmt_pair(ai_fallback):>{col_w-2}}", end="")
                print()

    # Per-user table
    sorted_results = sorted(valid, key=lambda r: r.get("category", ""))
    print(f"\n\n{'─'*100}")
    print("PER-USER RESULTS")
    print(f"{'─'*100}")
    print(f"{'Occupation':<44}  {'n_ai':>4}  {'n_reg':>5}  {'redund':>6}  {'rec_all':>7}  {'rec_ai':>6}  {'rec_reg':>7}  {'novel_ai':>8}")
    print(f"{'─'*100}")
    prev_cat = None
    for r in sorted_results:
        if r.get("category") != prev_cat:
            print(f"\n── {r.get('category', '')} ──")
            prev_cat = r.get("category")
        occ   = r.get("occupation", r["user_id"])[:43]
        n_ai  = r.get("n_ai_real", "")
        n_reg = r.get("n_non_ai_real", "")
        red   = r["redundancy"]["redundancy_rate"] if r.get("redundancy") else None
        rec_all = r.get("recall")
        rec_ai  = r.get("recall_ai")
        rec_reg = r.get("recall_non_ai")
        nov_ai  = r.get("novelty_ai")

        def _f(v): return f"{v:.3f}" if v is not None else "  —  "
        print(f"  {occ:<42}  {n_ai!s:>4}  {n_reg!s:>5}  {_f(red):>6}  {_f(rec_all):>7}  {_f(rec_ai):>6}  {_f(rec_reg):>7}  {_f(nov_ai):>8}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AI task generator: redundancy, recall, novelty")
    parser.add_argument("--input", required=True)
    parser.add_argument("--cache-dir", required=True, help="sim_cache with ai_tasks per user")
    parser.add_argument("--llm-cache-dir", default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--surgery", action="store_true")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    cache_dir = Path(args.cache_dir)
    llm_cache_dir = Path(args.llm_cache_dir) if args.llm_cache_dir else cache_dir.parent / "llm_diversity_cache"

    users_with_samples = []
    for user in data:
        uid = user["user_id"]
        samples = load_samples(uid, cache_dir, args.n_samples)
        if samples and any(s.get("ai_tasks") for s in samples):
            users_with_samples.append((user, samples))

    if args.num_users:
        users_with_samples = users_with_samples[: args.num_users]

    print(f"{len(users_with_samples)} users with ai_tasks data\n")

    all_results = []
    split_by_uid: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        ai_futures = {
            executor.submit(process_user, user, samples, llm_cache_dir, args.model, args.surgery): user["user_id"]
            for user, samples in users_with_samples
        }
        split_futures = {
            executor.submit(
                process_user_split, user, SPLIT_STRATEGIES, cache_dir, llm_cache_dir, args.model, args.surgery
            ): user["user_id"]
            for user, _ in users_with_samples
        }

        for future in tqdm(as_completed(ai_futures), total=len(ai_futures), desc="Evaluating AI tasks"):
            uid = ai_futures[future]
            try:
                r = future.result()
                if r:
                    all_results.append(r)
            except Exception as e:
                logging.error(f"Error for user {uid}: {e}")

        for future in tqdm(as_completed(split_futures), total=len(split_futures), desc="Strategy split"):
            uid = split_futures[future]
            try:
                r = future.result()
                if r:
                    split_by_uid[uid] = r
            except Exception as e:
                logging.error(f"Split error for user {uid}: {e}")

    print_results(all_results, llm_cache_dir, split_by_uid)


if __name__ == "__main__":
    main()
