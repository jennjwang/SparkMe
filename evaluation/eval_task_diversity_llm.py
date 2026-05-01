"""
eval_task_diversity_llm.py — LLM-judged task diversity evaluation.

Two metrics, compared across strategies (scratch, simple, post_hoc, combined, combined_no_gap):

  Cross-sample redundancy
    Pool tasks from all N samples for a given (user, strategy).
    LLM groups them by conceptual uniqueness; a task and its subtask are the same.
    unique_rate     = n_unique_groups / n_total_tasks   (higher = less redundant)
    redundancy_rate = 1 - unique_rate                   (lower is better)

  Recall and novelty vs. participant's real tasks (per sample, averaged)
    recall       = fraction of participant's real tasks covered by generated tasks
    novelty_rate = fraction of generated tasks not present in participant's real tasks
    Subtask relationships count as "same task" in both directions.

Usage:
    python evaluation/eval_task_diversity_llm.py \\
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
STRATEGIES = ["scratch", "simple", "post_hoc", "combined", "combined_no_gap", "combined_no_gap_participant", "combined_participant"]
DEFAULT_MODEL = "gpt-4.1-mini"
MAX_RETRIES = 3


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
        # Try to extract JSON from markdown code blocks
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Redundancy: group tasks across samples by conceptual uniqueness
# ---------------------------------------------------------------------------

REDUNDANCY_PROMPT = """\
You are evaluating task diversity for a {occupation}.

Below are work tasks generated across {n_samples} different samples (labeled [sampleN.taskM]).
Your job: group tasks that represent the same underlying work activity.

Two tasks belong in the same group ONLY if they are paraphrases or rewordings of each other \
(i.e., they describe the exact same activity). Do NOT group tasks just because one is a \
subtask or supertask of another — "send weekly status email" and "communicate project updates \
to stakeholders" are DIFFERENT tasks and should be in separate groups.

Tasks:
{task_list}

Return JSON with one group per unique conceptual task. Every task index must appear \
in exactly one group.
{{
  "groups": [
    {{"tasks": ["s0.t0", "s2.t3"], "label": "brief description"}},
    ...
  ]
}}"""


def compute_redundancy(
    occupation: str,
    samples: list[list[str]],  # samples[i] = list of task strings for sample i
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """
    Pool tasks across samples, ask LLM to group by conceptual uniqueness.
    Returns: {unique_rate, redundancy_rate, n_total, n_unique}
    """
    # Build labeled task list
    entries: list[tuple[str, str]] = []  # (label, text)
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
        logging.warning(f"Redundancy: failed to parse LLM response for {occupation}")
        return None

    n_total = len(entries)
    n_unique = len(parsed["groups"])
    unique_rate = n_unique / n_total if n_total > 0 else 0.0

    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "unique_rate": round(unique_rate, 4),
        "redundancy_rate": round(1.0 - unique_rate, 4),
        "groups": parsed["groups"],
    }


# ---------------------------------------------------------------------------
# Recall + Novelty: compare one sample vs. real tasks
# ---------------------------------------------------------------------------

RECALL_NOVELTY_PROMPT = """\
You are comparing work tasks for a {occupation}.

REAL TASKS (the participant's actual tasks, labeled A0, A1, ...):
{real_tasks}

GENERATED TASKS (AI-generated candidates, labeled B0, B1, ...):
{gen_tasks}

Two tasks "match" ONLY if they describe the exact same underlying work activity \
(paraphrases or rewordings). Do NOT match tasks just because one is a subtask or \
supertask of another — "schedule 1:1s" and "manage team calendar" are different tasks \
and should NOT be considered a match.

List every matched pair (one A and one B that describe the same activity). \
A single B may match at most one A, and vice versa.

Return JSON:
{{
  "matches": [
    {{"a": 0, "b": 3}},
    {{"a": 2, "b": 7}}
  ]
}}

Return an empty list if nothing matches: {{"matches": []}}"""


def compute_recall_novelty(
    occupation: str,
    real_tasks: list[str],
    gen_tasks: list[str],
    model: str = DEFAULT_MODEL,
) -> dict | None:
    """
    Compare one generated sample vs. real tasks.
    Returns: {recall, novelty_rate, n_real_covered, n_gen_novel, n_real, n_gen}
    """
    if not real_tasks or not gen_tasks:
        return None

    real_str = "\n".join(f"A{i}: {t}" for i, t in enumerate(real_tasks))
    gen_str = "\n".join(f"B{i}: {t}" for i, t in enumerate(gen_tasks))

    prompt = RECALL_NOVELTY_PROMPT.format(
        occupation=occupation,
        real_tasks=real_str,
        gen_tasks=gen_str,
    )

    raw = _call_llm([{"role": "user", "content": prompt}], model=model)
    parsed = _parse_json(raw)
    if not parsed or "matches" not in parsed:
        logging.warning(f"Recall/novelty: failed to parse LLM response for {occupation}")
        return None

    matches = parsed["matches"]
    # Validate and deduplicate — each A and each B can appear in at most one pair
    seen_a: set[int] = set()
    seen_b: set[int] = set()
    valid_matches = []
    for m in matches:
        a, b = m.get("a"), m.get("b")
        if not (isinstance(a, int) and isinstance(b, int)):
            continue
        if not (0 <= a < len(real_tasks) and 0 <= b < len(gen_tasks)):
            continue
        if a in seen_a or b in seen_b:
            continue
        seen_a.add(a)
        seen_b.add(b)
        valid_matches.append({"a": a, "b": b})

    covered_real = len(seen_a)
    novel_generated = len(gen_tasks) - len(seen_b)

    recall = covered_real / len(real_tasks)
    novelty_rate = novel_generated / len(gen_tasks)

    return {
        "recall": round(recall, 4),
        "novelty_rate": round(novelty_rate, 4),
        "n_real": len(real_tasks),
        "n_gen": len(gen_tasks),
        "n_real_covered": covered_real,
        "n_gen_novel": novel_generated,
    }


# ---------------------------------------------------------------------------
# Per-user processing
# ---------------------------------------------------------------------------

def process_user(
    user: dict,
    samples: list[dict],
    llm_cache_dir: Path,
    model: str = DEFAULT_MODEL,
    surgery: bool = False,
) -> dict | None:
    uid = user["user_id"]
    occupation = user.get("occupation", uid)
    cache_path = llm_cache_dir / f"{uid}.json"

    if not surgery and cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
            if cached and not cached.get("error"):
                # Only use cache if all strategies are already scored
                if all(s in cached for s in STRATEGIES):
                    return cached
        except Exception:
            pass

    real_tasks = [
        t["task_statement"]
        for t in user.get("tasks", [])
        if t.get("_screen_status") != "fail"
    ]
    if not real_tasks:
        return None

    # Load any previously cached strategy results to avoid re-computing them
    prior_cache: dict = {}
    if cache_path.exists() and not surgery:
        try:
            prior_cache = json.loads(cache_path.read_text())
        except Exception:
            pass

    result: dict = {"user_id": uid, "occupation": occupation, "category": user.get("category", "")}

    for strategy in STRATEGIES:
        # Reuse cached result if already computed for this strategy
        if strategy in prior_cache and prior_cache[strategy] is not None and not surgery:
            result[strategy] = prior_cache[strategy]
            continue

        strategy_samples: list[list[str]] = []
        for sample in samples:
            tasks = sample.get(strategy) or []
            if tasks:
                strategy_samples.append(tasks)

        if not strategy_samples:
            result[strategy] = None
            continue

        # Redundancy across all samples
        redundancy = compute_redundancy(occupation, strategy_samples, model=model)

        # Recall + novelty per sample, then average
        rn_rows = []
        for gen_tasks in strategy_samples:
            rn = compute_recall_novelty(occupation, real_tasks, gen_tasks, model=model)
            if rn:
                rn_rows.append(rn)

        if rn_rows:
            avg_recall = float(np.mean([r["recall"] for r in rn_rows]))
            avg_novelty = float(np.mean([r["novelty_rate"] for r in rn_rows]))
        else:
            avg_recall = avg_novelty = None

        result[strategy] = {
            "redundancy": redundancy,
            "recall": round(avg_recall, 4) if avg_recall is not None else None,
            "novelty_rate": round(avg_novelty, 4) if avg_novelty is not None else None,
            "n_samples": len(strategy_samples),
        }

    llm_cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Load samples from cache dir
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
# Aggregation + printing
# ---------------------------------------------------------------------------

def _agg(rows: list[float | None]) -> tuple[float, float] | None:
    vals = [r for r in rows if r is not None]
    if not vals:
        return None
    return float(np.mean(vals)), float(np.std(vals))


def print_results(all_results: list[dict]) -> None:
    valid = [r for r in all_results if r]

    # Aggregate per strategy
    agg: dict[str, dict] = {}
    for strategy in STRATEGIES:
        rows_redundancy, rows_recall, rows_novelty = [], [], []
        for r in valid:
            s = r.get(strategy)
            if not s:
                continue
            red = s.get("redundancy")
            rows_redundancy.append(red["redundancy_rate"] if red else None)
            rows_recall.append(s.get("recall"))
            rows_novelty.append(s.get("novelty_rate"))
        agg[strategy] = {
            "redundancy_rate": _agg(rows_redundancy),
            "recall": _agg(rows_recall),
            "novelty_rate": _agg(rows_novelty),
        }

    col_w = 24
    header = f"{'Metric':<26}" + "".join(f"{s:>{col_w}}" for s in STRATEGIES)
    print(header)
    print("-" * (26 + col_w * len(STRATEGIES)))

    for key, label, arrow in [
        ("redundancy_rate", "Redundancy rate↓", ""),
        ("recall", "Recall↑", ""),
        ("novelty_rate", "Novelty rate↑", ""),
    ]:
        row_str = f"{label:<26}"
        for strategy in STRATEGIES:
            val = agg[strategy][key]
            if val:
                row_str += f"  {val[0]:.3f} ± {val[1]:.3f}".rjust(col_w)
            else:
                row_str += f"{'—':>{col_w}}"
        print(row_str)

    # Per-user breakdown
    sorted_results = sorted(valid, key=lambda r: r.get("category", ""))
    occ_w = 44
    blk_w = 3 * 8

    print(f"\n\n{'─'*120}")
    print("PER-USER RESULTS")
    print(f"{'─'*120}")
    hdr = f"{'Occupation':<{occ_w}}  "
    hdr += "  ".join(f"{s:^{blk_w}}" for s in STRATEGIES)
    print(hdr)

    sub_hdrs = ["redund", "recall", "novel"]
    hdr2 = f"{'':>{occ_w}}  "
    hdr2 += "  ".join("  ".join(f"{h:>6}" for h in sub_hdrs) for _ in STRATEGIES)
    print(hdr2)
    print(f"{'─'*120}")

    prev_cat = None
    for r in sorted_results:
        if r.get("category") != prev_cat:
            print(f"\n── {r.get('category', '')} ──")
            prev_cat = r.get("category")

        occ = r.get("occupation", r["user_id"])[:occ_w - 1]
        line = f"{occ:<{occ_w}}  "
        blocks = []
        for strategy in STRATEGIES:
            s = r.get(strategy)
            if not s:
                blocks.append("  ".join(f"{'—':>6}" for _ in sub_hdrs))
                continue
            red = s.get("redundancy")
            redund_val = f"{red['redundancy_rate']:.3f}" if red else "  —  "
            recall_val = f"{s['recall']:.3f}" if s.get("recall") is not None else "  —  "
            novel_val = f"{s['novelty_rate']:.3f}" if s.get("novelty_rate") is not None else "  —  "
            blocks.append(f"{redund_val:>6}  {recall_val:>6}  {novel_val:>6}")
        line += "  ".join(blocks)
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="LLM-judged task diversity: redundancy, recall, novelty")
    parser.add_argument("--input", required=True, help="screened_study_tasks.json")
    parser.add_argument("--cache-dir", required=True, help="Dir with per-user generation cache (sim_cache)")
    parser.add_argument("--llm-cache-dir", default=None, help="Dir to store LLM judgment results")
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=6)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--surgery", action="store_true", help="Re-run even if cached results exist")
    args = parser.parse_args()

    data = json.loads(Path(args.input).read_text())
    cache_dir = Path(args.cache_dir)
    llm_cache_dir = Path(args.llm_cache_dir) if args.llm_cache_dir else cache_dir.parent / "llm_diversity_cache"

    users_with_samples = []
    for user in data:
        uid = user["user_id"]
        samples = load_samples(uid, cache_dir, args.n_samples)
        if samples:
            users_with_samples.append((user, samples))

    if args.num_users:
        users_with_samples = users_with_samples[: args.num_users]

    n_samples_found = max((len(s) for _, s in users_with_samples), default=0)
    print(f"{len(users_with_samples)} users, up to {n_samples_found} sample(s) each\n")

    all_results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_user, user, samples, llm_cache_dir, args.model, args.surgery): user["user_id"]
            for user, samples in users_with_samples
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            uid = futures[future]
            try:
                r = future.result()
                if r:
                    all_results.append(r)
            except Exception as e:
                logging.error(f"Error for user {uid}: {e}")

    print_results(all_results)


if __name__ == "__main__":
    main()
