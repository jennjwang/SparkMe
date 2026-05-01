"""
Compare task generator strategies against real user tasks.

Metrics computed per user, then aggregated across users:

  Semantic recall        — fraction of real tasks "covered" by at least one
                           generated task (cosine sim ≥ threshold).
                           Reported at three thresholds: 0.60, 0.70, 0.80.

  Soft recall            — mean of max-cosine-similarity of each real task to
                           the generated set (no threshold needed).

  Per-user mean pairwise — mean cosine distance within the generated set for
                           each user (fixes global inflation from cross-user
                           distances).

  Vendi Score            — exp(entropy of eigenvalues of the normalised
                           similarity matrix). Interpretable as the "effective
                           number of distinct tasks". Penalises near-duplicates
                           more smoothly than mean pairwise.

  Mean nearest-neighbor distance
                         — for each generated task, distance to its closest
                           generated sibling, averaged across the set.  This
                           emphasises local crowding instead of global spread.

  Near-duplicate rate    — fraction of task pairs with cosine sim ≥ 0.85.
                           A direct redundancy measure.

Usage:
    python evaluation/eval_strategy_comparison.py \
        --input   analysis/task_clustering/output/screened_study_tasks.json \
        --cache-dir analysis/task_clustering/output/sim_cache
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from openai import OpenAI

OPENAI_CLIENT = OpenAI()
EMBED_MODEL   = "text-embedding-3-large"
STRATEGIES    = ["scratch", "simple", "post_hoc", "combined", "combined_no_gap"]
THRESHOLDS    = [0.60, 0.70, 0.80]


# ---------------------------------------------------------------------------
# Embedding
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def semantic_recall(real_embs: np.ndarray, gen_embs: np.ndarray, threshold: float) -> float:
    """Fraction of real tasks with max cosine similarity to generated set >= threshold."""
    if len(real_embs) == 0 or len(gen_embs) == 0:
        return 0.0
    sim = real_embs @ gen_embs.T          # (n_real, n_gen)
    return float((sim.max(axis=1) >= threshold).mean())


def soft_recall(real_embs: np.ndarray, gen_embs: np.ndarray) -> float:
    """Mean of per-real-task maximum similarity to generated set (no threshold)."""
    if len(real_embs) == 0 or len(gen_embs) == 0:
        return 0.0
    sim = real_embs @ gen_embs.T
    return float(sim.max(axis=1).mean())


def mean_pairwise(embs: np.ndarray) -> float:
    """Mean cosine distance across all pairs within the set."""
    n = len(embs)
    if n < 2:
        return 0.0
    sim  = embs @ embs.T
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    mask = ~np.eye(n, dtype=bool)
    return float(dist[mask].mean())


def mean_nearest_neighbor_distance(embs: np.ndarray) -> float:
    """Mean distance from each task to its closest other task."""
    n = len(embs)
    if n < 2:
        return 0.0
    sim = embs @ embs.T
    np.fill_diagonal(sim, -np.inf)
    nearest_sim = sim.max(axis=1)
    nearest_dist = np.clip(1.0 - nearest_sim, 0.0, 2.0)
    return float(nearest_dist.mean())


def vendi_score(embs: np.ndarray) -> float:
    """
    Vendi Score = exp(H), where H is the entropy of the eigenvalues of
    the similarity matrix normalised by n.  Equals the "effective number
    of distinct tasks"; penalises near-duplicates continuously.
    Reference: Friedman & Dieng, 2022 (arXiv:2210.02410).
    """
    n = len(embs)
    if n < 2:
        return 1.0
    sim = (embs @ embs.T) / n
    eigvals = np.linalg.eigvalsh(sim)
    eigvals = eigvals[eigvals > 1e-10]
    H = float(-np.sum(eigvals * np.log(eigvals)))
    return float(np.exp(H))


def near_duplicate_rate(embs: np.ndarray, threshold: float = 0.85) -> float:
    """Fraction of off-diagonal pairs with cosine similarity >= threshold."""
    n = len(embs)
    if n < 2:
        return 0.0
    sim  = embs @ embs.T
    mask = ~np.eye(n, dtype=bool)
    return float((sim[mask] >= threshold).mean())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",     required=True, help="screened_study_tasks.json")
    parser.add_argument("--cache-dir", default=None,  help="Dir with per-user generation cache")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Number of samples to load (auto-detects if omitted)")
    args = parser.parse_args()

    data      = json.loads(Path(args.input).read_text())
    cache_dir = Path(args.cache_dir) if args.cache_dir else Path(args.input).parent / "sim_cache"

    # Auto-detect how many samples exist per user
    def _load_samples(uid: str) -> list[dict]:
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
            if args.n_samples is not None and si >= args.n_samples:
                continue
            sample_paths[si] = p

        samples = []
        for si in sorted(sample_paths):
            try:
                samples.append(json.loads(sample_paths[si].read_text()))
            except Exception:
                pass
        return samples

    # Collect real texts + all samples per user
    user_real_texts:  dict[str, list[str]] = {}
    user_samples:     dict[str, list[dict]] = {}   # uid → [sample_dict, ...]

    for user in data:
        uid = user["user_id"]
        user_real_texts[uid] = [
            t["task_statement"] for t in user["tasks"]
            if t.get("_screen_status") != "fail"
        ]
        samples = _load_samples(uid)
        if samples:
            user_samples[uid] = samples

    users_with_data = [u for u in data if u["user_id"] in user_samples]
    n_samples_found = max((len(v) for v in user_samples.values()), default=1)
    print(f"{len(users_with_data)} users, up to {n_samples_found} sample(s) each\n")

    # Build flat list: real tasks + all (user, sample, strategy) combos
    all_texts: list[str] = []
    # slices[uid]["real"] = (s, e)
    # slices[uid][strategy][sample_idx] = (s, e)
    slices: dict[str, dict] = {}

    for user in users_with_data:
        uid = user["user_id"]
        slices[uid] = {s: {} for s in STRATEGIES}

        s = len(all_texts)
        all_texts.extend(user_real_texts[uid])
        slices[uid]["real"] = (s, len(all_texts))

        for si, sample in enumerate(user_samples[uid]):
            for strategy in STRATEGIES:
                s = len(all_texts)
                all_texts.extend(sample.get(strategy) or [])
                slices[uid][strategy][si] = (s, len(all_texts))

    print(f"Embedding {len(all_texts)} tasks jointly…")
    embeddings = embed_texts(all_texts)

    # Per-user metrics — average across samples first, then across users
    results: dict[str, list[dict]] = {s: [] for s in STRATEGIES}

    for user in users_with_data:
        uid = user["user_id"]
        rs, real_end = slices[uid]["real"]
        real_embs = embeddings[rs:real_end]
        if len(real_embs) == 0:
            continue

        for strategy in STRATEGIES:
            sample_rows = []
            for si, (gs, ge) in slices[uid][strategy].items():
                gen_embs = embeddings[gs:ge]
                if len(gen_embs) == 0:
                    continue
                row = {
                    "soft_recall":   soft_recall(real_embs, gen_embs),
                    "mean_pairwise": mean_pairwise(gen_embs),
                    "mean_nn_dist":  mean_nearest_neighbor_distance(gen_embs),
                    "vendi":         vendi_score(gen_embs),
                    "near_dup_rate": near_duplicate_rate(gen_embs),
                }
                for t in THRESHOLDS:
                    row[f"recall@{t:.2f}"] = semantic_recall(real_embs, gen_embs, t)
                sample_rows.append(row)

            if not sample_rows:
                continue
            # Average across samples for this user
            averaged = {k: float(np.mean([r[k] for r in sample_rows])) for k in sample_rows[0]}
            results[strategy].append(averaged)

    # Aggregate and print
    def _agg(rows: list[dict], key: str) -> tuple[float, float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        return (np.mean(vals), np.std(vals)) if vals else (0.0, 0.0)

    metric_keys = (
        [f"recall@{t:.2f}" for t in THRESHOLDS]
        + ["soft_recall", "mean_pairwise", "mean_nn_dist", "vendi", "near_dup_rate"]
    )
    metric_labels = (
        [f"Recall@{t:.2f}" for t in THRESHOLDS]
        + ["Soft recall", "Mean pairwise↑", "Mean NN dist↑", "Vendi Score↑", "Near-dup rate↓"]
    )

    col_w = 22
    header = f"{'Metric':<22}" + "".join(f"{s:>{col_w}}" for s in STRATEGIES)
    print(header)
    print("-" * (22 + col_w * len(STRATEGIES)))

    for key, label in zip(metric_keys, metric_labels):
        row_str = f"{label:<22}"
        for strategy in STRATEGIES:
            mean, std = _agg(results[strategy], key)
            row_str += f"  {mean:.3f} ± {std:.3f}".rjust(col_w)
        print(row_str)

    # Also print Vendi Score for real tasks as baseline
    real_vendis, real_mpw, real_nn = [], [], []
    for user in users_with_data:
        uid = user["user_id"]
        rs, real_end = slices[uid]["real"]
        real_embs = embeddings[rs:real_end]
        if len(real_embs) >= 2:
            real_vendis.append(vendi_score(real_embs))
            real_mpw.append(mean_pairwise(real_embs))
            real_nn.append(mean_nearest_neighbor_distance(real_embs))

    print(f"\nReal tasks baseline (n={len(real_vendis)} users):")
    print(f"  Mean pairwise : {np.mean(real_mpw):.3f} ± {np.std(real_mpw):.3f}")
    print(f"  Mean NN dist  : {np.mean(real_nn):.3f} ± {np.std(real_nn):.3f}")
    print(f"  Vendi Score   : {np.mean(real_vendis):.3f} ± {np.std(real_vendis):.3f}")

    # ── Per-user full results ────────────────────────────────────────────────
    uid_to_occ = {u["user_id"]: u["occupation"] for u in data}
    uid_to_cat = {u["user_id"]: u.get("category", "") for u in data}

    per_user_rows = []
    for user in users_with_data:
        uid  = user["user_id"]
        rs, real_end = slices[uid]["real"]
        real_embs = embeddings[rs:real_end]
        if len(real_embs) == 0:
            continue
        row = {
            "occupation": uid_to_occ.get(uid, uid),
            "category":   uid_to_cat.get(uid, ""),
            "n_real":     len(real_embs),
            "real_mpw":   mean_pairwise(real_embs) if len(real_embs) >= 2 else None,
            "real_vendi": vendi_score(real_embs)   if len(real_embs) >= 2 else None,
        }
        strategy_prefixes = {
            "scratch": "scra",
            "simple": "simp",
            "post_hoc": "post",
            "combined": "comb",
            "combined_no_gap": "nogap",
        }
        for strategy in STRATEGIES:
            pfx = strategy_prefixes[strategy]
            sample_rows = []
            for gs, ge in slices[uid][strategy].values():
                gen_embs = embeddings[gs:ge]
                if len(gen_embs) == 0:
                    continue
                sample_rows.append({
                    "sr":  soft_recall(real_embs, gen_embs),
                    "mpw": mean_pairwise(gen_embs),
                    "vs":  vendi_score(gen_embs),
                    "r60": semantic_recall(real_embs, gen_embs, 0.60),
                    "r70": semantic_recall(real_embs, gen_embs, 0.70),
                })
            if not sample_rows:
                row[f"{pfx}_sr"]  = None
                row[f"{pfx}_mpw"] = None
                row[f"{pfx}_vs"]  = None
                row[f"{pfx}_r60"] = None
                row[f"{pfx}_r70"] = None
            else:
                row[f"{pfx}_sr"]  = round(float(np.mean([r["sr"]  for r in sample_rows])), 3)
                row[f"{pfx}_mpw"] = round(float(np.mean([r["mpw"] for r in sample_rows])), 3)
                row[f"{pfx}_vs"]  = round(float(np.mean([r["vs"]  for r in sample_rows])), 2)
                row[f"{pfx}_r60"] = round(float(np.mean([r["r60"] for r in sample_rows])), 2)
                row[f"{pfx}_r70"] = round(float(np.mean([r["r70"] for r in sample_rows])), 2)
        per_user_rows.append(row)

    per_user_rows.sort(key=lambda r: r["category"])

    strats_short = {
        "scratch": "scra",
        "simple": "simp",
        "post_hoc": "post",
        "combined": "comb",
        "combined_no_gap": "nogap",
    }
    sub_cols = ["sr", "mpw", "vs", "r60", "r70"]
    sub_hdrs = ["soft_r", "mpw", "vendi", "r@.60", "r@.70"]

    # Header
    occ_w = 42
    real_w = 22
    blk_w  = len(sub_cols) * 7 + 1

    print(f"\n\n{'─'*120}")
    print("PER-USER RESULTS")
    print(f"{'─'*120}")
    hdr1 = f"{'Occupation':<{occ_w}}  {'real_mpw':>8}  {'real_vs':>7}  "
    hdr1 += "  ".join(f"{s:^{blk_w}}" for s in STRATEGIES)
    print(hdr1)

    hdr2 = f"{'':>{occ_w}}  {'':>8}  {'':>7}  "
    hdr2 += "  ".join("  ".join(f"{h:>6}" for h in sub_hdrs) for _ in STRATEGIES)
    print(hdr2)
    print(f"{'─'*120}")

    prev_cat = None
    for row in per_user_rows:
        if row["category"] != prev_cat:
            print(f"\n── {row['category']} ──")
            prev_cat = row["category"]

        occ   = row["occupation"][:occ_w - 1]
        rmpw  = f"{row['real_mpw']:.3f}" if row["real_mpw"] is not None else "  —  "
        rvs   = f"{row['real_vendi']:.2f}" if row["real_vendi"] is not None else "  —  "
        line  = f"{occ:<{occ_w}}  {rmpw:>8}  {rvs:>7}  "

        blocks = []
        for strategy in STRATEGIES:
            pfx = strats_short[strategy]
            vals = [row.get(f"{pfx}_{c}") for c in sub_cols]
            blocks.append("  ".join(
                f"{v:>6.3f}" if isinstance(v, float) else f"{'—':>6}"
                for v in vals
            ))
        line += "  ".join(blocks)
        print(line)


if __name__ == "__main__":
    main()
