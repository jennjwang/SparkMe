"""
Order-sensitivity test for the online clustering pipeline.

Runs the pipeline N times with randomly shuffled interview orderings
(tasks within each interview stay in their original order) and saves each
run's cluster state plus a cross-run comparison summary.

Usage
-----
# 10 random orderings, reproducible seed
python analysis/task_clustering/order_sensitivity/run.py --n 10 --seed 42

# All 720 permutations (expensive — ~720 × 30 LLM calls)
python analysis/task_clustering/order_sensitivity/run.py --all

# Resume: skip runs whose output already exists
python analysis/task_clustering/order_sensitivity/run.py --n 20 --seed 42 --resume

Output
------
order_sensitivity/results/
  run_000/
    order.json       — interview order for this run
    clusters.json    — raw pipeline state
    summary.json     — per-cluster label + members + orphan list
  ...
  comparison.json    — cross-run statistics
"""

from __future__ import annotations
import argparse
import itertools
import json
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from analysis.task_clustering.models import TaskItem
from analysis.task_clustering.pipeline import OnlineClusteringPipeline
from analysis.task_clustering.cli import _load_config, _load_criteria


# ── paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent.parent.parent
CONFIG_PATH = ROOT / "analysis/task_clustering/config_cirs.json"
INPUT_PATH  = ROOT / "analysis/task_clustering/input_cirs.json"
OUT_DIR     = Path(__file__).parent / "results"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_interviews() -> list[dict]:
    """Return the CIRS input as a list of interview dicts, each with 'occupation' and 'tasks'."""
    with open(INPUT_PATH) as f:
        return json.load(f)


def interviews_to_task_items(interviews: list[dict]) -> list[TaskItem]:
    """Flatten interviews (in given order) into a list of TaskItems."""
    now = datetime.now()
    items = []
    for interview in interviews:
        source = interview.get("occupation", "")
        for task in interview.get("tasks", []):
            text = task.get("task_statement", "")
            if text:
                items.append(TaskItem(
                    id=str(uuid.uuid4()),
                    text=text,
                    source=source,
                    timestamp=now,
                    metadata={k: v for k, v in task.items() if k != "task_statement"},
                ))
    return items


def run_pipeline(items: list[TaskItem], cfg: dict, criteria: list[str] | None) -> OnlineClusteringPipeline:
    algo = cfg.get("algorithm", {})
    pipe = OnlineClusteringPipeline(
        lambda_=algo.get("lambda", 0.1),
        eps=algo.get("eps", 0.5),
        max_split_size=algo.get("max_split_size", 5),
        split_depth_limit=algo.get("split_depth_limit", 1),
        fade_interval_interviews=algo.get("fade_interval_interviews", 10),
        criteria=criteria,
        model=algo.get("model", "gpt-4.1"),
    )
    if cfg.get("taxonomy"):
        pipe.load_taxonomy(ROOT / cfg["taxonomy"])

    pipe.ingest_batch(items, verbose=False)
    pipe.force_fade()
    pipe.force_split_check()
    return pipe


def extract_summary(pipe: OnlineClusteringPipeline, items: list[TaskItem]) -> dict:
    """Extract a human-readable summary from the final pipeline state."""
    state = pipe.state

    # Collect assigned task texts per cluster
    clusters = []
    assigned_ids: set[str] = set()
    for c in sorted(state.clusters.values(), key=lambda x: -x.weight):
        if not c.is_leaf:
            continue
        member_texts = [state.items[mid].text for mid in c.members if mid in state.items]
        member_sources = [state.items[mid].source for mid in c.members if mid in state.items]
        assigned_ids.update(c.members)
        clusters.append({
            "label": c.leader,
            "level": c.level,
            "parent_id": c.parent_id,
            "weight": round(c.weight, 3),
            "members": [{"text": t, "source": s} for t, s in zip(member_texts, member_sources)],
        })

    # Orphans: tasks that were never assigned to a cluster (stayed in buffer)
    orphan_texts = [
        item.text for item in state.items.values()
        if item.id not in assigned_ids
    ]

    # Also include internal (non-leaf) cluster labels for hierarchy context
    parents = []
    for c in state.clusters.values():
        if not c.is_leaf and c.children:
            parents.append({
                "label": c.leader,
                "children": [
                    state.clusters[cid].leader
                    for cid in c.children
                    if cid in state.clusters
                ],
            })

    return {
        "n_clusters": len(clusters),
        "n_orphans": len(orphan_texts),
        "clusters": clusters,
        "parent_clusters": parents,
        "orphans": orphan_texts,
    }


def save_run(run_dir: Path, order: list[str], summary: dict, pipe: OnlineClusteringPipeline) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "order.json", "w") as f:
        json.dump({"interview_order": order}, f, indent=2)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    pipe.save(run_dir / "clusters.json")


def build_comparison(run_summaries: list[tuple[list[str], dict]]) -> dict:
    """
    Compute cross-run statistics:
    - cluster count distribution
    - orphan count distribution
    - task co-occurrence: for every (task_a, task_b) pair, how often they end up
      in the same leaf cluster across all runs
    - label inventory: all unique cluster labels seen
    """
    n_runs = len(run_summaries)
    cluster_counts = [s["n_clusters"] for _, s in run_summaries]
    orphan_counts  = [s["n_orphans"]  for _, s in run_summaries]

    # Collect all task texts
    all_texts: list[str] = []
    seen: set[str] = set()
    for _, s in run_summaries:
        for c in s["clusters"]:
            for m in c["members"]:
                t = m["text"]
                if t not in seen:
                    seen.add(t)
                    all_texts.append(t)
        for t in s["orphans"]:
            if t not in seen:
                seen.add(t)
                all_texts.append(t)

    text_idx = {t: i for i, t in enumerate(all_texts)}
    N = len(all_texts)

    # co_count[i][j] = number of runs where task i and task j are in the same cluster
    co_count: list[list[int]] = [[0] * N for _ in range(N)]

    all_labels: list[str] = []

    for _, s in run_summaries:
        for c in s["clusters"]:
            texts_in_cluster = [m["text"] for m in c["members"]]
            all_labels.append(c["label"])
            idxs = [text_idx[t] for t in texts_in_cluster if t in text_idx]
            for a in idxs:
                for b in idxs:
                    if a != b:
                        co_count[a][b] += 1

    # Stability: for each pair that co-occurs at least once, what fraction of runs?
    stable_pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            if co_count[i][j] > 0:
                freq = co_count[i][j] / n_runs
                stable_pairs.append({
                    "task_a": all_texts[i],
                    "task_b": all_texts[j],
                    "co_occurrences": co_count[i][j],
                    "stability": round(freq, 3),
                })
    stable_pairs.sort(key=lambda x: -x["stability"])

    # Orphan frequency: how often is each task an orphan?
    orphan_count: dict[str, int] = {t: 0 for t in all_texts}
    for _, s in run_summaries:
        for t in s["orphans"]:
            if t in orphan_count:
                orphan_count[t] += 1
    orphan_freq = [
        {"task": t, "orphan_runs": c, "orphan_rate": round(c / n_runs, 3)}
        for t, c in sorted(orphan_count.items(), key=lambda x: -x[1])
        if c > 0
    ]

    unique_labels = sorted(set(all_labels))

    return {
        "n_runs": n_runs,
        "cluster_count": {
            "min": min(cluster_counts),
            "max": max(cluster_counts),
            "mean": round(sum(cluster_counts) / n_runs, 2),
            "per_run": cluster_counts,
        },
        "orphan_count": {
            "min": min(orphan_counts),
            "max": max(orphan_counts),
            "mean": round(sum(orphan_counts) / n_runs, 2),
            "per_run": orphan_counts,
        },
        "task_co_occurrence": stable_pairs,
        "orphan_frequency": orphan_freq,
        "all_cluster_labels": unique_labels,
        "run_orders": [order for order, _ in run_summaries],
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Order-sensitivity test for task clustering")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n",   type=int, help="Number of random orderings to test")
    group.add_argument("--all", action="store_true", help="Run all 6! = 720 permutations")
    parser.add_argument("--seed",   type=int, default=0,    help="Random seed (default: 0)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip runs whose output directory already exists")
    args = parser.parse_args()

    cfg = _load_config(CONFIG_PATH)
    criteria = _load_criteria(ROOT / cfg["criteria"]) if cfg.get("criteria") else None
    interviews = load_interviews()
    interview_names = [d["occupation"] for d in interviews]

    # Build the list of orderings to test
    if args.all:
        orderings = list(itertools.permutations(range(len(interviews))))
    else:
        rng = random.Random(args.seed)
        indices = list(range(len(interviews)))
        orderings_set: set[tuple] = set()
        # Always include the original order first
        orderings_set.add(tuple(indices))
        while len(orderings_set) < args.n:
            perm = list(indices)
            rng.shuffle(perm)
            orderings_set.add(tuple(perm))
        orderings = [list(p) for p in sorted(orderings_set)]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(orderings)} orderings → {OUT_DIR}\n")

    run_summaries: list[tuple[list[str], dict]] = []

    for run_idx, order in enumerate(orderings):
        run_dir = OUT_DIR / f"run_{run_idx:03d}"
        order_names = [interview_names[i] for i in order]

        if args.resume and (run_dir / "summary.json").exists():
            print(f"[{run_idx:03d}] skipping (already done) — {order_names}")
            with open(run_dir / "summary.json") as f:
                summary = json.load(f)
            run_summaries.append((order_names, summary))
            continue

        print(f"[{run_idx:03d}] order: {order_names}")
        ordered_interviews = [interviews[i] for i in order]
        items = interviews_to_task_items(ordered_interviews)

        pipe = run_pipeline(items, cfg, criteria)
        summary = extract_summary(pipe, items)
        save_run(run_dir, order_names, summary, pipe)

        n_c = summary["n_clusters"]
        n_o = summary["n_orphans"]
        labels = [c["label"][:50] for c in summary["clusters"]]
        print(f"        → {n_c} clusters, {n_o} orphans")
        for lbl in labels:
            print(f"           · {lbl}")
        print()

        run_summaries.append((order_names, summary))

    # Cross-run comparison
    comparison = build_comparison(run_summaries)
    comp_path = OUT_DIR / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved → {comp_path}")

    # Print a quick human-readable summary
    cc = comparison["cluster_count"]
    oc = comparison["orphan_count"]
    print(f"\n{'='*60}")
    print(f"Cluster count across {len(orderings)} runs:  "
          f"min={cc['min']}  max={cc['max']}  mean={cc['mean']}")
    print(f"Orphan count across {len(orderings)} runs:   "
          f"min={oc['min']}  max={oc['max']}  mean={oc['mean']}")

    print(f"\nMost stable task pairs (always in same cluster):")
    for p in comparison["task_co_occurrence"][:5]:
        print(f"  {p['stability']:.0%} — {p['task_a'][:55]!r}")
        print(f"          + {p['task_b'][:55]!r}")

    print(f"\nMost frequent orphans:")
    for o in comparison["orphan_frequency"][:5]:
        print(f"  {o['orphan_rate']:.0%} orphan rate — {o['task'][:70]!r}")

    print(f"\nAll cluster labels seen across runs ({len(comparison['all_cluster_labels'])}):")
    for lbl in comparison["all_cluster_labels"]:
        print(f"  · {lbl}")


if __name__ == "__main__":
    main()
