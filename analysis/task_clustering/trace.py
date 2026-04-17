"""
Trace script: shows how the cluster tree evolves as tasks are ingested one by one.
Automatically saves a trace JSON (used by the visualizer's timeline slider).

Usage:
    python analysis/task_clustering/trace.py --config analysis/task_clustering/config_cirs.json
    python analysis/task_clustering/trace.py --config config_cirs.json --limit 10
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.task_clustering.models import TaskItem
from analysis.task_clustering.pipeline import OnlineClusteringPipeline
from analysis.task_clustering.cli import _load_config, _load_items, _load_criteria
from analysis.task_clustering import leader as leader_mod


def _print_tree(pipe: OnlineClusteringPipeline, indent: str = "  ") -> None:
    """Print the current cluster tree."""
    state = pipe.state
    roots = [c for c in state.clusters.values() if c.parent_id is None]
    roots.sort(key=lambda c: -c.weight)
    if not roots:
        print(f"{indent}(empty)")
        return
    for root in roots:
        direct = len(root.members)
        child_total = sum(
            len(state.clusters[cid].members)
            for cid in root.children
            if cid in state.clusters
        )
        print(f"{indent}[L0] {root.leader[:60]!r}  "
              f"(direct={direct}, via children={child_total}, w={root.weight:.1f})")
        for cid in root.children:
            if cid not in state.clusters:
                continue
            child = state.clusters[cid]
            print(f"{indent}      [L1] {child.leader[:55]!r}  "
                  f"(members={len(child.members)}, w={child.weight:.1f})")


def _snapshot(pipe: OnlineClusteringPipeline, step: int,
              task: TaskItem, assigned_id: str) -> dict:
    """Capture a lightweight snapshot of the current pipeline state."""
    state = pipe.state
    assigned = state.clusters.get(assigned_id)

    clusters_snap = []
    for c in state.clusters.values():
        # Count members per source
        sources: dict[str, int] = {}
        for mid in c.members:
            item = state.items.get(mid)
            if item and item.source:
                sources[item.source] = sources.get(item.source, 0) + 1
        members = [
            {"text": state.items[mid].text, "source": state.items[mid].source or ""}
            for mid in c.members
            if mid in state.items
        ]
        clusters_snap.append({
            "id": c.id,
            "leader": c.leader,
            "level": c.level,
            "parent_id": c.parent_id,
            "children": list(c.children),
            "member_count": len(c.members),
            "weight": round(c.weight, 3),
            "sources": sources,
            "anchored": c.anchored,
            "members": members,
        })

    return {
        "step": step,
        "task_text": task.text,
        "task_source": task.source or "",
        "assigned_id": assigned_id,
        "assigned_label": assigned.leader if assigned else "",
        "assigned_level": assigned.level if assigned else 0,
        "clusters": clusters_snap,
    }


def main():
    parser = argparse.ArgumentParser(description="Trace cluster tree evolution item by item")
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after this many tasks (default: all)")
    args = parser.parse_args()

    cfg = _load_config(args.config)
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    algo_cfg = cfg.get("algorithm", {})

    criteria = _load_criteria(cfg["criteria"]) if cfg.get("criteria") else None
    items = _load_items(
        input_path=Path(input_cfg["path"]),
        text_field=input_cfg.get("text_field", "text"),
        source_field=input_cfg.get("source_field"),
        timestamp_field=input_cfg.get("timestamp_field"),
        items_field=input_cfg.get("items_field"),
    )

    # Derive trace output path from config output path
    state_path = Path(output_cfg.get("state", "analysis/task_clustering/output/clusters.json"))
    trace_path = state_path.parent / (state_path.stem + "_trace.json")

    pipe = OnlineClusteringPipeline(
        lambda_=algo_cfg.get("lambda", 0.1),
        eps=algo_cfg.get("eps", 0.5),
        max_split_size=algo_cfg.get("max_split_size", 5),
        split_depth_limit=algo_cfg.get("split_depth_limit", 1),
        fade_interval_interviews=999999,  # no fading during trace
        criteria=criteria,
        model=algo_cfg.get("model", "gpt-4.1"),
    )

    if args.limit:
        items = items[:args.limit]

    if cfg.get("taxonomy"):
        pipe.load_taxonomy(cfg["taxonomy"])
        print("=== Initial taxonomy ===")
        _print_tree(pipe)
        print()

    print(f"Ingesting {len(items)} tasks...\n")
    print("=" * 60)

    snapshots = []
    for i, task in enumerate(items):
        cluster_id, reasonings = leader_mod.process_task(
            task, pipe.state, pipe.llm, model=pipe.model, return_reasoning=True
        )
        pipe.state.total_processed += 1
        # skip fading (fade_interval set to 999999)
        # incremental split check
        from analysis.task_clustering import divisive
        divisive.try_split(cluster_id, pipe.state, pipe.llm, model=pipe.model)

        assigned = pipe.state.clusters.get(cluster_id)
        label = assigned.leader[:55] if assigned else "?"
        level = assigned.level if assigned else "?"

        print(f"\n[{i+1:02d}] {task.text[:70]!r}")
        for r in reasonings:
            print(f"     {r}")
        print(f"     → L{level}: {label!r}")
        _print_tree(pipe)

        snap = _snapshot(pipe, i + 1, task, cluster_id)
        snap["reasonings"] = reasonings
        snapshots.append(snap)

    print("\n" + "=" * 60)
    print("Final tree:")
    _print_tree(pipe)
    s = pipe.summary()
    print(f"\n{s['total_clusters']} clusters ({s['leaf_clusters']} leaves, "
          f"{s['internal_clusters']} internal), {s['total_items']} items")

    # Save trace
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_path, "w") as f:
        json.dump(snapshots, f, indent=2)
    print(f"\nTrace saved to {trace_path}")

    # Auto-generate both visualizers
    from analysis.task_clustering.visualize import generate_trace_html, build_data, generate_html

    viz_trace_path = trace_path.parent / (state_path.stem + "_viz_trace.html")
    viz_trace_path.write_text(generate_trace_html(snapshots), encoding="utf-8")
    print(f"Trace visualizer updated: {viz_trace_path}")

    # Also regenerate the static viz from the final pipeline state
    pipe.save(state_path)
    with open(state_path) as f:
        import json as _json
        saved_state = _json.load(f)
    viz_static_path = trace_path.parent / "viz.html"
    data = build_data(saved_state)
    viz_static_path.write_text(generate_html(data), encoding="utf-8")
    print(f"Static visualizer updated: {viz_static_path}")


if __name__ == "__main__":
    main()
