"""
CLI entry point for the general online task clustering pipeline.

Reads a config file (JSON) and an input data file, runs the pipeline,
and writes the cluster hierarchy + state to disk.

Usage examples
--------------
# Run with a config file
python analysis/task_clustering/cli.py --config analysis/task_clustering/config.json

# Override a specific parameter
python analysis/task_clustering/cli.py --config config.json --no-splits

# Resume from a previous run
python analysis/task_clustering/cli.py --config config.json --resume output/clusters.json
"""

from __future__ import annotations
import argparse
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from analysis.task_clustering.models import TaskItem
from analysis.task_clustering.pipeline import OnlineClusteringPipeline


def _load_config(path: str | Path) -> dict:
    """Load a JSON config file and return its contents as a dict."""
    with open(path) as f:
        return json.load(f)


def _load_items(
    input_path: Path,
    text_field: str,
    source_field: str | None,
    timestamp_field: str | None,
    items_field: str | None,
) -> list[TaskItem]:
    """
    Load TaskItems from a JSON file.

    Handles two layouts:
    1. Flat list: [{text_field: "...", ...}, ...]
    2. Nested:    [{items_field: [{text_field: "...", ...}], source_field: "...", ...}, ...]
    """
    with open(input_path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"[error] Input JSON must be a list, got {type(data).__name__}", file=sys.stderr)
        sys.exit(1)

    items: list[TaskItem] = []
    now = datetime.now()

    if items_field:
        for outer in data:
            source = str(outer.get(source_field, "")) if source_field else ""
            inner_list = outer.get(items_field, [])
            if not isinstance(inner_list, list):
                inner_list = [inner_list]
            for record in inner_list:
                text = record.get(text_field, "")
                if not text:
                    continue
                ts_raw = record.get(timestamp_field) if timestamp_field else None
                ts = _parse_ts(ts_raw) or now
                items.append(TaskItem(
                    id=str(uuid.uuid4()),
                    text=str(text),
                    source=source,
                    timestamp=ts,
                    metadata={k: v for k, v in record.items() if k != text_field},
                ))
    else:
        for record in data:
            text = record.get(text_field, "")
            if not text:
                continue
            source = str(record.get(source_field, "")) if source_field else ""
            ts_raw = record.get(timestamp_field) if timestamp_field else None
            ts = _parse_ts(ts_raw) or now
            items.append(TaskItem(
                id=str(uuid.uuid4()),
                text=str(text),
                source=source,
                timestamp=ts,
                metadata={k: v for k, v in record.items() if k != text_field},
            ))

    return items


def _load_criteria(path: str | Path) -> list[str] | None:
    """Load similarity criteria from a text file (one rule per line, # comments ignored)."""
    lines = []
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                lines.append(stripped)
    if not lines:
        print(f"[warning] Criteria file {path} is empty, using defaults", file=sys.stderr)
        return None
    return lines


def _parse_ts(value) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except Exception:
            return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Online task clustering: Leader + DenStream + Incremental Divisive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True,
                        help="Path to JSON config file (see config.json for format)")
    parser.add_argument("--resume", default=None,
                        help="Path to a previously saved state file to resume from")
    parser.add_argument("--no-splits", action="store_true",
                        help="Override: disable divisive splitting")

    args = parser.parse_args()

    # ---- Load config ----------------------------------------------------
    cfg = _load_config(args.config)
    input_cfg = cfg.get("input", {})
    output_cfg = cfg.get("output", {})
    algo_cfg = cfg.get("algorithm", {})

    input_path = Path(input_cfg["path"])
    text_field = input_cfg.get("text_field", "text")
    source_field = input_cfg.get("source_field")
    timestamp_field = input_cfg.get("timestamp_field")
    items_field = input_cfg.get("items_field")

    output_path = Path(output_cfg.get("state", "output/clusters.json"))
    hierarchy_path = output_cfg.get("hierarchy")

    taxonomy_path = cfg.get("taxonomy")
    criteria_path = cfg.get("criteria")
    model = algo_cfg.get("model", "gpt-4.1")

    lambda_ = algo_cfg.get("lambda", 0.1)
    eps = algo_cfg.get("eps", 0.5)
    max_split_size = algo_cfg.get("max_split_size", 10)
    split_depth = algo_cfg.get("split_depth_limit", 3)
    fade_interval = algo_cfg.get("fade_interval_interviews", 10)
    no_splits = args.no_splits or algo_cfg.get("no_splits", False)

    # ---- Load criteria from file ----------------------------------------
    criteria = None
    if criteria_path:
        criteria = _load_criteria(criteria_path)

    # ---- Load or create pipeline ----------------------------------------
    resume_path = args.resume or cfg.get("resume")
    if resume_path:
        print(f"Resuming from {resume_path}")
        pipe = OnlineClusteringPipeline.load(
            resume_path,
            lambda_=lambda_,
            eps=eps,
            max_split_size=max_split_size,
            split_depth_limit=0 if no_splits else split_depth,
            fade_interval_interviews=fade_interval,
            criteria=criteria,
            model=model,
        )
    else:
        pipe = OnlineClusteringPipeline(
            lambda_=lambda_,
            eps=eps,
            max_split_size=max_split_size,
            split_depth_limit=0 if no_splits else split_depth,
            fade_interval_interviews=fade_interval,
            criteria=criteria,
            model=model,
        )

    # ---- Taxonomy warmup ------------------------------------------------
    if taxonomy_path:
        n = pipe.load_taxonomy(taxonomy_path)
        print(f"Loaded taxonomy: {n} nodes")

    # ---- Load input data ------------------------------------------------
    items = _load_items(
        input_path=input_path,
        text_field=text_field,
        source_field=source_field,
        timestamp_field=timestamp_field,
        items_field=items_field,
    )
    print(f"Loaded {len(items)} task items from {input_path}")

    # ---- Run pipeline ---------------------------------------------------
    assignments = pipe.ingest_batch(items, verbose=True)

    # Final fade + split pass
    pipe.force_fade()
    n_flushed = pipe.flush_buffer()
    if n_flushed:
        print(f"Flushed {n_flushed} buffered items → singleton clusters")
    if not no_splits:
        new_ids = pipe.force_split_check()
        if new_ids:
            print(f"Final split pass: {len(new_ids)} new sub-clusters")

    # ---- Summary --------------------------------------------------------
    s = pipe.summary()
    print("\n=== Cluster summary ===")
    for k, v in s.items():
        print(f"  {k}: {v}")

    # ---- Save output ----------------------------------------------------
    output_path = Path(output_path)
    pipe.save(output_path)
    print(f"\nState saved to {output_path}")

    hier_path = Path(hierarchy_path) if hierarchy_path else \
        output_path.parent / (output_path.stem + "_hierarchy.json")
    hier_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hier_path, "w") as f:
        json.dump(pipe.get_hierarchy(), f, indent=2)
    print(f"Hierarchy saved to {hier_path}")


if __name__ == "__main__":
    main()
