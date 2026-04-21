#!/usr/bin/env python3
"""Summarize per-turn latency logs.

Reads one or more turn_latency_breakdown.csv files and prints aggregate
percentiles (p50/p90/p95/p99) for latency stages.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean
from typing import Dict, List


METRIC_COLUMNS: Dict[str, str] = {
    "end_to_end": "End-to-End Delay (seconds)",
    "generation": "Generation Delay (seconds)",
    "delivery": "Delivery Delay (seconds)",
    "queue": "Queue Delay (seconds)",
}


def _percentile(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * (p / 100.0)
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] + (values[hi] - values[lo]) * frac


def _parse_float(value: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _discover_csvs(root: Path, user_id: str | None) -> List[Path]:
    pattern = "**/evaluations/turn_latency_breakdown.csv"
    files = sorted(root.glob(pattern))
    if user_id:
        needle = f"/{user_id}/"
        files = [f for f in files if needle in f.as_posix()]
    return files


def _load_values(csv_paths: List[Path], metric_col: str, session_id: str | None) -> List[float]:
    values: List[float] = []
    for csv_path in csv_paths:
        with csv_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if session_id and str(row.get("Session ID", "")) != str(session_id):
                    continue
                metric_value = _parse_float(row.get(metric_col, ""))
                if metric_value is None:
                    continue
                values.append(metric_value)
    return values


def _print_summary(metric_name: str, values: List[float]) -> None:
    if not values:
        print(f"No samples found for metric '{metric_name}'.")
        return

    values = sorted(values)
    print(f"Metric: {metric_name}")
    print(f"Count:  {len(values)}")
    print(f"Mean:   {mean(values):.3f}s")
    print(f"Min:    {values[0]:.3f}s")
    print(f"P50:    {_percentile(values, 50):.3f}s")
    print(f"P90:    {_percentile(values, 90):.3f}s")
    print(f"P95:    {_percentile(values, 95):.3f}s")
    print(f"P99:    {_percentile(values, 99):.3f}s")
    print(f"Max:    {values[-1]:.3f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize turn latency logs")
    parser.add_argument(
        "--logs-root",
        type=Path,
        default=Path("logs"),
        help="Root logs directory (default: logs)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Explicit turn_latency_breakdown.csv file (skip discovery)",
    )
    parser.add_argument(
        "--user-id",
        type=str,
        default=None,
        help="Filter discovered CSVs to a user id",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Filter rows to one session id",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_COLUMNS.keys()),
        default="end_to_end",
        help="Metric to summarize",
    )
    args = parser.parse_args()

    if args.csv:
        csv_paths = [args.csv]
    else:
        csv_paths = _discover_csvs(args.logs_root, args.user_id)

    if not csv_paths:
        print("No turn_latency_breakdown.csv files found.")
        return

    metric_col = METRIC_COLUMNS[args.metric]
    values = _load_values(csv_paths, metric_col, args.session_id)

    print("Files:")
    for path in csv_paths:
        print(f"- {path}")
    print()

    _print_summary(args.metric, values)


if __name__ == "__main__":
    main()
