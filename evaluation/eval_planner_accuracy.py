"""
eval_planner_accuracy.py  —  Gap 3: StrategicPlanner rollout accuracy

For each planning event the StrategicPlanner fires, it saves a
strategic_state_turn_N.json file with rollout predictions. This script
asks: did the subtopics the planner predicted would be covered in the
next few turns actually get covered?

Approach
--------
For each session directory that contains both strategic_state_turn_N.json
files and session_agenda_snap_*.json files:

  - strategic_state_turn_N.json: planner fired at turn N, contains rollout
    predictions with subtopics_covered per predicted turn.
  - snap_(N-1).json: coverage state just before turn N (the "before" state).
  - snap_(N+lookahead-1).json: coverage state after lookahead more turns.

Predicted subtopics = union of all rollouts' predicted subtopics_covered.

Metrics
-------
precision : fraction of predicted subtopics that actually became covered
recall    : fraction of newly-covered subtopics that the planner predicted

Output per user (JSON):
{
  "summary": {
    "planning_events": K,
    "mean_precision": ..., "mean_recall": ..., "mean_f1": ...
  },
  "events": [
    {
      "planning_turn": N,
      "snap_idx_before": N-1,
      "snap_idx_after": N+lookahead-1,
      "predicted_subtopics": [...],
      "newly_covered": [...],
      "precision": ..., "recall": ..., "f1": ...
    }, ...
  ]
}

Usage
-----
python evaluation/eval_planner_accuracy.py \\
    --base-path logs/ \\
    --sample-users-path analysis/sample_users_50.json \\
    --output-dir results/planner_accuracy \\
    --summary-path results/planner_accuracy_summary.json
"""

import argparse
import glob
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# ---------------------------------------------------------------------------
# Load coverage from a snap file
# ---------------------------------------------------------------------------

def load_snap_coverage(snap_path: str) -> Dict[str, bool]:
    """Return {subtopic_id: is_covered} from a session_agenda_snap file."""
    with open(snap_path) as f:
        snap = json.load(f)
    coverage = {}
    for tid, topic in snap.get("interview_topic_manager", {}).get("core_topic_dict", {}).items():
        for sid, sub in topic.get("required_subtopics", {}).items():
            coverage[sid] = bool(sub.get("is_covered", False))
    return coverage


def load_snap_index(session_dir: str) -> Dict[int, str]:
    """Return {snap_N: file_path} for all snap files in session_dir."""
    index = {}
    for path in glob.glob(os.path.join(session_dir, "session_agenda_snap_*.json")):
        n = int(os.path.basename(path).replace("session_agenda_snap_", "").replace(".json", ""))
        index[n] = path
    return index


# ---------------------------------------------------------------------------
# Load strategic state files
# ---------------------------------------------------------------------------

def load_strategic_states(session_dir: str) -> List[Dict]:
    """
    Load all strategic_state_turn_N.json files and return one entry per
    unique planning event.

    The StrategicPlanner saves a state file at EVERY turn (turn_N), but
    planning only fires occasionally. The `last_planning_turn` field records
    the most recent turn when planning actually happened.

    Strategy: for each unique `last_planning_turn` value P, use the file with
    the lowest N >= P (the first state snapshot written after planning at P),
    and treat P as the planning turn for snap-index matching.
    """
    # Map: last_planning_turn → (file_n, data)
    by_planning_turn: Dict[int, tuple] = {}
    for path in glob.glob(os.path.join(session_dir, "strategic_state_turn_*.json")):
        n = int(os.path.basename(path).replace("strategic_state_turn_", "").replace(".json", ""))
        with open(path) as f:
            data = json.load(f)
        p = data.get("last_planning_turn")
        if p is None:
            continue
        # Keep the earliest file for each planning turn
        if p not in by_planning_turn or n < by_planning_turn[p][0]:
            by_planning_turn[p] = (n, data)

    states = []
    for planning_turn, (_, data) in sorted(by_planning_turn.items()):
        rollouts = data.get("rollout_predictions", [])
        # Pick the winning rollout by highest utility_score
        if not rollouts:
            continue
        winner = max(rollouts, key=lambda r: r.get("utility_score", float("-inf")))
        predicted: Set[str] = set()
        for turn in winner.get("predicted_turns", []):
            predicted.update(turn.get("subtopics_covered", []))

        states.append({
            "turn_n": planning_turn,      # snap-index anchor = when planning fired
            "predicted_subtopics": sorted(predicted),
        })

    return states


# ---------------------------------------------------------------------------
# Core evaluation per session
# ---------------------------------------------------------------------------

def evaluate_session(session_dir: str, lookahead: int,
                     snap_index_override: Optional[Dict[int, str]] = None) -> List[Dict]:
    """Evaluate all planning events in one session directory."""
    snap_index = snap_index_override if snap_index_override is not None else load_snap_index(session_dir)
    if not snap_index:
        return []

    states = load_strategic_states(session_dir)
    if not states:
        return []

    max_snap = max(snap_index.keys())
    evaluated = []

    for state in states:
        n = state["turn_n"]
        predicted = set(state["predicted_subtopics"])

        if not predicted:
            continue

        # snap_before = snap_(n-1) if it exists, else snap_0
        before_idx = n - 1
        while before_idx > 0 and before_idx not in snap_index:
            before_idx -= 1
        if before_idx not in snap_index:
            before_idx = min(snap_index.keys())

        # snap_after = snap_(n + lookahead - 1), clamped to max
        after_idx = n + lookahead - 1
        while after_idx > max_snap and after_idx > before_idx:
            after_idx -= 1
        if after_idx not in snap_index:
            # Find nearest available snap >= n
            candidates = [k for k in snap_index if k >= n]
            if not candidates:
                continue
            after_idx = max(candidates)

        before_cov = load_snap_coverage(snap_index[before_idx])
        after_cov = load_snap_coverage(snap_index[after_idx])

        if before_idx == after_idx:
            # No lookahead available; skip
            continue

        newly_covered = {sid for sid, cov in after_cov.items()
                         if cov and not before_cov.get(sid, False)}

        tp = len(predicted & newly_covered)
        precision = tp / len(predicted) if predicted else 0.0

        evaluated.append({
            "planning_turn": n,
            "snap_idx_before": before_idx,
            "snap_idx_after": after_idx,
            "predicted_subtopics": sorted(predicted),
            "newly_covered": sorted(newly_covered),
            "true_positives": sorted(predicted & newly_covered),
            "false_positives": sorted(predicted - newly_covered),
            "precision": precision,
        })

    return evaluated


# ---------------------------------------------------------------------------
# Core evaluation per user
# ---------------------------------------------------------------------------

def evaluate_user(user_id: str, base_path: str, output_dir: str,
                  lookahead: int = 5, overwrite: bool = False) -> Optional[Dict]:
    save_path = os.path.join(output_dir, f"{user_id}.json")
    if not overwrite and os.path.exists(save_path):
        return None

    exec_dir = os.path.join(base_path, user_id, "execution_logs")
    if not os.path.isdir(exec_dir):
        return None

    # Collect ALL snaps across ALL session directories (they accumulate in
    # session_0/ because session_agenda.session_id stays at 0).
    global_snap_index: Dict[int, str] = {}
    for session_name in sorted(os.listdir(exec_dir)):
        session_dir = os.path.join(exec_dir, session_name)
        if os.path.isdir(session_dir):
            global_snap_index.update(load_snap_index(session_dir))

    all_evaluated = []
    for session_name in sorted(os.listdir(exec_dir)):
        session_dir = os.path.join(exec_dir, session_name)
        if not os.path.isdir(session_dir):
            continue
        # Pass the global snap index so planning events in session_1+ can be
        # matched against snaps that landed in session_0.
        events = evaluate_session(session_dir, lookahead, snap_index_override=global_snap_index)
        all_evaluated.extend(events)

    if not all_evaluated:
        logging.warning(f"{user_id}: no evaluable planning events found")
        return None

    valid_prec = [e["precision"] for e in all_evaluated]

    result = {
        "user_id": user_id,
        "lookahead_snaps": lookahead,
        "summary": {
            "planning_events": len(all_evaluated),
            "mean_precision": sum(valid_prec) / len(valid_prec) if valid_prec else None,
        },
        "events": all_evaluated,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StrategicPlanner rollout accuracy (Gap 3)")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--sample-users-path", default="analysis/sample_users_50.json")
    parser.add_argument("--output-dir", default="results/planner_accuracy")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--lookahead", type=int, default=5,
                        help="How many turns after planning to check for coverage (default 5)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.sample_users_path) as f:
        sample_users = json.load(f)
    user_ids = [u["User ID"] for u in sample_users[: args.num_users]]

    all_results = []

    def _run(uid):
        return evaluate_user(uid, args.base_path, args.output_dir,
                             args.lookahead, args.overwrite)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, uid): uid for uid in user_ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Evaluating planner accuracy"):
            uid = futures[fut]
            try:
                result = fut.result()
                if result:
                    all_results.append(result)
            except Exception as e:
                logging.error(f"{uid}: {e}")

    if not all_results:
        logging.warning("No results collected.")
        return

    precs = [r["summary"]["mean_precision"] for r in all_results if r["summary"]["mean_precision"] is not None]

    summary = {
        "num_users": len(all_results),
        "lookahead_snaps": args.lookahead,
        "mean_precision": sum(precs) / len(precs) if precs else None,
    }
    logging.info(
        f"Done. Users={summary['num_users']}  "
        f"Prec={summary['mean_precision']:.3f}"
    )

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump({"summary": summary, "per_user": all_results}, f, indent=2)
        logging.info(f"Summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
