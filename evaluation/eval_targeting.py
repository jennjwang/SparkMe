"""
eval_targeting.py  —  Gap 1: Interviewer targeting assessment

For each interview session, reconstructs the turn-by-turn question trajectory
from consecutive session_agenda snapshots and evaluates whether each new question
was aimed at a subtopic that was still uncovered at the time it was asked.

Metrics
-------
targeting_precision : fraction of new questions aimed at uncovered subtopics
re_ask_rate         : fraction of new questions aimed at already-covered subtopics
                      (= 1 - targeting_precision)
coverage_efficiency : coverage gained per question asked  (Δcovered_subtopics / Δquestions)

Output per user (JSON):
{
  "summary": {
    "total_new_questions": N,
    "targeted_uncovered":  K,
    "targeting_precision": K/N,
    "re_ask_rate":         (N-K)/N,
    "coverage_efficiency": ...
  },
  "per_topic": { "1": { "topic": "...", "precision": ..., ...}, ... },
  "turns": [
    { "snap_idx": t, "subtopic_id": "1.3", "was_covered_before": true, "question": "..." },
    ...
  ]
}

Usage
-----
python evaluation/eval_targeting.py \\
    --base-path logs/ \\
    --sample-users-path analysis/sample_users_50.json \\
    --output-dir results/targeting \\
    --summary-path results/targeting_summary.json
"""

import argparse
import glob
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)


# ---------------------------------------------------------------------------
# Snapshot loading helpers
# ---------------------------------------------------------------------------

def load_ordered_snapshots(session_dir: str) -> List[Dict]:
    """Return session_agenda_snap_*.json contents in ascending snap-index order."""
    pattern = os.path.join(session_dir, "session_agenda_snap_*.json")
    paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(os.path.basename(p).replace("session_agenda_snap_", "").replace(".json", ""))
    )
    snaps = []
    for path in paths:
        with open(path) as f:
            snaps.append(json.load(f))
    return snaps


def find_session_dir(base_path: str, user_id: str) -> Optional[str]:
    """Return the first session directory that has chat_history.log."""
    exec_dir = os.path.join(base_path, user_id, "execution_logs")
    if not os.path.isdir(exec_dir):
        return None
    for entry in sorted(os.listdir(exec_dir)):
        candidate = os.path.join(exec_dir, entry)
        if os.path.isdir(candidate) and glob.glob(os.path.join(candidate, "session_agenda_snap_*.json")):
            return candidate
    return None


# ---------------------------------------------------------------------------
# Per-snapshot state extraction
# ---------------------------------------------------------------------------

def extract_state(snap: Dict) -> Tuple[Dict[str, set], Dict[str, bool], Dict[str, str]]:
    """
    Returns:
        qid_sets    : {subtopic_id → set of question_ids}
        covered     : {subtopic_id → is_covered bool}
        topic_names : {topic_id → topic description}
    """
    qid_sets: Dict[str, set] = {}
    covered: Dict[str, bool] = {}
    topic_names: Dict[str, str] = {}

    tm = snap.get("interview_topic_manager", {})
    for tid, topic in tm.get("core_topic_dict", {}).items():
        topic_names[str(tid)] = topic.get("description", "")
        for sid, sub in topic.get("required_subtopics", {}).items():
            qid_sets[sid] = {q["question_id"] for q in sub.get("questions", [])}
            covered[sid] = bool(sub.get("is_covered", False))

    return qid_sets, covered, topic_names


def extract_new_questions(snap: Dict, prev_qid_sets: Dict[str, set]) -> List[Dict]:
    """
    Find questions that appear in snap but not in prev_qid_sets.
    Returns list of {subtopic_id, question_id, question_text}.
    """
    new_qs = []
    tm = snap.get("interview_topic_manager", {})
    for tid, topic in tm.get("core_topic_dict", {}).items():
        for sid, sub in topic.get("required_subtopics", {}).items():
            prev_ids = prev_qid_sets.get(sid, set())
            for q in sub.get("questions", []):
                if q["question_id"] not in prev_ids:
                    new_qs.append({
                        "subtopic_id": sid,
                        "topic_id": str(tid),
                        "question_id": q["question_id"],
                        "question": q.get("question", ""),
                    })
    return new_qs


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_session(session_dir: str,
                     bulk_threshold: int = 3) -> Optional[Dict]:
    """
    bulk_threshold: if more than this many new questions appear in a single
    snapshot transition, treat it as a StrategicPlanner pre-generation event
    (not an actual interviewer ask) and skip it.
    """
    snaps = load_ordered_snapshots(session_dir)
    if len(snaps) < 2:
        return None

    topic_names: Dict[str, str] = {}
    turns = []
    skipped_bulk_snaps = 0

    prev_qids, prev_covered, topic_names = extract_state(snaps[0])
    initial_covered = sum(1 for v in prev_covered.values() if v)

    for snap_idx, snap in enumerate(snaps[1:], start=1):
        curr_qids, curr_covered, tn = extract_state(snap)
        topic_names.update(tn)

        new_qs = extract_new_questions(snap, prev_qids)

        if len(new_qs) > bulk_threshold:
            # Bulk pre-generation by StrategicPlanner — skip for targeting eval
            skipped_bulk_snaps += 1
            prev_qids = curr_qids
            prev_covered = curr_covered
            continue

        for q in new_qs:
            sid = q["subtopic_id"]
            was_covered = prev_covered.get(sid, False)
            turns.append({
                "snap_idx": snap_idx,
                "subtopic_id": sid,
                "topic_id": q["topic_id"],
                "question_id": q["question_id"],
                "question": q["question"],
                "was_covered_before": was_covered,
            })

        prev_qids = curr_qids
        prev_covered = curr_covered

    if not turns:
        return None

    final_covered = sum(1 for v in prev_covered.values() if v)
    total_qs = len(turns)
    targeted_uncovered = sum(1 for t in turns if not t["was_covered_before"])
    re_asked = total_qs - targeted_uncovered
    coverage_gained = final_covered - initial_covered
    _ = skipped_bulk_snaps  # referenced below

    # Per-topic breakdown
    per_topic: Dict[str, Dict] = {}
    for turn in turns:
        tid = turn["topic_id"]
        per_topic.setdefault(tid, {
            "topic": topic_names.get(tid, ""),
            "total_questions": 0,
            "targeted_uncovered": 0,
        })
        per_topic[tid]["total_questions"] += 1
        if not turn["was_covered_before"]:
            per_topic[tid]["targeted_uncovered"] += 1

    for tid, d in per_topic.items():
        n = d["total_questions"]
        d["targeting_precision"] = d["targeted_uncovered"] / n if n else 0.0
        d["re_ask_rate"] = 1.0 - d["targeting_precision"]

    return {
        "summary": {
            "total_new_questions": total_qs,
            "targeted_uncovered": targeted_uncovered,
            "re_asked_covered": re_asked,
            "targeting_precision": targeted_uncovered / total_qs if total_qs else 0.0,
            "re_ask_rate": re_asked / total_qs if total_qs else 0.0,
            "coverage_efficiency": coverage_gained / total_qs if total_qs else 0.0,
            "subtopics_covered": final_covered,
            "total_subtopics": len(prev_covered),
            "skipped_bulk_snaps": skipped_bulk_snaps,
            "bulk_threshold": bulk_threshold,
        },
        "per_topic": per_topic,
        "turns": turns,
    }


# ---------------------------------------------------------------------------
# Per-user runner
# ---------------------------------------------------------------------------

def evaluate_user(user_id: str, base_path: str, output_dir: str,
                  overwrite: bool = False, bulk_threshold: int = 3) -> Optional[Dict]:
    save_path = os.path.join(output_dir, f"{user_id}.json")
    if not overwrite and os.path.exists(save_path):
        return None

    session_dir = find_session_dir(base_path, user_id)
    if session_dir is None:
        logging.warning(f"{user_id}: no session directory found")
        return None

    result = evaluate_session(session_dir, bulk_threshold=bulk_threshold)
    if result is None:
        logging.warning(f"{user_id}: insufficient snapshots")
        return None

    result["user_id"] = user_id
    result["session_dir"] = session_dir

    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interviewer targeting assessment (Gap 1)")
    parser.add_argument("--base-path", required=True, help="Base path to logs directory")
    parser.add_argument("--sample-users-path", default="analysis/sample_users_50.json")
    parser.add_argument("--output-dir", default="results/targeting")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--bulk-threshold", type=int, default=3,
                        help="Max new questions per snap to count as real asks (default 3); "
                             "larger bursts are treated as StrategicPlanner pre-generation")
    args = parser.parse_args()

    with open(args.sample_users_path) as f:
        sample_users = json.load(f)
    user_ids = [u["User ID"] for u in sample_users[: args.num_users]]

    all_results = []

    def _run(uid):
        return evaluate_user(uid, args.base_path, args.output_dir, args.overwrite,
                             bulk_threshold=args.bulk_threshold)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, uid): uid for uid in user_ids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating targeting"):
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

    precisions = [r["summary"]["targeting_precision"] for r in all_results]
    re_ask_rates = [r["summary"]["re_ask_rate"] for r in all_results]
    efficiencies = [r["summary"]["coverage_efficiency"] for r in all_results]

    summary = {
        "num_users": len(all_results),
        "mean_targeting_precision": sum(precisions) / len(precisions),
        "mean_re_ask_rate": sum(re_ask_rates) / len(re_ask_rates),
        "mean_coverage_efficiency": sum(efficiencies) / len(efficiencies),
    }

    logging.info(
        f"Done. Users={summary['num_users']}  "
        f"Precision={summary['mean_targeting_precision']:.3f}  "
        f"Re-ask={summary['mean_re_ask_rate']:.3f}  "
        f"Efficiency={summary['mean_coverage_efficiency']:.3f}"
    )

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump({"summary": summary, "per_user": all_results}, f, indent=2)
        logging.info(f"Summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
