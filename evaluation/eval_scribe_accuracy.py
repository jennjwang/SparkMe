"""
eval_scribe_accuracy.py  —  Gap 2: Scribe self-assessment accuracy

Validates whether the SessionScribe's per-subtopic is_covered flag
matches ground-truth recall (from eval_fact_recall outputs).

The scribe internally marks each subtopic as covered/uncovered via criteria
checks. This script asks: when is_covered=True, does the subtopic actually
have high GT recall? When False, does recall remain low?

Metrics (treating is_covered as a binary classifier of GT recall >= threshold)
-----------------------------------------------------------------------
accuracy    : fraction of subtopics correctly classified
precision   : P(high GT recall | is_covered=True)
recall      : P(is_covered=True | high GT recall)
f1          : harmonic mean
calibration : mean GT recall when is_covered=True vs False

Both per-snapshot and final-snapshot results are reported.

Prerequisites
-------------
Run eval_fact_recall.py first to produce per-user fact recall files at:
  {base_path}/{user_id}/evaluations_fact_recall/snap_eval_{idx}.json

Usage
-----
python evaluation/eval_scribe_accuracy.py \\
    --base-path logs/ \\
    --sample-users-path analysis/sample_users_50.json \\
    --output-dir results/scribe_accuracy \\
    --summary-path results/scribe_accuracy_summary.json \\
    --recall-threshold 0.5
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
# Helpers
# ---------------------------------------------------------------------------

def find_session_snap_dir(base_path: str, user_id: str) -> Optional[str]:
    """Return the session directory used by eval_fact_recall (session_0)."""
    path = os.path.join(base_path, user_id, "execution_logs", "session_0")
    return path if os.path.isdir(path) else None


def load_snap_coverage(snap_path: str) -> Dict[str, bool]:
    """Extract {subtopic_id: is_covered} from a session_agenda_snap file."""
    with open(snap_path) as f:
        snap = json.load(f)
    coverage = {}
    tm = snap.get("interview_topic_manager", {})
    for tid, topic in tm.get("core_topic_dict", {}).items():
        for sid, sub in topic.get("required_subtopics", {}).items():
            coverage[sid] = bool(sub.get("is_covered", False))
    return coverage


def load_gt_recall(recall_path: str) -> Dict[str, float]:
    """Extract {subtopic_id: recall_0_to_1} from a snap_eval_*.json file."""
    with open(recall_path) as f:
        data = json.load(f)
    per_subtopic = data.get("per_subtopic", {})
    return {sid: info["recall"] for sid, info in per_subtopic.items()}


def classification_metrics(
    pairs: List[Tuple[bool, bool]]  # (predicted_covered, gt_high_recall)
) -> Dict:
    tp = sum(1 for p, g in pairs if p and g)
    tn = sum(1 for p, g in pairs if not p and not g)
    fp = sum(1 for p, g in pairs if p and not g)
    fn = sum(1 for p, g in pairs if not p and g)
    total = len(pairs)

    accuracy  = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "accuracy": accuracy, "precision": precision,
            "recall": recall, "f1": f1, "n": total}


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_user(user_id: str, base_path: str, output_dir: str,
                  recall_threshold: float = 0.5,
                  overwrite: bool = False) -> Optional[Dict]:
    save_path = os.path.join(output_dir, f"{user_id}.json")
    if not overwrite and os.path.exists(save_path):
        return None

    snap_dir = find_session_snap_dir(base_path, user_id)
    if snap_dir is None:
        logging.warning(f"{user_id}: session_0 directory not found")
        return None

    recall_dir = os.path.join(base_path, user_id, "evaluations_fact_recall")
    if not os.path.isdir(recall_dir):
        logging.warning(f"{user_id}: no eval_fact_recall results found — run eval_fact_recall.py first")
        return None

    snap_paths = sorted(
        glob.glob(os.path.join(snap_dir, "session_agenda_snap_*.json")),
        key=lambda p: int(os.path.basename(p).replace("session_agenda_snap_", "").replace(".json", ""))
    )

    per_snap = []
    all_pairs: List[Tuple[bool, bool]] = []

    for snap_path in snap_paths:
        idx = int(os.path.basename(snap_path).replace("session_agenda_snap_", "").replace(".json", ""))
        recall_path = os.path.join(recall_dir, f"snap_eval_{idx}.json")
        if not os.path.exists(recall_path):
            continue

        scribe_coverage = load_snap_coverage(snap_path)
        gt_recall = load_gt_recall(recall_path)

        pairs: List[Tuple[bool, bool]] = []
        details = []
        for sid in scribe_coverage:
            if sid not in gt_recall:
                continue
            predicted = scribe_coverage[sid]
            gt_high   = gt_recall[sid] >= recall_threshold
            pairs.append((predicted, gt_high))
            all_pairs.append((predicted, gt_high))
            details.append({
                "subtopic_id": sid,
                "is_covered": predicted,
                "gt_recall": gt_recall[sid],
                "gt_high_recall": gt_high,
                "correct": predicted == gt_high,
            })

        if not pairs:
            continue

        metrics = classification_metrics(pairs)
        # Calibration: mean GT recall when scribe says covered vs not
        covered_recalls   = [gt_recall[sid] for sid, cov in scribe_coverage.items()
                             if cov and sid in gt_recall]
        uncovered_recalls = [gt_recall[sid] for sid, cov in scribe_coverage.items()
                             if not cov and sid in gt_recall]

        per_snap.append({
            "snap_idx": idx,
            "metrics": metrics,
            "calibration": {
                "mean_recall_when_covered":   sum(covered_recalls) / len(covered_recalls)   if covered_recalls   else None,
                "mean_recall_when_uncovered": sum(uncovered_recalls) / len(uncovered_recalls) if uncovered_recalls else None,
            },
            "details": details,
        })

    if not per_snap:
        logging.warning(f"{user_id}: no matching snap/recall pairs found")
        return None

    # Final-snapshot summary
    final = per_snap[-1]
    overall = classification_metrics(all_pairs)

    result = {
        "user_id": user_id,
        "recall_threshold": recall_threshold,
        "overall": overall,
        "final_snap": final,
        "per_snap": per_snap,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scribe self-assessment accuracy (Gap 2)")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--sample-users-path", default="analysis/sample_users_50.json")
    parser.add_argument("--output-dir", default="results/scribe_accuracy")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--recall-threshold", type=float, default=0.5,
                        help="GT recall >= this counts as 'high recall' (default 0.5)")
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
                             args.recall_threshold, args.overwrite)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, uid): uid for uid in user_ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Evaluating scribe accuracy"):
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

    def _mean(key):
        vals = [r["overall"][key] for r in all_results]
        return sum(vals) / len(vals)

    summary = {
        "recall_threshold": args.recall_threshold,
        "num_users": len(all_results),
        "mean_accuracy":  _mean("accuracy"),
        "mean_precision": _mean("precision"),
        "mean_recall":    _mean("recall"),
        "mean_f1":        _mean("f1"),
    }
    logging.info(
        f"Done. Users={summary['num_users']}  "
        f"Acc={summary['mean_accuracy']:.3f}  "
        f"Prec={summary['mean_precision']:.3f}  "
        f"Rec={summary['mean_recall']:.3f}  "
        f"F1={summary['mean_f1']:.3f}"
    )

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump({"summary": summary, "per_user": all_results}, f, indent=2)
        logging.info(f"Summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
