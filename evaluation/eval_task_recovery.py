"""
eval_task_recovery.py — Precision and recall for task recovery.

Given a ground-truth set of tasks and a predicted (recovered) set of tasks,
uses an LLM judge to determine semantic matches, then computes:
  - precision = matched predicted / total predicted
  - recall    = matched ground truth / total ground truth
  - f1        = harmonic mean of precision and recall

A "match" means the two tasks describe the same underlying work activity,
even if phrased differently.

Usage
-----
# Two inline lists
python evaluation/eval_task_recovery.py \
    --ground-truth "reading research papers" "writing code" "attending meetings" \
    --predicted "reading papers" "coding experiments" "weekly meetings"

# From JSON files (each file: a list of task strings)
python evaluation/eval_task_recovery.py \
    --ground-truth-file gt_tasks.json \
    --predicted-file predicted_tasks.json

# From pilot user portrait vs. a sim session's portrait
python evaluation/eval_task_recovery.py \
    --pilot-user-id seQCtyofmgnnQCZWjS-VWQ \
    --logs-user-id sim_seQCtyof_h000_1234567 \
    --pilot-dir pilot \
    --logs-dir logs
"""

import argparse
import json
import os
import sys
import textwrap
from typing import Optional

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI

_client: Optional[OpenAI] = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


_MATCH_PROMPT = textwrap.dedent("""\
    You are judging whether tasks from a "predicted" list semantically match
    tasks from a "ground truth" list.

    Two tasks match when they describe the **same underlying work activity**,
    even if phrased differently. Minor differences in wording, specificity, or
    framing are fine as long as the core activity is the same.

    ## Ground truth tasks (numbered from 0)
    {gt_block}

    ## Predicted tasks (numbered from 0)
    {pred_block}

    ## Instructions
    For each predicted task, list every ground-truth task index it matches
    (0-indexed). A predicted task may match more than one ground-truth task
    if they all describe the same activity. If a predicted task matches
    nothing, use an empty list.

    Return ONLY a JSON object with a single key "matches": a list of lists,
    one per predicted task (in order).

    Example format (3 predicted tasks):
    {{"matches": [[0, 2], [], [1]]}}
""")


def _llm_match(ground_truth: list[str], predicted: list[str], model: str = "gpt-4.1-mini") -> list[list[int]]:
    """Ask the LLM to match predicted tasks to ground-truth tasks.

    Returns a list of length len(predicted), where each element is a list of
    ground-truth indices that the predicted task matches.
    """
    gt_block = "\n".join(f"  {i}. {t}" for i, t in enumerate(ground_truth))
    pred_block = "\n".join(f"  {i}. {t}" for i, t in enumerate(predicted))

    prompt = _MATCH_PROMPT.format(gt_block=gt_block, pred_block=pred_block)

    response = _get_client().chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    data = json.loads(raw)
    matches = data.get("matches", [])

    # Pad or truncate to match len(predicted)
    while len(matches) < len(predicted):
        matches.append([])
    return matches[: len(predicted)]


def compute_metrics(
    ground_truth: list[str],
    predicted: list[str],
    model: str = "gpt-4.1-mini",
    verbose: bool = False,
) -> dict:
    """Compute precision, recall, and F1 for task recovery.

    Args:
        ground_truth: The reference task list.
        predicted: The recovered task list to evaluate.
        model: OpenAI model to use for semantic matching.
        verbose: If True, print per-task match details.

    Returns:
        dict with keys: precision, recall, f1, matched_predicted,
        total_predicted, matched_gt, total_gt, match_matrix
    """
    if not ground_truth or not predicted:
        return {
            "precision": 0.0 if not predicted else 1.0,
            "recall": 0.0,
            "f1": 0.0,
            "matched_predicted": 0,
            "total_predicted": len(predicted),
            "matched_gt": 0,
            "total_gt": len(ground_truth),
            "match_matrix": [],
        }

    match_matrix = _llm_match(ground_truth, predicted, model=model)

    # Which GT tasks were hit by at least one predicted task?
    gt_covered: set[int] = set()
    for gt_indices in match_matrix:
        gt_covered.update(gt_indices)

    # Which predicted tasks matched at least one GT task?
    pred_matched = sum(1 for gt_indices in match_matrix if gt_indices)

    precision = pred_matched / len(predicted)
    recall = len(gt_covered) / len(ground_truth)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    if verbose:
        print("\n=== Match Details ===")
        for i, (task, gt_indices) in enumerate(zip(predicted, match_matrix)):
            label = f"  GT{gt_indices}" if gt_indices else "  (no match)"
            print(f"  P{i}: {task!r}  →  {label}")
        print(f"\nGT tasks NOT recovered ({len(ground_truth) - len(gt_covered)}):")
        for j, t in enumerate(ground_truth):
            if j not in gt_covered:
                print(f"  GT{j}: {t!r}")

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "matched_predicted": pred_matched,
        "total_predicted": len(predicted),
        "matched_gt": len(gt_covered),
        "total_gt": len(ground_truth),
        "match_matrix": match_matrix,
    }


# ---------------------------------------------------------------------------
# Helpers for loading tasks from pilot portraits / log portraits
# ---------------------------------------------------------------------------

def _load_portrait(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _tasks_from_portrait(portrait: dict) -> list[str]:
    tasks = portrait.get("Task Inventory", [])
    # Filter to non-empty strings
    return [t.strip() for t in tasks if isinstance(t, str) and t.strip()]


def _load_pilot_tasks(user_id: str, pilot_dir: str = "pilot") -> list[str]:
    path = os.path.join(pilot_dir, user_id, "user_portrait.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pilot portrait not found: {path}")
    return _tasks_from_portrait(_load_portrait(path))


def _load_logs_tasks(user_id: str, logs_dir: str = "logs") -> list[str]:
    path = os.path.join(logs_dir, user_id, "user_portrait.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Logs portrait not found: {path}")
    return _tasks_from_portrait(_load_portrait(path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precision/recall for task recovery using semantic LLM matching"
    )

    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--ground-truth", nargs="+", metavar="TASK",
                     help="Ground truth task strings (inline)")
    src.add_argument("--ground-truth-file", metavar="FILE",
                     help="JSON file containing ground truth task list")
    src.add_argument("--pilot-user-id", metavar="USER_ID",
                     help="Pilot user ID to load ground truth from pilot/<id>/user_portrait.json")

    pred = parser.add_mutually_exclusive_group()
    pred.add_argument("--predicted", nargs="+", metavar="TASK",
                      help="Predicted task strings (inline)")
    pred.add_argument("--predicted-file", metavar="FILE",
                      help="JSON file containing predicted task list")
    pred.add_argument("--logs-user-id", metavar="USER_ID",
                      help="Logs user ID to load predicted tasks from logs/<id>/user_portrait.json")

    parser.add_argument("--pilot-dir", default="pilot",
                        help="Path to pilot directory (default: pilot)")
    parser.add_argument("--logs-dir", default="logs",
                        help="Path to logs directory (default: logs)")
    parser.add_argument("--model", default="gpt-4.1-mini",
                        help="OpenAI model for semantic matching (default: gpt-4.1-mini)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-task match details")
    parser.add_argument("--output", metavar="FILE",
                        help="Write results JSON to this file")

    args = parser.parse_args()

    # Load ground truth
    if args.ground_truth:
        gt_tasks = args.ground_truth
    elif args.ground_truth_file:
        with open(args.ground_truth_file) as f:
            gt_tasks = json.load(f)
    else:
        gt_tasks = _load_pilot_tasks(args.pilot_user_id, args.pilot_dir)

    # Load predicted
    if args.predicted:
        pred_tasks = args.predicted
    elif args.predicted_file:
        with open(args.predicted_file) as f:
            pred_tasks = json.load(f)
    elif args.logs_user_id:
        pred_tasks = _load_logs_tasks(args.logs_user_id, args.logs_dir)
    else:
        parser.error("Provide one of --predicted, --predicted-file, or --logs-user-id")

    print(f"Ground truth: {len(gt_tasks)} tasks")
    print(f"Predicted:    {len(pred_tasks)} tasks")

    results = compute_metrics(gt_tasks, pred_tasks, model=args.model, verbose=args.verbose)

    print(f"\nPrecision: {results['precision']:.2%}  ({results['matched_predicted']}/{results['total_predicted']} predicted matched a GT task)")
    print(f"Recall:    {results['recall']:.2%}  ({results['matched_gt']}/{results['total_gt']} GT tasks recovered)")
    print(f"F1:        {results['f1']:.2%}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")


if __name__ == "__main__":
    main()
