"""
Evaluate semantic diversity of the task generator's output.

For each user, loads the generated task list from evaluations/task_widget_data.json
and computes three embedding-based diversity metrics:

  mean_pairwise_dist  — mean cosine distance across all task pairs (higher = more diverse)
  min_pairwise_dist   — minimum cosine distance (catches near-duplicate pairs)
  centroid_spread     — mean distance of each task to the set centroid

Usage:
    python evaluation/eval_task_diversity.py --base-path logs/ --sample-users-path analysis/sample_users_50.json
    python evaluation/eval_task_diversity.py --base-path logs/ --user-id 1241161077
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import numpy as np
from openai import OpenAI
from tqdm import tqdm

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

OPENAI_CLIENT = OpenAI()
EMBED_MODEL = "text-embedding-3-small"
EMBED_BATCH_SIZE = 100


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> np.ndarray:
    """Return (N, D) float32 array of unit-normalised embeddings."""
    all_vecs = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i : i + EMBED_BATCH_SIZE]
        response = OPENAI_CLIENT.embeddings.create(input=batch, model=EMBED_MODEL)
        vecs = np.array([r.embedding for r in response.data], dtype=np.float32)
        all_vecs.append(vecs)
    mat = np.vstack(all_vecs)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def diversity_metrics(embeddings: np.ndarray) -> dict:
    """
    Given an (N, D) array of unit-normalised embeddings, return:
      mean_pairwise_dist  — mean of all pairwise cosine distances
      min_pairwise_dist   — minimum pairwise cosine distance (nearest-neighbour gap)
      centroid_spread     — mean distance of each embedding to the set centroid
    """
    n = len(embeddings)
    if n < 2:
        return {"mean_pairwise_dist": None, "min_pairwise_dist": None, "centroid_spread": None, "n_tasks": n}

    # Cosine similarity matrix (dot product of unit vecs)
    sim = embeddings @ embeddings.T  # (N, N)
    # Cosine distance = 1 - cosine similarity, clipped to [0, 2]
    dist = np.clip(1.0 - sim, 0.0, 2.0)

    # Mask diagonal
    mask = ~np.eye(n, dtype=bool)
    pairwise = dist[mask]

    mean_pairwise = float(np.mean(pairwise))
    min_pairwise = float(np.min(pairwise))

    centroid = embeddings.mean(axis=0)
    centroid_norm = centroid / (np.linalg.norm(centroid) or 1.0)
    centroid_dists = np.clip(1.0 - (embeddings @ centroid_norm), 0.0, 2.0)
    centroid_spread = float(np.mean(centroid_dists))

    return {
        "mean_pairwise_dist": round(mean_pairwise, 4),
        "min_pairwise_dist": round(min_pairwise, 4),
        "centroid_spread": round(centroid_spread, 4),
        "n_tasks": n,
    }


# ---------------------------------------------------------------------------
# Per-user processing
# ---------------------------------------------------------------------------

def _extract_task_names(listed_tasks: list) -> list[str]:
    """Extract string names from the listed_tasks field (list of str or dict)."""
    names = []
    for t in listed_tasks:
        if isinstance(t, str):
            name = t.strip()
        elif isinstance(t, dict):
            name = (t.get("text") or t.get("name") or "").strip()
        else:
            continue
        if name:
            names.append(name)
    return names


def process_user(user_id: str, base_path: str, surgery: bool = False) -> Optional[dict]:
    widget_path = Path(base_path) / user_id / "evaluations" / "task_widget_data.json"
    save_path = Path(base_path) / user_id / "evaluations" / "task_diversity.json"

    if not widget_path.exists():
        logging.debug(f"[{user_id}] task_widget_data.json not found — skipping")
        return None

    if not surgery and save_path.exists():
        try:
            existing = json.loads(save_path.read_text())
            if existing and not existing.get("error"):
                return existing
        except Exception:
            pass

    try:
        data = json.loads(widget_path.read_text())
    except Exception as e:
        logging.warning(f"[{user_id}] Failed to load task_widget_data.json: {e}")
        return None

    listed = data.get("listed") or []
    task_names = _extract_task_names(listed)

    if len(task_names) < 2:
        logging.warning(f"[{user_id}] Only {len(task_names)} task(s) — skipping diversity calc")
        result = {"user_id": user_id, "n_tasks": len(task_names), "error": "too_few_tasks"}
        save_path.write_text(json.dumps(result, indent=2))
        return result

    try:
        embeddings = embed_texts(task_names)
        metrics = diversity_metrics(embeddings)
        result = {
            "user_id": user_id,
            "error": False,
            "tasks": task_names,
            **metrics,
        }
    except Exception as e:
        logging.error(f"[{user_id}] Embedding error: {e}")
        result = {"user_id": user_id, "error": str(e)}

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(result, indent=2))
    return result


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[dict]) -> dict:
    valid = [r for r in results if r and not r.get("error")]
    if not valid:
        return {"n_users": 0}

    def _mean(key):
        vals = [r[key] for r in valid if r.get(key) is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    return {
        "n_users": len(valid),
        "avg_n_tasks": round(float(np.mean([r["n_tasks"] for r in valid])), 1),
        "avg_mean_pairwise_dist": _mean("mean_pairwise_dist"),
        "avg_min_pairwise_dist": _mean("min_pairwise_dist"),
        "avg_centroid_spread": _mean("centroid_spread"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate task generator diversity via embeddings")
    parser.add_argument("--base-path", type=str, required=True, help="Path to logs directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample-users-path", type=str, help="Path to sample users JSON")
    group.add_argument("--user-id", type=str, help="Single user ID to evaluate")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--num-users", type=int, default=None)
    parser.add_argument("--surgery", action="store_true", help="Re-evaluate even if results exist")
    args = parser.parse_args()

    if args.user_id:
        user_ids = [args.user_id]
    else:
        with open(args.sample_users_path) as f:
            sample = json.load(f)
        user_ids = [u["User ID"] for u in sample]
        if args.num_users:
            user_ids = user_ids[: args.num_users]

    logging.info(f"Processing {len(user_ids)} user(s) with up to {args.max_workers} workers")

    results = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(process_user, uid, args.base_path, args.surgery): uid
            for uid in user_ids
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating diversity"):
            try:
                r = future.result()
                if r:
                    results.append(r)
            except Exception as e:
                logging.error(f"Error for user {futures[future]}: {e}")

    agg = aggregate(results)
    logging.info("Aggregate results:")
    for k, v in agg.items():
        logging.info(f"  {k}: {v}")

    agg_path = Path(args.base_path) / "eval_task_diversity_aggregate.json"
    agg_path.write_text(json.dumps(agg, indent=2))
    logging.info(f"Saved aggregate to {agg_path}")


if __name__ == "__main__":
    main()
