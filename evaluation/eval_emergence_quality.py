"""
eval_emergence_quality.py  —  Gap 4: Emergence quality

The core problem with evaluating emergence: there's no ground truth for
what *should* have been emergent. This script constructs a pseudo-ground-truth
from the gap between _bio_notes.md and _topics_filled.json:

  Latent emergent facts = bio_notes facts that don't map to ANY subtopic
                          in the agenda (topics_filled.json)

These are facts the user has that the interview plan didn't anticipate.
If the interview surfaces them and the system labels them as emergent, that's
genuine, high-quality emergence.

Pipeline
--------
Step 1 (classify): For each bio note, ask an LLM whether it maps to any
                   existing subtopic. Notes that don't map = latent emergent set.

Step 2 (compare):  For each detected emergent subtopic (from evaluations_emergence/),
                   check whether it corresponds to ≥1 latent emergent note.

Metrics
-------
precision : fraction of detected emergent subtopics grounded in latent facts
recall    : fraction of latent emergent notes captured by any detected subtopic
            (approximates "how much out-of-agenda content was surfaced")

Usage
-----
python evaluation/eval_emergence_quality.py \\
    --base-path logs/ \\
    --ground-truth-path data/sample_user_profiles \\
    --sample-users-path analysis/sample_users_50.json \\
    --output-dir results/emergence_quality \\
    --summary-path results/emergence_quality_summary.json
"""

import argparse
import glob
import json
import logging
import os
import re
import time
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

OPENAI_RETRIES = 3
MAX_TOKENS = 4096
MODEL_CLIENT = None
MODEL_CONFIG = None


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLASSIFY_NOTES_PROMPT = """\
You are classifying biographical facts against an interview agenda.

Given a list of numbered facts about a person and the agenda topics/subtopics
below, determine for each fact whether it is **covered by any existing subtopic**.

A fact is covered if it falls within the scope of any subtopic's description,
even loosely. Only mark a fact as NOT covered if it introduces entirely new
information that no subtopic could plausibly address.

Return JSON:
{
  "classifications": [
    {"index": 0, "covered": true,  "subtopic_id": "2.1"},
    {"index": 1, "covered": false, "subtopic_id": null}
  ]
}
"""

MATCH_EMERGENT_PROMPT = """\
You are checking whether a detected emergent subtopic is grounded in a set of
out-of-agenda facts (facts not covered by the original interview plan).

A detected subtopic is "grounded" if at least one of the latent facts below
is clearly related to it — i.e., if the detected subtopic is the kind of topic
you would naturally ask about given those facts.

Return JSON:
{
  "grounded": true or false,
  "matching_fact_indices": [list of fact indices that support this, or []]
}
"""


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def initialize_model(config_path: Optional[str] = None):
    global MODEL_CLIENT, MODEL_CONFIG
    client = OpenAI()
    if config_path is None:
        MODEL_CLIENT = client
        MODEL_CONFIG = {"provider_name": "openai", "model_name": "gpt-4.1-nano",
                        "generation_args": {"temperature": 0, "max_tokens": MAX_TOKENS}}
        return
    with open(config_path) as f:
        MODEL_CONFIG = json.load(f)
    MODEL_CLIENT = client


def call_openai(messages: List[Dict]) -> Optional[str]:
    model = MODEL_CONFIG.get("model_name", "gpt-4.1-nano")
    gen_args = MODEL_CONFIG.get("generation_args", {})
    for attempt in range(OPENAI_RETRIES):
        try:
            resp = MODEL_CLIENT.chat.completions.create(
                model=model, messages=messages, **gen_args)
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"OpenAI error (attempt {attempt+1}): {e}")
            time.sleep(61)
    return None


def safe_parse_json(text: str):
    text = text.strip()
    for pat in [r"```json(.*?)```", r"```(.*?)```"]:
        m = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        pass
    try:
        r = ast.literal_eval(text)
        return r if isinstance(r, dict) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1: Classify bio notes against agenda
# ---------------------------------------------------------------------------

def build_agenda_summary(topics_filled: List[Dict]) -> str:
    lines = ["Interview agenda subtopics:"]
    for topic in topics_filled:
        lines.append(f"\n## {topic.get('topic', '')}")
        for sub in topic.get("subtopics", []):
            lines.append(f"  {sub['subtopic_id']}: {sub['subtopic_description']}")
    return "\n".join(lines)


def classify_bio_notes(bio_notes: List[str], agenda_summary: str) -> List[Dict]:
    """
    Returns list of {index, note, covered, subtopic_id}.
    """
    facts_text = "\n".join(f"{i}. {note}" for i, note in enumerate(bio_notes))
    messages = [
        {"role": "system", "content": CLASSIFY_NOTES_PROMPT},
        {"role": "user", "content": (
            f"# Agenda\n\n{agenda_summary}\n\n"
            f"# Facts\n\n{facts_text}\n\n# Your Output\n"
        )},
    ]
    response = call_openai(messages)
    if response is None:
        return [{"index": i, "note": n, "covered": None, "subtopic_id": None, "error": True}
                for i, n in enumerate(bio_notes)]

    parsed = safe_parse_json(response)
    if parsed is None or "classifications" not in parsed:
        return [{"index": i, "note": n, "covered": None, "subtopic_id": None, "error": True}
                for i, n in enumerate(bio_notes)]

    lookup = {e["index"]: e for e in parsed["classifications"] if "index" in e}
    result = []
    for i, note in enumerate(bio_notes):
        entry = lookup.get(i, {})
        result.append({
            "index": i,
            "note": note,
            "covered": entry.get("covered"),
            "subtopic_id": entry.get("subtopic_id"),
            "error": "covered" not in entry,
        })
    return result


# ---------------------------------------------------------------------------
# Step 2: Match detected emergent subtopics to latent facts
# ---------------------------------------------------------------------------

def load_detected_emergent(base_path: str, user_id: str) -> List[Dict]:
    """
    Load all detected emergent subtopics from evaluations_emergence/.
    Uses the latest snap_eval file that has non-empty emergent_subtopics.
    """
    em_dir = os.path.join(base_path, user_id, "evaluations_emergence")
    if not os.path.isdir(em_dir):
        return []

    all_detected = []
    for path in glob.glob(os.path.join(em_dir, "snap_eval_*.json")):
        try:
            with open(path) as f:
                data = json.load(f)
            ems = data.get("emergent_subtopics", [])
            if isinstance(ems, list):
                all_detected.extend(ems)
        except Exception:
            pass

    # Deduplicate by emergent_subtopic name
    seen = set()
    deduped = []
    for e in all_detected:
        key = e.get("emergent_subtopic", "")
        if key and key not in seen:
            seen.add(key)
            deduped.append(e)
    return deduped


def match_emergent_to_latent(detected: Dict, latent_notes: List[str]) -> Dict:
    """Check if a detected emergent subtopic is grounded in latent facts."""
    if not latent_notes:
        return {"grounded": False, "matching_fact_indices": []}

    facts_text = "\n".join(f"{i}. {n}" for i, n in enumerate(latent_notes))
    subtopic_desc = detected.get("emergent_subtopic", "")
    rationale = detected.get("rationale", "")

    messages = [
        {"role": "system", "content": MATCH_EMERGENT_PROMPT},
        {"role": "user", "content": (
            f"# Detected Emergent Subtopic\n\n"
            f"Name: {subtopic_desc}\n"
            f"Rationale: {rationale}\n\n"
            f"# Out-of-Agenda Facts\n\n{facts_text}\n\n# Your Output\n"
        )},
    ]
    response = call_openai(messages)
    if response is None:
        return {"grounded": None, "matching_fact_indices": [], "error": True}

    parsed = safe_parse_json(response)
    if parsed is None or "grounded" not in parsed:
        return {"grounded": None, "matching_fact_indices": [], "error": True}

    return {
        "grounded": bool(parsed.get("grounded", False)),
        "matching_fact_indices": parsed.get("matching_fact_indices", []),
        "error": False,
    }


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_user(user_id: str, base_path: str, ground_truth_path: str,
                  output_dir: str, overwrite: bool = False) -> Optional[Dict]:
    save_path = os.path.join(output_dir, f"{user_id}.json")
    if not overwrite and os.path.exists(save_path):
        return None

    # Load bio notes
    bio_path = os.path.join(ground_truth_path, user_id, f"{user_id}_bio_notes.md")
    if not os.path.exists(bio_path):
        logging.warning(f"{user_id}: bio_notes.md not found")
        return None

    with open(bio_path) as f:
        raw = f.read()
    bio_notes = [line.lstrip("- ").strip() for line in raw.splitlines()
                 if line.strip() and not line.startswith("#")]

    # Load structured agenda
    gt_path = os.path.join(ground_truth_path, user_id, f"{user_id}_topics_filled.json")
    if not os.path.exists(gt_path):
        logging.warning(f"{user_id}: topics_filled.json not found")
        return None

    with open(gt_path) as f:
        topics_filled = json.load(f)

    agenda_summary = build_agenda_summary(topics_filled)

    # Step 1: Classify bio notes
    classifications = classify_bio_notes(bio_notes, agenda_summary)
    latent = [c for c in classifications if c.get("covered") is False and not c.get("error")]
    latent_notes = [c["note"] for c in latent]

    # Step 2: Load detected emergent subtopics
    detected = load_detected_emergent(base_path, user_id)

    # Step 3: Match each detected emergent to latent facts
    matched = []
    for em in detected:
        match = match_emergent_to_latent(em, latent_notes)
        matched.append({
            "emergent_subtopic": em.get("emergent_subtopic"),
            "topic": em.get("topic"),
            "rationale": em.get("rationale"),
            **match,
        })

    # Metrics
    n_detected = len(matched)
    n_latent   = len(latent)

    grounded_detected = [m for m in matched if m.get("grounded") is True]
    precision = len(grounded_detected) / n_detected if n_detected else None

    # Recall: which latent fact indices were matched by at least one detected subtopic
    covered_latent_indices = set()
    for m in grounded_detected:
        covered_latent_indices.update(m.get("matching_fact_indices", []))
    recall = len(covered_latent_indices) / n_latent if n_latent else None

    result = {
        "user_id": user_id,
        "summary": {
            "n_bio_notes": len(bio_notes),
            "n_latent_emergent_facts": n_latent,
            "n_detected_emergent": n_detected,
            "n_grounded_detected": len(grounded_detected),
            "precision": precision,
            "recall": recall,
        },
        "latent_facts": latent,
        "detected_emergent": matched,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Emergence quality evaluation (Gap 4)")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--ground-truth-path", default="data/sample_user_profiles")
    parser.add_argument("--sample-users-path", default="analysis/sample_users_50.json")
    parser.add_argument("--output-dir", default="results/emergence_quality")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--model-config", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    initialize_model(args.model_config)

    with open(args.sample_users_path) as f:
        sample_users = json.load(f)
    user_ids = [u["User ID"] for u in sample_users[: args.num_users]]

    all_results = []

    def _run(uid):
        return evaluate_user(uid, args.base_path, args.ground_truth_path,
                             args.output_dir, args.overwrite)

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, uid): uid for uid in user_ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Evaluating emergence quality"):
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

    precs = [r["summary"]["precision"] for r in all_results if r["summary"]["precision"] is not None]
    recs  = [r["summary"]["recall"]    for r in all_results if r["summary"]["recall"]    is not None]

    summary = {
        "num_users": len(all_results),
        "mean_precision": sum(precs) / len(precs) if precs else None,
        "mean_recall":    sum(recs)  / len(recs)  if recs  else None,
        "users_with_detections": sum(1 for r in all_results if r["summary"]["n_detected_emergent"] > 0),
    }
    logging.info(
        f"Done. Users={summary['num_users']}  "
        f"Precision={summary['mean_precision']}  "
        f"Recall={summary['mean_recall']}"
    )

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump({"summary": summary, "per_user": all_results}, f, indent=2)
        logging.info(f"Summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
