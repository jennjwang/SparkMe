"""
eval_profile_extraction.py

End-to-end evaluation pipeline:

  Ground truth (topics_filled.json)
        ↓  [user simulator uses this as factual basis]
  Interview session
        ↓
  Extract structured profile from transcript  ← THIS SCRIPT
        ↓
  Compare extracted profile to ground truth
        ↓
  Per-subtopic recall metrics

Usage:
    python evaluation/eval_profile_extraction.py \
        --mode sparkme \
        --base-path logs/ \
        --ground-truth-path data/sample_user_profiles \
        --output results/profile_extraction.json

Modes: sparkme | storysage | llmroleplay | freeform
"""

import argparse
import json
import logging
import os
import re
import ast
import time
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

OPENAI_RETRIES = 3
MAX_TOKENS = 8192
MODEL_CLIENT = None
MODEL_CONFIG = None


# ---------------------------------------------------------------------------
# LLM prompts
# ---------------------------------------------------------------------------

EXTRACT_PROFILE_SYSTEM_PROMPT = """\
You are extracting a structured profile from an interview transcript.

The interview covers a worker's role, tasks, and relationship with AI tools.
You are given a list of subtopics. For each subtopic, extract ALL facts that were
**explicitly stated or clearly conveyed** in the transcript. Do NOT infer or add details
not present in the conversation.

Return JSON with this exact structure:
{
  "subtopics": {
    "<subtopic_id>": {
      "description": "<subtopic_description>",
      "extracted_facts": ["<fact1>", "<fact2>", ...]
    },
    ...
  }
}

If nothing was discussed for a subtopic, use an empty list for extracted_facts.
"""

RECALL_JUDGE_SYSTEM_PROMPT = """\
You are evaluating whether specific ground truth facts are captured in an extracted profile.

For each numbered fact, determine whether it is **explicitly stated or clearly conveyed**
in the extracted profile facts. A fact counts as recalled if:
- The core information is present (exact wording is not required).
- Paraphrasing or summarization that preserves the meaning counts as recalled.

A fact is NOT recalled if:
- Only the general topic is mentioned without the specific detail.
- The information must be inferred rather than being stated.

Return JSON:
{
  "facts": [
    {"index": 0, "recalled": true,  "evidence": "<brief quote or paraphrase>"},
    {"index": 1, "recalled": false, "evidence": ""}
  ]
}
"""


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------

def initialize_model(config_path: Optional[str] = None):
    global MODEL_CLIENT, MODEL_CONFIG

    openai_client = OpenAI()

    if config_path is None:
        MODEL_CLIENT = openai_client
        MODEL_CONFIG = {
            "provider_name": "openai",
            "model_name": "gpt-4.1-nano",
            "generation_args": {"temperature": 0, "max_tokens": MAX_TOKENS},
        }
        logging.info("Using default OpenAI gpt-4.1-nano")
        return

    with open(config_path) as f:
        MODEL_CONFIG = json.load(f)

    if MODEL_CONFIG.get("provider_name", "openai") == "openai":
        MODEL_CLIENT = openai_client
    else:
        raise ValueError("Only openai provider is supported in this script.")


def call_openai(messages: List[Dict]) -> Optional[str]:
    model = MODEL_CONFIG.get("model_name", "gpt-4.1-nano")
    gen_args = MODEL_CONFIG.get("generation_args", {})
    for attempt in range(OPENAI_RETRIES):
        try:
            resp = MODEL_CLIENT.chat.completions.create(
                model=model, messages=messages, **gen_args
            )
            return resp.choices[0].message.content
        except Exception as e:
            logging.warning(f"OpenAI error (attempt {attempt + 1}): {e}")
            time.sleep(61)
    return None


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def safe_parse_json(text: str) -> Optional[Dict]:
    text = text.strip()
    if not text:
        return None
    for pattern in [r"```json(.*?)```", r"```(.*?)```"]:
        m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def load_transcript_sparkme(base_path: str, user_id: str) -> Optional[str]:
    """Load the chat_history.log from the latest sparkme session."""
    exec_dir = os.path.join(base_path, user_id, "execution_logs")
    if not os.path.isdir(exec_dir):
        return None

    # Find the highest-numbered session directory that has a chat_history.log
    transcript = None
    for entry in sorted(os.listdir(exec_dir)):
        candidate = os.path.join(exec_dir, entry, "chat_history.log")
        if os.path.exists(candidate):
            transcript = candidate   # keep updating to get the last one

    if transcript is None:
        return None

    lines = []
    with open(transcript) as f:
        for line in f:
            # Format: "timestamp - INFO - Role: message"
            m = re.match(r"[\d\-:, ]+ - \w+ - (Interviewer|User): (.+)", line.strip())
            if m:
                lines.append(f"{m.group(1)}: {m.group(2)}")

    return "\n".join(lines) if lines else None


def load_transcript_jsonl(base_path: str, user_id: str) -> Optional[str]:
    """Load interview_log.jsonl (llmroleplay / freeform modes)."""
    path = os.path.join(base_path, user_id, "interview_log.jsonl")
    if not os.path.exists(path):
        return None
    lines = []
    with open(path) as f:
        for raw in f:
            obj = json.loads(raw)
            if obj.get("user_message"):
                lines.append(f"User: {obj['user_message']}")
            if obj.get("assistant_message"):
                lines.append(f"Interviewer: {obj['assistant_message']}")
    return "\n".join(lines) if lines else None


def load_transcript(base_path: str, user_id: str, mode: str) -> Optional[str]:
    if mode in ("sparkme", "storysage"):
        return load_transcript_sparkme(base_path, user_id)
    else:
        return load_transcript_jsonl(base_path, user_id)


# ---------------------------------------------------------------------------
# Ground truth helpers
# ---------------------------------------------------------------------------

def build_subtopic_schema(ground_truth: List[Dict]) -> Dict[str, Dict]:
    """Return {subtopic_id: {description, notes}} from topics_filled.json."""
    schema = {}
    for topic in ground_truth:
        for sub in topic.get("subtopics", []):
            schema[sub["subtopic_id"]] = {
                "description": sub["subtopic_description"],
                "notes": sub.get("notes", []),
            }
    return schema


# ---------------------------------------------------------------------------
# Step 1: Extract profile from transcript
# ---------------------------------------------------------------------------

def extract_profile(transcript: str, ground_truth: List[Dict]) -> Optional[Dict]:
    """Ask the LLM to extract per-subtopic facts from the transcript."""
    schema = build_subtopic_schema(ground_truth)

    schema_lines = []
    for sid, info in sorted(schema.items()):
        schema_lines.append(f'  "{sid}": "{info["description"]}"')
    schema_text = "{\n" + ",\n".join(schema_lines) + "\n}"

    user_content = (
        "# Subtopic Schema\n\n"
        f"{schema_text}\n\n"
        "# Interview Transcript\n\n"
        f"<transcript>\n{transcript}\n</transcript>\n\n"
        "# Your Output\n"
    )

    messages = [
        {"role": "system", "content": EXTRACT_PROFILE_SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]

    response = call_openai(messages)
    if response is None:
        return None
    return safe_parse_json(response)


# ---------------------------------------------------------------------------
# Step 2: Compare extracted profile to ground truth
# ---------------------------------------------------------------------------

def recall_for_subtopic(
    subtopic_id: str,
    gt_notes: List[str],
    extracted_facts: List[str],
) -> List[Dict]:
    """Check which GT notes are recalled in the extracted facts for one subtopic."""
    if not gt_notes:
        return []

    facts_text = "\n".join(f"{i}. {note}" for i, note in enumerate(gt_notes))
    extracted_text = "\n".join(f"- {f}" for f in extracted_facts) if extracted_facts else "(nothing extracted)"

    messages = [
        {"role": "system", "content": RECALL_JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "# Ground Truth Facts\n\n"
                f"{facts_text}\n\n"
                "# Extracted Profile Facts\n\n"
                f"{extracted_text}\n\n"
                "# Your Output\n"
            ),
        },
    ]

    response = call_openai(messages)
    if response is None:
        return [{"subtopic_id": subtopic_id, "fact_index": i, "fact": n,
                 "recalled": None, "evidence": "", "error": True}
                for i, n in enumerate(gt_notes)]

    parsed = safe_parse_json(response)
    if parsed is None or "facts" not in parsed:
        return [{"subtopic_id": subtopic_id, "fact_index": i, "fact": n,
                 "recalled": None, "evidence": "", "error": True}
                for i, n in enumerate(gt_notes)]

    lookup = {e["index"]: e for e in parsed["facts"] if "index" in e}
    results = []
    for i, note in enumerate(gt_notes):
        entry = lookup.get(i)
        if entry is None:
            results.append({"subtopic_id": subtopic_id, "fact_index": i, "fact": note,
                             "recalled": None, "evidence": "", "error": True})
        else:
            results.append({"subtopic_id": subtopic_id, "fact_index": i, "fact": note,
                             "recalled": bool(entry.get("recalled", False)),
                             "evidence": entry.get("evidence", ""),
                             "error": False})
    return results


# ---------------------------------------------------------------------------
# Step 3: Compute metrics
# ---------------------------------------------------------------------------

def compute_metrics(fact_results: List[Dict], ground_truth: List[Dict]) -> Dict:
    valid = [r for r in fact_results if not r["error"]]
    total = len(valid)
    recalled = sum(1 for r in valid if r["recalled"])

    topic_names = {}
    subtopic_descs = {}
    for topic in ground_truth:
        for sub in topic.get("subtopics", []):
            tid = sub["subtopic_id"].split(".")[0]
            topic_names[tid] = topic.get("topic", "")
            subtopic_descs[sub["subtopic_id"]] = sub["subtopic_description"]

    per_topic: Dict[str, Dict] = {}
    per_subtopic: Dict[str, Dict] = {}

    for r in valid:
        tid = r["subtopic_id"].split(".")[0]
        per_topic.setdefault(tid, {"topic": topic_names.get(tid, ""), "total": 0, "recalled": 0})
        per_topic[tid]["total"] += 1
        if r["recalled"]:
            per_topic[tid]["recalled"] += 1

        sid = r["subtopic_id"]
        per_subtopic.setdefault(sid, {"description": subtopic_descs.get(sid, ""), "total": 0, "recalled": 0})
        per_subtopic[sid]["total"] += 1
        if r["recalled"]:
            per_subtopic[sid]["recalled"] += 1

    for d in per_topic.values():
        d["recall"] = d["recalled"] / d["total"] if d["total"] else 0.0
    for d in per_subtopic.values():
        d["recall"] = d["recalled"] / d["total"] if d["total"] else 0.0

    return {
        "summary": {
            "total_gt_facts": len(fact_results),
            "evaluated_facts": total,
            "recalled_facts": recalled,
            "recall": recalled / total if total else 0.0,
            "error_count": sum(1 for r in fact_results if r["error"]),
        },
        "per_topic": per_topic,
        "per_subtopic": per_subtopic,
        "fact_details": fact_results,
    }


# ---------------------------------------------------------------------------
# Per-user evaluation
# ---------------------------------------------------------------------------

def evaluate_user(user_id: str, base_path: str, ground_truth_path: str,
                  mode: str, output_dir: str, overwrite: bool = False) -> Optional[Dict]:
    save_path = os.path.join(output_dir, f"{user_id}.json")
    if not overwrite and os.path.exists(save_path):
        return None  # already done

    gt_file = os.path.join(ground_truth_path, user_id, f"{user_id}_topics_filled.json")
    if not os.path.exists(gt_file):
        logging.warning(f"{user_id}: ground truth not found at {gt_file}")
        return None

    with open(gt_file) as f:
        ground_truth = json.load(f)

    transcript = load_transcript(base_path, user_id, mode)
    if not transcript:
        logging.warning(f"{user_id}: no transcript found")
        return None

    # Step 1: Extract profile
    extracted = extract_profile(transcript, ground_truth)
    if extracted is None or "subtopics" not in extracted:
        logging.warning(f"{user_id}: profile extraction failed")
        extracted_subtopics = {}
    else:
        extracted_subtopics = extracted["subtopics"]

    # Step 2: Compare to ground truth per subtopic
    schema = build_subtopic_schema(ground_truth)
    all_fact_results = []

    for sid, info in sorted(schema.items()):
        extracted_entry = extracted_subtopics.get(sid, {})
        extracted_facts = extracted_entry.get("extracted_facts", []) if isinstance(extracted_entry, dict) else []
        results = recall_for_subtopic(sid, info["notes"], extracted_facts)
        all_fact_results.extend(results)

    # Step 3: Compute metrics
    metrics = compute_metrics(all_fact_results, ground_truth)
    metrics["user_id"] = user_id
    metrics["extracted_profile"] = extracted_subtopics

    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Profile extraction evaluation")
    parser.add_argument("--mode", required=True,
                        choices=["sparkme", "storysage", "llmroleplay", "freeform"])
    parser.add_argument("--base-path", required=True,
                        help="Base path to logs directory")
    parser.add_argument("--ground-truth-path", default="data/sample_user_profiles",
                        help="Directory containing per-user ground truth JSON files")
    parser.add_argument("--sample-users-path", default="analysis/sample_users_50.json",
                        help="JSON list of users to evaluate")
    parser.add_argument("--output-dir", default="results/profile_extraction",
                        help="Directory to write per-user result JSON files")
    parser.add_argument("--summary-path", default=None,
                        help="Optional path to write aggregated summary JSON")
    parser.add_argument("--model-config", default=None,
                        help="Path to model config JSON (defaults to OpenAI gpt-4.1-nano)")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-evaluate even if result already exists")
    args = parser.parse_args()

    initialize_model(args.model_config)

    with open(args.sample_users_path) as f:
        sample_users = json.load(f)
    user_ids = [u["User ID"] for u in sample_users[: args.num_users]]

    all_results = []

    def _run(uid):
        return evaluate_user(
            user_id=uid,
            base_path=args.base_path,
            ground_truth_path=args.ground_truth_path,
            mode=args.mode,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, uid): uid for uid in user_ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=f"Evaluating ({args.mode})"):
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

    # Aggregate summary
    recalls = [r["summary"]["recall"] for r in all_results]
    summary = {
        "mode": args.mode,
        "num_users": len(all_results),
        "mean_recall": sum(recalls) / len(recalls),
        "min_recall": min(recalls),
        "max_recall": max(recalls),
    }
    logging.info(
        f"Done. Users={summary['num_users']}  "
        f"Mean recall={summary['mean_recall']:.3f}  "
        f"[{summary['min_recall']:.3f}, {summary['max_recall']:.3f}]"
    )

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        agg = {"summary": summary, "per_user": all_results}
        with open(args.summary_path, "w") as f:
            json.dump(agg, f, indent=2)
        logging.info(f"Summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
