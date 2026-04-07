import argparse
import json
import logging
import time
import os
import re
import ast
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from openai import OpenAI
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

# Constants
OPENAI_RETRIES = 3
MAX_TOKENS = 8192
OPENAI_CLIENT = None

# Global model client (will be initialized based on config)
MODEL_CLIENT = None
MODEL_CONFIG = None
TOKENIZER = None

# Import note extraction functions from eval_coverage
from eval_coverage import (
    extract_all_notes_interviewer,
    extract_all_notes_storysage,
    extract_all_notes_llmroleplay,
    extract_all_notes_freeform,
    EvaluationConfig,
    safe_parse_json,
)

JUDGE_PROMPT = """# Instruction
You are evaluating whether specific ground truth facts were captured in interview notes.

For each numbered fact below, determine whether it is **explicitly stated or clearly conveyed** in the interview notes. A fact counts as recalled if:
- The core information is present (exact wording is not required).
- Components may be spread across different parts of the notes.
- Paraphrasing or summarization that preserves the meaning counts as recalled.

A fact is NOT recalled if:
- Only the general topic is mentioned without the specific detail.
- The information must be inferred rather than being stated.
- The fact is contradicted in the notes.

# Output Format (JSON)
Return a JSON object with a "facts" array. Each entry must have:
- "index": the fact number (integer)
- "recalled": true or false
- "evidence": if recalled, a brief quote or paraphrase from the notes that supports it. If not recalled, an empty string.

Example:
```json
{
    "facts": [
        {"index": 0, "recalled": true, "evidence": "the notes mention that..."},
        {"index": 1, "recalled": false, "evidence": ""}
    ]
}
```
"""


def initialize_model(config_path: Optional[str] = None):
    """Initialize model client based on configuration"""
    global MODEL_CLIENT, MODEL_CONFIG, TOKENIZER, OPENAI_CLIENT

    OPENAI_CLIENT = OpenAI()

    if config_path is None:
        MODEL_CLIENT = OPENAI_CLIENT
        MODEL_CONFIG = {
            "provider_name": "openai",
            "model_name": "gpt-4.1-nano",
            "generation_args": {
                "temperature": 0,
                "max_tokens": MAX_TOKENS
            }
        }
        logging.info("Using default OpenAI client")
        return

    with open(config_path, 'r') as f:
        MODEL_CONFIG = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "openai":
        MODEL_CLIENT = OPENAI_CLIENT
        logging.info(f"Using OpenAI client with model {MODEL_CONFIG.get('model_name', 'gpt-4.1-nano')}")
    elif provider == "local":
        try:
            from vllm import LLM
            model_name = MODEL_CONFIG["model_name"]
            model_args = MODEL_CONFIG.get("model_args", {})
            logging.info(f"Initializing vLLM with model {model_name}")
            MODEL_CLIENT = LLM(model=model_name, **model_args)
            logging.info(f"Loading tokenizer for {model_name}")
            TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            logging.info("vLLM client and tokenizer initialized successfully")
        except ImportError:
            logging.error("vLLM not installed. Please install with: pip install vllm")
            raise
    else:
        raise ValueError(f"Unknown provider: {provider}")


def request_openai_completion(msg: List[Dict]) -> str:
    """Request completion from OpenAI with retries"""
    global MODEL_CLIENT, MODEL_CONFIG

    model_name = MODEL_CONFIG.get("model_name", "gpt-4.1-nano")
    gen_args = MODEL_CONFIG.get("generation_args", {})

    for attempt in range(OPENAI_RETRIES):
        try:
            response = MODEL_CLIENT.chat.completions.create(
                model=model_name,
                messages=msg,
                **gen_args
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.warning(f"Error calling OpenAI API (attempt {attempt + 1}/{OPENAI_RETRIES}): {e}")
            time.sleep(61)

    logging.error(f"Could not resolve error after {OPENAI_RETRIES} attempts")
    return None


def flatten_ground_truth(ground_truth: List[Dict]) -> List[Dict]:
    """Flatten topics_filled.json into individual facts with metadata.

    Returns list of dicts with keys: topic, topic_id, subtopic_id, subtopic_description, fact_index, fact
    """
    facts = []
    for topic_data in ground_truth:
        topic_name = topic_data.get("topic", "")
        for subtopic in topic_data.get("subtopics", []):
            subtopic_id = subtopic["subtopic_id"]
            topic_id = subtopic_id.split(".")[0]
            subtopic_desc = subtopic["subtopic_description"]
            for i, note in enumerate(subtopic.get("notes", [])):
                facts.append({
                    "topic": topic_name,
                    "topic_id": topic_id,
                    "subtopic_id": subtopic_id,
                    "subtopic_description": subtopic_desc,
                    "fact_index": i,
                    "fact": note,
                })
    return facts


def group_facts_by_subtopic(facts: List[Dict]) -> Dict[str, List[Dict]]:
    """Group flat facts by subtopic_id for batched evaluation."""
    groups = {}
    for fact in facts:
        sid = fact["subtopic_id"]
        if sid not in groups:
            groups[sid] = []
        groups[sid].append(fact)
    return groups


def create_fact_recall_messages(
    ground_truth: List[Dict],
    session_data,
    mode: str,
    snapshot_idx: int = None,
) -> Tuple[List[List[Dict]], List[List[Dict]]]:
    """Create per-subtopic evaluation messages for fact-level recall.

    Returns:
        Tuple of (list of chat messages, list of fact groups) where each entry
        corresponds to one subtopic batch.
    """
    all_facts = flatten_ground_truth(ground_truth)
    grouped = group_facts_by_subtopic(all_facts)

    # Extract ALL notes from the session
    if mode == 'sparkme':
        all_notes = extract_all_notes_interviewer(session_data)
    elif mode == 'storysage':
        all_notes = extract_all_notes_storysage(session_data, snapshot_idx)
    elif mode == 'llmroleplay':
        all_notes = extract_all_notes_llmroleplay(session_data, snapshot_idx)
    elif mode == 'freeform':
        all_notes = extract_all_notes_freeform(session_data, snapshot_idx)
    else:
        raise ValueError(f"Invalid mode: {mode}")

    if not all_notes or len(all_notes.strip()) == 0:
        return [], []

    messages_list = []
    fact_groups = []

    for subtopic_id, facts in sorted(grouped.items()):
        # Format the facts as a numbered list
        facts_text = "\n".join(
            f"{i}. [{facts[i]['subtopic_description']}] {facts[i]['fact']}"
            for i in range(len(facts))
        )

        msg = [
            {"role": "system", "content": JUDGE_PROMPT},
            {
                "role": "user",
                "content": (
                    "# Ground Truth Facts to Check\n\n"
                    f"{facts_text}\n\n"
                    "# Interview Notes\n\n"
                    f"{all_notes}\n\n"
                    "# Your Output\n"
                ),
            },
        ]

        messages_list.append(msg)
        fact_groups.append(facts)

    return messages_list, fact_groups


def parse_fact_recall_response(response_text: str, facts: List[Dict]) -> List[Dict]:
    """Parse the LLM response into per-fact recall results."""
    parsed = safe_parse_json(response_text)

    if parsed is None or "facts" not in parsed:
        # Return all as errors
        return [
            {
                "subtopic_id": f["subtopic_id"],
                "fact_index": f["fact_index"],
                "fact": f["fact"],
                "recalled": None,
                "evidence": "",
                "error": True,
            }
            for f in facts
        ]

    # Build lookup from response
    response_lookup = {}
    for entry in parsed["facts"]:
        idx = entry.get("index")
        if idx is not None:
            response_lookup[idx] = entry

    results = []
    for i, f in enumerate(facts):
        entry = response_lookup.get(i)
        if entry is None:
            results.append({
                "subtopic_id": f["subtopic_id"],
                "fact_index": f["fact_index"],
                "fact": f["fact"],
                "recalled": None,
                "evidence": "",
                "error": True,
            })
        else:
            results.append({
                "subtopic_id": f["subtopic_id"],
                "fact_index": f["fact_index"],
                "fact": f["fact"],
                "recalled": bool(entry.get("recalled", False)),
                "evidence": entry.get("evidence", ""),
                "error": False,
            })

    return results


def compute_metrics(fact_results: List[Dict], ground_truth: List[Dict]) -> Dict:
    """Compute overall, per-topic, and per-subtopic recall metrics."""
    valid = [r for r in fact_results if r["error"] is False]
    total = len(valid)
    recalled = sum(1 for r in valid if r["recalled"])

    summary = {
        "total_gt_facts": len(fact_results),
        "evaluated_facts": total,
        "recalled_facts": recalled,
        "recall": recalled / total if total > 0 else 0.0,
        "error_count": sum(1 for r in fact_results if r["error"]),
    }

    # Per-topic
    per_topic = {}
    topic_names = {}
    for topic_data in ground_truth:
        for subtopic in topic_data.get("subtopics", []):
            tid = subtopic["subtopic_id"].split(".")[0]
            topic_names[tid] = topic_data.get("topic", "")

    for r in valid:
        tid = r["subtopic_id"].split(".")[0]
        if tid not in per_topic:
            per_topic[tid] = {"topic": topic_names.get(tid, ""), "total": 0, "recalled": 0}
        per_topic[tid]["total"] += 1
        if r["recalled"]:
            per_topic[tid]["recalled"] += 1

    for tid, data in per_topic.items():
        data["recall"] = data["recalled"] / data["total"] if data["total"] > 0 else 0.0

    # Per-subtopic
    per_subtopic = {}
    subtopic_descs = {}
    for topic_data in ground_truth:
        for subtopic in topic_data.get("subtopics", []):
            subtopic_descs[subtopic["subtopic_id"]] = subtopic["subtopic_description"]

    for r in valid:
        sid = r["subtopic_id"]
        if sid not in per_subtopic:
            per_subtopic[sid] = {
                "description": subtopic_descs.get(sid, ""),
                "total": 0,
                "recalled": 0,
            }
        per_subtopic[sid]["total"] += 1
        if r["recalled"]:
            per_subtopic[sid]["recalled"] += 1

    for sid, data in per_subtopic.items():
        data["recall"] = data["recalled"] / data["total"] if data["total"] > 0 else 0.0

    return {
        "summary": summary,
        "per_topic": per_topic,
        "per_subtopic": per_subtopic,
        "fact_details": fact_results,
    }


def prepare_session_data(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """Prepare session data and paths for fact-level recall evaluation."""
    if config.mode == 'sparkme':
        session_path = f"{config.base_path}/{user_id}/execution_logs/session_0/session_agenda_snap_{snapshot_idx}.json"
    elif config.mode == 'storysage':
        session_path = f"{config.base_path}/{user_id}/memory_bank_content.json"
    elif config.mode in ('llmroleplay', 'freeform'):
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    eval_dir = f"{config.base_path}/{user_id}/evaluations_fact_recall"
    save_eval_path = f"{eval_dir}/snap_eval_{snapshot_idx}.json"

    if not os.path.exists(session_path):
        return None

    if not surgery and os.path.exists(save_eval_path):
        with open(save_eval_path, 'r') as f:
            existing = json.load(f)
            if len(existing) > 0:
                return None

    # Load session data
    if config.mode in ('llmroleplay', 'freeform'):
        history = []
        with open(session_path) as f:
            for line in f.readlines():
                history.append(json.loads(line))
        if snapshot_idx >= len(history) + 4:
            return None
        session_data = history
    else:
        with open(session_path) as f:
            session_data = json.load(f)

    messages_list, fact_groups = create_fact_recall_messages(
        ground_truth, session_data, config.mode, snapshot_idx
    )

    if len(messages_list) == 0:
        return None

    return {
        'user_id': user_id,
        'snapshot_idx': snapshot_idx,
        'messages_list': messages_list,
        'fact_groups': fact_groups,
        'save_path': save_eval_path,
        'eval_dir': eval_dir,
        'ground_truth': ground_truth,
    }


def process_session_openai(user_id: str, snapshot_idx: int, ground_truth: List[Dict], config: EvaluationConfig, surgery: bool = False):
    """Process a single session snapshot with OpenAI."""
    session_info = prepare_session_data(user_id, snapshot_idx, ground_truth, config, surgery=surgery)

    if session_info is None:
        return

    all_fact_results = []

    for msg, facts in zip(session_info['messages_list'], session_info['fact_groups']):
        response = request_openai_completion(msg)
        if response is None:
            # Mark all facts in this batch as errors
            for f in facts:
                all_fact_results.append({
                    "subtopic_id": f["subtopic_id"],
                    "fact_index": f["fact_index"],
                    "fact": f["fact"],
                    "recalled": None,
                    "evidence": "",
                    "error": True,
                })
        else:
            results = parse_fact_recall_response(response, facts)
            all_fact_results.extend(results)

    # Compute metrics and save
    output = compute_metrics(all_fact_results, session_info['ground_truth'])

    os.makedirs(session_info['eval_dir'], exist_ok=True)
    with open(session_info['save_path'], 'w') as f:
        json.dump(output, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description='Fact-level recall evaluation for interview sessions')
    parser.add_argument('--mode', type=str, required=True, choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'],
                        help='Evaluation mode')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--sample-users-path', type=str,
                        default='analysis/sample_users_50.json',
                        help='Path to sample users JSON')
    parser.add_argument('--ground-truth-path', type=str,
                        default='data/sample_user_profiles',
                        help='Path to ground truth data directory')
    parser.add_argument('--model-config', type=str, default=None,
                        help='Path to model config JSON file (optional, defaults to OpenAI)')
    parser.add_argument('--max-workers', type=int, default=16,
                        help='Maximum number of parallel workers')
    parser.add_argument('--num-users', type=int, default=30,
                        help='Number of users to process')
    parser.add_argument('--snapshot-start', type=int, default=1,
                        help='Starting snapshot index')
    parser.add_argument('--snapshot-end', type=int, default=80,
                        help='Ending snapshot index')
    parser.add_argument('--snapshot-step', type=int, default=1,
                        help='Snapshot step size')
    parser.add_argument('--surgery', action="store_true", help="Re-evaluate even if results exist")

    args = parser.parse_args()

    initialize_model(args.model_config)

    config = EvaluationConfig(
        mode=args.mode,
        base_path=args.base_path,
        sample_users_path=args.sample_users_path,
        ground_truth_path=args.ground_truth_path,
        max_workers=args.max_workers
    )

    with open(config.sample_users_path, 'r') as f:
        sample_users = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "local":
        logging.info("Using vLLM - collecting all sessions for batch inference")

        # Step 1: Prepare all session data
        all_session_infos = []
        for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
            user_id = person_profile["User ID"]
            gt_path = f"{config.ground_truth_path}/{user_id}/{user_id}_topics_filled.json"
            with open(gt_path) as f:
                ground_truth = json.load(f)

            for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                session_info = prepare_session_data(user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                if session_info is not None:
                    all_session_infos.append(session_info)

        logging.info(f"Prepared {len(all_session_infos)} sessions for evaluation")

        # Step 2: Collect ALL messages and metadata
        all_messages = []
        all_metadata = []

        for session_info in all_session_infos:
            for msg, facts in zip(session_info['messages_list'], session_info['fact_groups']):
                all_messages.append(msg)
                all_metadata.append({
                    'session_info': session_info,
                    'facts': facts,
                })

        logging.info(f"Total prompts to evaluate: {len(all_messages)}")

        # Step 3: Batch inference
        from vllm import SamplingParams

        prompts = []
        for msg in tqdm(all_messages, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        # Step 4: Parse results and group by session
        logging.info("Parsing results and grouping by session...")
        session_results = {}  # {(user_id, snapshot_idx): {'fact_results': [...], ...}}

        for metadata, output in tqdm(zip(all_metadata, outputs), total=len(outputs), desc="Processing outputs"):
            session_info = metadata['session_info']
            facts = metadata['facts']
            session_key = (session_info['user_id'], session_info['snapshot_idx'])

            generated_text = output.outputs[0].text.strip()
            results = parse_fact_recall_response(generated_text, facts)

            if session_key not in session_results:
                session_results[session_key] = {
                    'fact_results': [],
                    'session_info': session_info,
                }
            session_results[session_key]['fact_results'].extend(results)

        # Step 5: Compute metrics and write results
        logging.info(f"Writing results to {len(session_results)} files...")
        for session_key, data in tqdm(session_results.items(), desc="Writing results"):
            si = data['session_info']
            output = compute_metrics(data['fact_results'], si['ground_truth'])
            os.makedirs(si['eval_dir'], exist_ok=True)
            with open(si['save_path'], 'w') as f:
                json.dump(output, f, indent=2)

        logging.info("vLLM batch evaluation complete!")

    else:
        logging.info("Using OpenAI - processing sessions in parallel with threading")

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []

            for person_profile in sample_users[:args.num_users]:
                user_id = person_profile["User ID"]

                gt_path = f"{config.ground_truth_path}/{user_id}/{user_id}_topics_filled.json"
                with open(gt_path) as f:
                    ground_truth = json.load(f)

                for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                    futures.append(
                        executor.submit(process_session_openai, user_id, snapshot_idx, ground_truth, config, surgery=args.surgery)
                    )

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating fact recall ({config.mode} mode)"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing session: {e}")


if __name__ == "__main__":
    main()
