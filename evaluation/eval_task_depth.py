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

# The 10 dimensions to evaluate per task
TASK_DIMENSIONS = ["what", "duration", "tools", "who", "when", "method", "judgment", "quality_criteria", "skills", "information_sources"]

JUDGE_PROMPT = """# Instruction
You are evaluating interview notes to assess the **depth of task descriptions** captured during a work-activity interview.

Your job:
1. Identify each distinct **task** mentioned in the interview notes.
2. For each task, determine which of the following **10 dimensions** are explicitly stated or clearly inferable from the notes:
   - **what**: What the task is (a description or name of the activity)
   - **duration**: How long the task takes or what fraction of time it occupies
   - **tools**: What software, platforms, or tools are used for the task
   - **who**: Who is involved (collaborators, clients, manager, etc.)
   - **when**: When the task is done (daily, weekly, ad-hoc, specific timing, etc.)
   - **method**: How the task is carried out — the process, sequence, or technique, beyond simply restating the action
   - **judgment**: Where the worker exercises discretion — any decision point requiring evaluation, comparison, prioritization, diagnosis, or selection, and the basis for that decision
   - **quality_criteria**: What counts as done and done well — the standards, checks, thresholds, or completion conditions used to evaluate the output
   - **skills**: Specific skills, knowledge areas, or capabilities the worker draws on to perform the task
   - **information_sources**: Where guidance, inputs, or reference material comes from — policies, procedures, data feeds, prior cases, or people the worker draws on

Rules:
- Count a dimension as present only if it is **explicitly stated** — do not infer.
- If the notes mention a task but give no details, only "what" is covered.
- Each task's **depth_score** = number of dimensions present (0–10). Since "what" is always present if the task is identified, the minimum meaningful score is 1.
- Report one entry per distinct task. If multiple tasks are described together, split them.

# Output Format (JSON)
{
  "tasks": [
    {
      "task_name": "<short label for the task>",
      "dimensions": {
        "what": true,
        "duration": true or false,
        "tools": true or false,
        "who": true or false,
        "when": true or false,
        "method": true or false,
        "judgment": true or false,
        "quality_criteria": true or false,
        "skills": true or false,
        "information_sources": true or false
      },
      "depth_score": <integer 1-10>
    }
  ],
  "num_tasks": <integer>,
  "avg_depth_score": <float, average depth_score across all tasks>
}

If no tasks are found in the notes, return:
{"tasks": [], "num_tasks": 0, "avg_depth_score": 0}
"""


class EvaluationConfig:
    """Configuration for evaluation run"""
    def __init__(self, mode: str, base_path: str, sample_users_path: str,
                 max_workers: int = 16):
        self.mode = mode
        self.base_path = base_path
        self.sample_users_path = sample_users_path
        self.max_workers = max_workers

        if mode not in ['sparkme', 'storysage', 'llmroleplay', 'freeform']:
            raise ValueError(f"Invalid mode: {mode}. Must be 'sparkme', 'storysage', 'llmroleplay', or 'freeform'")


def initialize_model(config_path: Optional[str] = None):
    """Initialize model client based on configuration"""
    global MODEL_CLIENT, MODEL_CONFIG, TOKENIZER

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
        except Exception as e:
            logging.error(f"Error initializing vLLM: {e}")
            raise
    else:
        raise ValueError(f"Unknown provider: {provider}")


def safe_parse_json(text: str):
    """Parse JSON from various formats including code blocks"""
    text = text.strip()
    if not text:
        return None

    codeblock_match = re.search(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if codeblock_match:
        candidate = codeblock_match.group(1).strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    try:
        parsed_dict = ast.literal_eval(text)
        if isinstance(parsed_dict, dict):
            return parsed_dict
    except Exception:
        pass

    return None


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


# ---------------------------------------------------------------------------
# Note extraction helpers — Task Inventory only
# ---------------------------------------------------------------------------

TASK_INVENTORY_KEYWORDS = ["task inventory", "task", "activities", "deliverables", "time allocation"]


def _is_task_inventory_topic(description: str) -> bool:
    """Return True if the topic description looks like Task Inventory."""
    desc_lower = description.lower()
    return "task inventory" in desc_lower or (
        "task" in desc_lower and "inventor" in desc_lower
    ) or desc_lower.strip() == "task inventory"


def extract_task_inventory_notes_sparkme(session_agenda: Dict) -> str:
    """Extract notes for the Task Inventory topic from a SparkMe session agenda."""
    parts = []

    for core_topic_id, core_topic in session_agenda["interview_topic_manager"]["core_topic_dict"].items():
        topic_desc = core_topic.get("description", "")
        if not _is_task_inventory_topic(topic_desc):
            continue

        for subto_id, subto in core_topic.get("required_subtopics", {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")

            if final_summary:
                parts.append(f"**Subtopic: {subto_desc}**\nSummary: {final_summary}")
            elif notes:
                notes_str = "\n - ".join(notes)
                parts.append(f"**Subtopic: {subto_desc}**\n - {notes_str}")

        for subto_id, subto in core_topic.get("emergent_subtopics", {}).items():
            subto_desc = subto.get("description", "")
            notes = list(set(subto.get("notes", [])))
            final_summary = subto.get("final_summary", "")

            if final_summary:
                parts.append(f"**Subtopic: {subto_desc}**\nSummary: {final_summary}")
            elif notes:
                notes_str = "\n - ".join(notes)
                parts.append(f"**Subtopic: {subto_desc}**\n - {notes_str}")

    return "\n\n".join(parts)


def extract_task_inventory_notes_storysage(memory_bank: dict, max_turn: int) -> str:
    """Extract task-related notes from a StorySage memory bank.

    StorySage doesn't have topic labels on memories, so we pull all memories
    and let the LLM judge decide which ones describe tasks.
    """
    parts = []
    for mem in memory_bank.get("memories", []):
        turn = mem.get("metadata", {}).get("turn", -1)
        if turn > max_turn:
            continue
        text = mem.get("text", "").strip()
        if text:
            parts.append(f"- {text}")
    return "\n".join(parts)


def extract_task_inventory_notes_llmroleplay(history: List[Dict], max_turn: int) -> str:
    """Extract task-inventory notes from an LLMRoleplay / freeform log."""
    # Keep only the latest notes per subtopic (they accumulate across turns)
    subtopic_notes: Dict[str, str] = {}

    for i, item in enumerate(history):
        if i > max_turn:
            break
        notes = item.get("notes", None)
        if notes and len(notes.strip()) > 0:
            topic_idx = item.get("topic_index", 0)
            question_idx = item.get("question_index", 0)
            # Task Inventory is topic index 1 (0-based) based on topics_intake.json
            if topic_idx == 1:
                key = f"{topic_idx}.{question_idx}"
                subtopic_notes[key] = notes

    parts = [f"**Subtopic {k}**\n{v}" for k, v in subtopic_notes.items()]
    return "\n\n".join(parts)


def extract_task_inventory_notes_freeform(history: List[Dict], max_turn: int) -> str:
    """Extract all notes from a freeform log (no topic labels)."""
    parts = []
    for i, item in enumerate(history):
        if i > max_turn:
            break
        notes = item.get("notes", None)
        if notes and len(notes.strip()) > 0:
            parts.append(notes)
    return "\n- ".join(parts)


# ---------------------------------------------------------------------------
# Message construction
# ---------------------------------------------------------------------------

def build_evaluation_message(task_notes: str) -> List[Dict]:
    """Build the LLM judge message for task-depth evaluation."""
    return [
        {"role": "system", "content": JUDGE_PROMPT},
        {
            "role": "user",
            "content": (
                "# Interview Notes (Task Inventory)\n\n"
                f"{task_notes}\n\n"
                "# Your Output\n"
            )
        }
    ]


# ---------------------------------------------------------------------------
# Session processing
# ---------------------------------------------------------------------------

def prepare_session_data(user_id: str, snapshot_idx: int, config: EvaluationConfig, surgery: bool = False):
    """Load session, extract Task Inventory notes, build eval message."""
    if config.mode == 'sparkme':
        session_path = f"{config.base_path}/{user_id}/execution_logs/session_0/session_agenda_snap_{snapshot_idx}.json"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_task_depth"
    elif config.mode == 'storysage':
        session_path = f"{config.base_path}/{user_id}/memory_bank_content.json"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_task_depth"
    elif config.mode in ('llmroleplay', 'freeform'):
        session_path = f"{config.base_path}/{user_id}/interview_log.jsonl"
        eval_dir = f"{config.base_path}/{user_id}/evaluations_task_depth"
    else:
        raise ValueError(f"Invalid mode: {config.mode}")

    save_eval_path = f"{eval_dir}/snap_eval_{snapshot_idx}.json"

    if not os.path.exists(session_path):
        return None

    if not surgery and os.path.exists(save_eval_path):
        with open(save_eval_path, 'r') as f:
            existing = json.load(f)
            if existing:
                return None

    # Load session data and extract Task Inventory notes
    if config.mode in ('llmroleplay', 'freeform'):
        history = []
        with open(session_path) as f:
            for line in f.readlines():
                history.append(json.loads(line))

        if snapshot_idx >= len(history) + 4:
            return None

        if config.mode == 'llmroleplay':
            task_notes = extract_task_inventory_notes_llmroleplay(history, snapshot_idx)
        else:
            task_notes = extract_task_inventory_notes_freeform(history, snapshot_idx)
    elif config.mode == 'storysage':
        with open(session_path) as f:
            memory_bank = json.load(f)
        task_notes = extract_task_inventory_notes_storysage(memory_bank, snapshot_idx)
    else:  # sparkme
        with open(session_path) as f:
            session_agenda = json.load(f)
        task_notes = extract_task_inventory_notes_sparkme(session_agenda)

    if not task_notes.strip():
        return None

    message = build_evaluation_message(task_notes)

    return {
        'user_id': user_id,
        'snapshot_idx': snapshot_idx,
        'message': message,
        'task_notes': task_notes,
        'save_path': save_eval_path,
        'eval_dir': eval_dir
    }


def process_session_openai(user_id: str, snapshot_idx: int, config: EvaluationConfig, surgery: bool = False):
    """Evaluate task depth for one session snapshot using OpenAI."""
    session_info = prepare_session_data(user_id, snapshot_idx, config, surgery=surgery)
    if session_info is None:
        return

    response = request_openai_completion(session_info['message'])
    if response is None:
        result = {"error": True, "rationale": "API request failed"}
    else:
        parsed = safe_parse_json(response)
        if parsed is None:
            result = {"error": True, "rationale": "Failed to parse response", "raw": response}
        else:
            result = parsed
            result["error"] = False

    os.makedirs(session_info['eval_dir'], exist_ok=True)
    with open(session_info['save_path'], 'w') as f:
        json.dump(result, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate task description depth in Task Inventory notes')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['sparkme', 'storysage', 'llmroleplay', 'freeform'],
                        help='Evaluation mode')
    parser.add_argument('--base-path', type=str, required=True,
                        help='Base path to logs directory')
    parser.add_argument('--sample-users-path', type=str,
                        default='analysis/sample_users_50.json',
                        help='Path to sample users JSON')
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
    parser.add_argument('--surgery', action='store_true', help='Re-evaluate already saved snapshots')

    args = parser.parse_args()

    initialize_model(args.model_config)

    config = EvaluationConfig(
        mode=args.mode,
        base_path=args.base_path,
        sample_users_path=args.sample_users_path,
        max_workers=args.max_workers
    )

    with open(config.sample_users_path, 'r') as f:
        sample_users = json.load(f)

    provider = MODEL_CONFIG.get("provider_name", "openai")

    if provider == "local":
        logging.info("Using vLLM — collecting all sessions for batch inference")

        all_session_infos = []
        for person_profile in tqdm(sample_users[:args.num_users], desc="Preparing sessions"):
            user_id = person_profile["User ID"]
            for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                session_info = prepare_session_data(user_id, snapshot_idx, config, surgery=args.surgery)
                if session_info is not None:
                    all_session_infos.append(session_info)

        logging.info(f"Prepared {len(all_session_infos)} sessions for evaluation")

        from vllm import SamplingParams

        prompts = []
        for session_info in tqdm(all_session_infos, desc="Applying chat templates"):
            prompt = TOKENIZER.apply_chat_template(
                session_info['message'], tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        gen_args = MODEL_CONFIG.get("generation_args", {})
        sampling_params = SamplingParams(**gen_args)

        logging.info("Running batch inference...")
        outputs = MODEL_CLIENT.generate(prompts, sampling_params)

        logging.info("Parsing and saving results...")
        for session_info, output in tqdm(zip(all_session_infos, outputs), total=len(outputs)):
            generated_text = output.outputs[0].text.strip()
            parsed = safe_parse_json(generated_text)

            if parsed is None:
                result = {"error": True, "rationale": "Failed to parse response", "raw": generated_text}
            else:
                result = parsed
                result["error"] = False

            os.makedirs(session_info['eval_dir'], exist_ok=True)
            with open(session_info['save_path'], 'w') as f:
                json.dump(result, f, indent=2)

        logging.info("vLLM batch evaluation complete!")

    else:
        logging.info("Using OpenAI — processing sessions in parallel with threading")

        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = []

            for person_profile in sample_users[:args.num_users]:
                user_id = person_profile["User ID"]

                for snapshot_idx in range(args.snapshot_start, args.snapshot_end, args.snapshot_step):
                    futures.append(
                        executor.submit(
                            process_session_openai, user_id, snapshot_idx, config, args.surgery
                        )
                    )

            for future in tqdm(as_completed(futures), total=len(futures),
                               desc=f"Evaluating task depth ({config.mode} mode)"):
                try:
                    future.result()
                except Exception as e:
                    logging.error(f"Error processing session: {e}")


if __name__ == "__main__":
    main()
