"""
eval_question_adoption.py — Gap 5: StrategicPlanner question adoption rate

After each planning event the StrategicPlanner suggests strategic questions.
This script checks whether the Interviewer actually asked those (or semantically
similar) questions in the turns following the planning event.

Data sources
------------
- StrategicPlanner_event_stream.log  — contains `generate_strategic_questions_response`
  events with the suggested question JSON inside XML tags.
- chat_history.log                   — contains timestamped Interviewer / User turns.
- strategic_state_turn_N.json        — used to identify planning-event timestamps
  (same as eval_planner_accuracy: deduplicate by last_planning_turn).

Matching strategy
-----------------
For each planning event at timestamp T:
  1. Extract all suggested questions from the event.
  2. Collect Interviewer messages that fall after T (within the same "replay
     window" — up to the next planning event's timestamp, or --window-minutes
     if no next event follows quickly).
  3. Filter out empty messages and session-opening greetings.
  4. Embed all suggested questions + collected Interviewer messages with
     text-embedding-3-small.
  5. A suggestion is "adopted" if its cosine-similarity to any Interviewer
     message exceeds --sim-threshold (default 0.75).

Metrics per user
----------------
- adoption_rate       : fraction of suggestions adopted across all planning events
- per_event breakdown : planning_ts, num_suggestions, num_adopted, adoption_rate,
                        adopted_questions, missed_questions

Aggregate (summary)
-------------------
- mean_adoption_rate  : across all users with ≥1 evaluable planning event

Usage
-----
python evaluation/eval_question_adoption.py \\
    --base-path logs/ \\
    --sample-users-path analysis/sample_users_study.json \\
    --output-dir results/question_adoption \\
    --summary-path results/question_adoption_summary.json
"""

import argparse
import glob
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path:
    sys.path.insert(0, _root)

from dotenv import load_dotenv
load_dotenv(override=True)

from openai import OpenAI

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

_oai_client: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _oai_client
    if _oai_client is None:
        _oai_client = OpenAI()
    return _oai_client


def embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Return (N, D) embedding matrix for a list of texts."""
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    response = _get_client().embeddings.create(input=texts, model=model)
    return np.array([item.embedding for item in response.data], dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarities: (N,D) x (M,D) → (N,M)."""
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------

_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+")
_QUESTIONS_JSON_RE = re.compile(r"<questions>\s*(\[.*?\])\s*</questions>", re.DOTALL)
_QUESTION_CONTENT_RE = re.compile(r"<content>(.*?)</content>", re.DOTALL)
_QUESTION_SUBTOPIC_RE = re.compile(r"<subtopic_id>(.*?)</subtopic_id>", re.DOTALL)
_QUESTION_ELEM_RE = re.compile(r"<question>(.*?)</question>", re.DOTALL)
_SPEAKER_RE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - (Interviewer|User): (.*)", re.DOTALL)

# Patterns that indicate a session-opening greeting (not a substantive question)
_GREETING_PREFIXES = (
    "Hi, thanks so much for taking the time",
    "Hi there",
    "Hello",
)


def _parse_timestamp(line: str) -> Optional[datetime]:
    m = _TS_RE.match(line)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _is_greeting(text: str) -> bool:
    t = text.strip()
    return any(t.startswith(p) for p in _GREETING_PREFIXES)


def parse_planning_events(event_stream_path: str) -> List[Dict]:
    """
    Parse StrategicPlanner_event_stream.log for generate_strategic_questions_response
    events. Returns list of dicts: {timestamp, questions: [{content, subtopic_id, ...}]}.
    """
    if not os.path.exists(event_stream_path):
        return []

    with open(event_stream_path) as f:
        raw = f.read()

    # Split on each new log-line timestamp so we can isolate multi-line events
    # A new log entry starts at column 0 with a date like "2026-..."
    entries = re.split(r"(?m)^(?=\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - )", raw)

    events = []
    for entry in entries:
        if "generate_strategic_questions_response" not in entry:
            continue
        ts = _parse_timestamp(entry)
        if ts is None:
            continue
        # Format 1: JSON array  <questions>[ {...}, ... ]</questions>
        m = _QUESTIONS_JSON_RE.search(entry)
        if m:
            try:
                questions = json.loads(m.group(1))
            except json.JSONDecodeError:
                questions = None
            if isinstance(questions, list) and questions:
                events.append({"timestamp": ts, "questions": questions})
                continue

        # Format 2: XML elements  <question><content>...</content>...</question>
        xml_questions = []
        for qm in _QUESTION_ELEM_RE.finditer(entry):
            qblock = qm.group(1)
            cm = _QUESTION_CONTENT_RE.search(qblock)
            sm = _QUESTION_SUBTOPIC_RE.search(qblock)
            if cm:
                xml_questions.append({
                    "content": cm.group(1).strip(),
                    "subtopic_id": sm.group(1).strip() if sm else "",
                })
        if xml_questions:
            events.append({"timestamp": ts, "questions": xml_questions})

    return events


def parse_interviewer_turns(chat_history_path: str) -> List[Dict]:
    """
    Parse chat_history.log for Interviewer messages.
    Returns list of dicts: {timestamp, text}.
    Only non-empty, non-greeting messages are returned.
    """
    if not os.path.exists(chat_history_path):
        return []

    turns = []
    with open(chat_history_path) as f:
        for line in f:
            m = _SPEAKER_RE.match(line)
            if not m or m.group(1) != "Interviewer":
                continue
            text = m.group(2).strip()
            if not text or _is_greeting(text):
                continue
            ts = _parse_timestamp(line)
            if ts is None:
                continue
            turns.append({"timestamp": ts, "text": text})

    return turns


# ---------------------------------------------------------------------------
# Core evaluation per session
# ---------------------------------------------------------------------------

def evaluate_session(
    session_dir: str,
    window_minutes: int,
    sim_threshold: float,
) -> List[Dict]:
    """
    Evaluate question adoption for all planning events in one session directory.
    Returns one entry per planning event.
    """
    event_stream_path = os.path.join(session_dir, "StrategicPlanner_event_stream.log")
    chat_history_path = os.path.join(session_dir, "chat_history.log")

    planning_events = parse_planning_events(event_stream_path)
    if not planning_events:
        return []

    interviewer_turns = parse_interviewer_turns(chat_history_path)
    if not interviewer_turns:
        return []

    # Sort everything by timestamp
    planning_events.sort(key=lambda e: e["timestamp"])
    interviewer_turns.sort(key=lambda t: t["timestamp"])

    results = []
    for i, event in enumerate(planning_events):
        t_start = event["timestamp"]
        # Window end: next planning event's timestamp or t_start + window_minutes
        if i + 1 < len(planning_events):
            t_end = planning_events[i + 1]["timestamp"]
        else:
            from datetime import timedelta
            t_end = t_start + timedelta(minutes=window_minutes)

        # Collect Interviewer messages in (t_start, t_end)
        window_turns = [
            t for t in interviewer_turns
            if t_start < t["timestamp"] <= t_end
        ]

        questions = event["questions"]
        suggestion_texts = [q.get("content", "") for q in questions]
        suggestion_texts = [s for s in suggestion_texts if s.strip()]
        window_texts = [t["text"] for t in window_turns]

        if not suggestion_texts:
            continue

        # Embed in one batch
        all_texts = suggestion_texts + window_texts
        try:
            embs = embed_texts(all_texts)
        except Exception as e:
            logging.warning(f"Embedding failed for {session_dir}: {e}")
            continue

        sug_embs = embs[: len(suggestion_texts)]
        win_embs = embs[len(suggestion_texts) :]

        adopted_indices = []
        missed_indices = []

        if win_embs.shape[0] == 0:
            # No interviewer questions in window — nothing adopted
            missed_indices = list(range(len(suggestion_texts)))
        else:
            sim_matrix = cosine_sim(sug_embs, win_embs)  # (N_sug, N_win)
            max_sims = sim_matrix.max(axis=1)             # (N_sug,)
            for j, (sug, max_s) in enumerate(zip(suggestion_texts, max_sims)):
                if max_s >= sim_threshold:
                    adopted_indices.append(j)
                else:
                    missed_indices.append(j)

        num_adopted = len(adopted_indices)
        num_suggestions = len(suggestion_texts)
        adoption_rate = num_adopted / num_suggestions if num_suggestions else 0.0

        results.append({
            "planning_ts": t_start.isoformat(),
            "num_suggestions": num_suggestions,
            "num_adopted": num_adopted,
            "adoption_rate": adoption_rate,
            "window_turns": len(window_turns),
            "adopted_questions": [suggestion_texts[j] for j in adopted_indices],
            "missed_questions": [suggestion_texts[j] for j in missed_indices],
        })

    return results


# ---------------------------------------------------------------------------
# Core evaluation per user
# ---------------------------------------------------------------------------

def evaluate_user(
    user_id: str,
    base_path: str,
    output_dir: str,
    window_minutes: int = 20,
    sim_threshold: float = 0.75,
    overwrite: bool = False,
) -> Optional[Dict]:
    save_path = os.path.join(output_dir, f"{user_id}.json")
    if not overwrite and os.path.exists(save_path):
        return None

    exec_dir = os.path.join(base_path, user_id, "execution_logs")
    if not os.path.isdir(exec_dir):
        return None

    all_events = []
    for session_name in sorted(os.listdir(exec_dir)):
        session_dir = os.path.join(exec_dir, session_name)
        if not os.path.isdir(session_dir):
            continue
        events = evaluate_session(session_dir, window_minutes, sim_threshold)
        all_events.extend(events)

    if not all_events:
        logging.warning(f"{user_id}: no evaluable planning events found")
        return None

    rates = [e["adoption_rate"] for e in all_events]
    mean_rate = sum(rates) / len(rates)

    result = {
        "user_id": user_id,
        "window_minutes": window_minutes,
        "sim_threshold": sim_threshold,
        "summary": {
            "planning_events": len(all_events),
            "mean_adoption_rate": mean_rate,
            "total_suggestions": sum(e["num_suggestions"] for e in all_events),
            "total_adopted": sum(e["num_adopted"] for e in all_events),
        },
        "events": all_events,
    }

    os.makedirs(output_dir, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(result, f, indent=2)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="StrategicPlanner question adoption (Gap 5)")
    parser.add_argument("--base-path", required=True)
    parser.add_argument("--sample-users-path", default="analysis/sample_users_study.json")
    parser.add_argument("--output-dir", default="results/question_adoption")
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--window-minutes", type=int, default=20,
                        help="Minutes after planning event to look for adopted questions (default 20)")
    parser.add_argument("--sim-threshold", type=float, default=0.75,
                        help="Cosine similarity threshold for adoption (default 0.75)")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Parallel workers (keep low to avoid OpenAI rate limits)")
    parser.add_argument("--num-users", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    with open(args.sample_users_path) as f:
        sample_users = json.load(f)
    user_ids = [u["User ID"] for u in sample_users[: args.num_users]]

    all_results = []

    def _run(uid):
        return evaluate_user(
            uid, args.base_path, args.output_dir,
            args.window_minutes, args.sim_threshold, args.overwrite,
        )

    with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
        futures = {pool.submit(_run, uid): uid for uid in user_ids}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Evaluating question adoption"):
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

    rates = [r["summary"]["mean_adoption_rate"] for r in all_results]
    total_sug = sum(r["summary"]["total_suggestions"] for r in all_results)
    total_adp = sum(r["summary"]["total_adopted"] for r in all_results)

    summary = {
        "num_users": len(all_results),
        "window_minutes": args.window_minutes,
        "sim_threshold": args.sim_threshold,
        "mean_adoption_rate": sum(rates) / len(rates) if rates else None,
        "global_adoption_rate": total_adp / total_sug if total_sug else None,
        "total_suggestions": total_sug,
        "total_adopted": total_adp,
    }
    logging.info(
        f"Done. Users={summary['num_users']}  "
        f"MeanAdoption={summary['mean_adoption_rate']:.3f}  "
        f"GlobalAdoption={summary['global_adoption_rate']:.3f}"
    )

    if args.summary_path:
        os.makedirs(os.path.dirname(args.summary_path) or ".", exist_ok=True)
        with open(args.summary_path, "w") as f:
            json.dump({"summary": summary, "per_user": all_results}, f, indent=2)
        logging.info(f"Summary written to {args.summary_path}")


if __name__ == "__main__":
    main()
