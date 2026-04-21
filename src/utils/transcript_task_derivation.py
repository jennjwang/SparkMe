"""Helpers for deriving task seeds from pilot chat transcripts."""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any, Iterable, Sequence

from src.utils.task_hierarchy import organize_tasks

_LOG_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+ - INFO - (Interviewer|User): (.*)"
)
_SESSION_RE = re.compile(r"^session_(\d+)$")
_TASKS_TAG_RE = re.compile(r"<tasks>(.*?)</tasks>", re.IGNORECASE | re.DOTALL)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]+|\d+[.)])\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_CADENCE_RE = re.compile(
    r"\b(weekly|monthly|daily|annual|quarterly|biweekly|bi-weekly|regular|occasional|periodic)\b"
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_IRREGULAR_STEMS = {
    "reading": "read",
    "reads": "read",
    "wrote": "write",
    "written": "write",
    "writing": "write",
    "running": "run",
    "ran": "run",
    "built": "build",
    "building": "build",
    "met": "meet",
    "meeting": "meet",
}

_TASK_DERIVATION_PROMPT = """You are extracting recurring work tasks from an interview transcript.

Return ONLY a JSON array wrapped in <tasks>...</tasks>.

Rules:
- Use only tasks explicitly described by the user in the transcript.
- Each item must be a concrete recurring activity (action + object, and include purpose when stated).
- Keep tasks flat: no grouped labels, headings, or categories.
- Deduplicate semantically equivalent tasks.
- Exclude role labels, traits, goals, and one-off personal anecdotes.
- Keep each task concise and readable.

Transcript:
{transcript}
"""


def parse_chat_history_lines(lines: Iterable[str]) -> list[dict[str, str]]:
    """Parse SparkMe chat-history lines, preserving multiline message bodies."""
    messages: list[dict[str, str]] = []
    current: dict[str, str] | None = None

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        match = _LOG_RE.match(line)
        if match:
            if current and current.get("content", "").strip():
                current["content"] = current["content"].strip()
                messages.append(current)
            current = {"speaker": match.group(1), "content": match.group(2).rstrip()}
            continue

        if current is not None:
            current["content"] += "\n" + line.rstrip()

    if current and current.get("content", "").strip():
        current["content"] = current["content"].strip()
        messages.append(current)

    return messages


def load_chat_history_messages(log_path: str) -> list[dict[str, str]]:
    """Load and parse one chat_history.log file."""
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_chat_history_lines(f)


def role_turn_count(messages: Sequence[dict[str, str]], speaker: str) -> int:
    return sum(1 for m in messages if m.get("speaker") == speaker and m.get("content", "").strip())


def is_valid_transcript(
    messages: Sequence[dict[str, str]],
    min_user_turns: int = 1,
    min_interviewer_turns: int = 1,
) -> bool:
    """Check whether parsed transcript content is minimally usable."""
    if not messages:
        return False
    return (
        role_turn_count(messages, "User") >= min_user_turns
        and role_turn_count(messages, "Interviewer") >= min_interviewer_turns
    )


def select_latest_valid_chat_history_log(
    pilot_dir: str,
    user_id: str,
    min_user_turns: int = 1,
    min_interviewer_turns: int = 1,
) -> dict[str, Any] | None:
    """Select the latest valid pilot session chat_history.log for a user."""
    logs_dir = os.path.join(pilot_dir, user_id, "execution_logs")
    if not os.path.isdir(logs_dir):
        return None

    sessions: list[tuple[int, str, str]] = []
    for session_name in os.listdir(logs_dir):
        session_path = os.path.join(logs_dir, session_name)
        if not os.path.isdir(session_path):
            continue
        match = _SESSION_RE.match(session_name)
        if not match:
            continue
        log_path = os.path.join(session_path, "chat_history.log")
        if not os.path.exists(log_path):
            continue
        sessions.append((int(match.group(1)), session_name, log_path))

    sessions.sort(key=lambda x: (x[0], x[1]), reverse=True)

    for _, session_name, log_path in sessions:
        try:
            messages = load_chat_history_messages(log_path)
        except OSError:
            continue
        if is_valid_transcript(
            messages,
            min_user_turns=min_user_turns,
            min_interviewer_turns=min_interviewer_turns,
        ):
            return {
                "session_name": session_name,
                "log_path": log_path,
                "messages": messages,
            }

    return None


def format_transcript(messages: Sequence[dict[str, str]]) -> str:
    """Serialize parsed messages into a role-prefixed transcript string."""
    lines: list[str] = []
    for msg in messages:
        speaker = msg.get("speaker", "").strip() or "Unknown"
        content = str(msg.get("content", "")).strip()
        if not content:
            continue
        lines.append(f"{speaker}: {content}")
    return "\n".join(lines)


def normalize_task_text(task: str) -> str:
    text = str(task or "").strip()
    if not text:
        return ""
    text = _BULLET_PREFIX_RE.sub("", text)
    text = text.strip()
    text = text.rstrip(".,;:")
    text = _WHITESPACE_RE.sub(" ", text)
    return text.strip()


def normalize_deduplicate_tasks(tasks: Iterable[str]) -> list[str]:
    """Normalize and deduplicate while preserving first-seen order."""
    out: list[str] = []
    seen: set[str] = set()
    for task in tasks:
        for part in str(task or "").splitlines():
            cleaned = normalize_task_text(part)
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
    return out


def _task_core(task: str) -> str:
    core = str(task or "").lower().strip()
    core = _CADENCE_RE.sub("", core)
    core = _WHITESPACE_RE.sub(" ", core).strip()
    if " to " in core:
        core = core.split(" to ", 1)[0].strip()
    return core


def _token_jaccard(a: str, b: str) -> float:
    a_tokens = _tokenize(a)
    b_tokens = _tokenize(b)
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _stem_token(token: str) -> str:
    tok = token.lower()
    if tok in _IRREGULAR_STEMS:
        return _IRREGULAR_STEMS[tok]
    if tok.endswith("ies") and len(tok) > 4:
        return tok[:-3] + "y"
    if tok.endswith("ing") and len(tok) > 5:
        return tok[:-3]
    if tok.endswith("ed") and len(tok) > 4:
        return tok[:-2]
    if tok.endswith("s") and len(tok) > 3:
        return tok[:-1]
    return tok


def _tokenize(text: str) -> set[str]:
    return {_stem_token(tok) for tok in _TOKEN_RE.findall(text.lower())}


def _is_semantic_duplicate(core: str, kept_cores: Sequence[str]) -> bool:
    core_tokens = _tokenize(core)
    for kept in kept_cores:
        if core == kept or core in kept or kept in core:
            return True
        jaccard = _token_jaccard(core, kept)
        kept_tokens = _tokenize(kept)
        overlap = core_tokens & kept_tokens
        smaller = min(len(core_tokens), len(kept_tokens))
        if smaller >= 2 and len(overlap) >= 2 and (len(overlap) == smaller or jaccard >= 0.75):
            return True
        if SequenceMatcher(None, core, kept).ratio() >= 0.85 and jaccard >= 0.6:
            return True
    return False


def deduplicate_task_variants(tasks: Sequence[str]) -> list[str]:
    """Collapse near-duplicates that share the same action+object core."""
    deduped: list[str] = []
    kept_cores: list[str] = []
    for task in tasks:
        core = _task_core(task)
        if not core:
            continue
        if _is_semantic_duplicate(core, kept_cores):
            continue
        deduped.append(task)
        kept_cores.append(core)
    return deduped


def _strip_markdown_fences(text: str) -> str:
    content = text.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        content = "\n".join(lines).strip()
    return content


def _coerce_task_list(value: Any) -> list[str]:
    if isinstance(value, list):
        tasks: list[str] = []
        for item in value:
            if isinstance(item, str):
                tasks.append(item)
            elif isinstance(item, dict):
                for key in ("task", "name", "text"):
                    if isinstance(item.get(key), str):
                        tasks.append(item[key])
                        break
        return tasks

    if isinstance(value, dict):
        for key in ("tasks", "task_list", "items"):
            if key in value:
                return _coerce_task_list(value[key])

    return []


def parse_llm_task_response(response_text: str) -> list[str]:
    """Parse task arrays from tagged or raw LLM output."""
    candidates = []
    match = _TASKS_TAG_RE.search(response_text or "")
    if match:
        candidates.append(match.group(1))
    candidates.append(response_text or "")

    for block in candidates:
        payload = _strip_markdown_fences(block)
        if not payload:
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        tasks = _coerce_task_list(parsed)
        if tasks:
            return normalize_deduplicate_tasks(tasks)

    # Fallback for malformed output: parse bullet/numbered lines heuristically.
    fallback_lines = []
    for line in (response_text or "").splitlines():
        cleaned = normalize_task_text(line)
        if cleaned and not cleaned.lower().startswith("tasks"):
            fallback_lines.append(cleaned)
    return normalize_deduplicate_tasks(fallback_lines)


def extract_tasks_with_llm(
    transcript_text: str,
    client: Any,
    model: str = "gpt-4.1-mini",
) -> list[str]:
    """Extract recurring concrete tasks from transcript text."""
    prompt = _TASK_DERIVATION_PROMPT.format(transcript=transcript_text)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1200,
    )
    content = response.choices[0].message.content or ""
    return parse_llm_task_response(content)


def flatten_leaf_tasks(tree: Sequence[dict[str, Any]]) -> list[str]:
    """Flatten a grouped task tree into only leaf task names."""
    leaves: list[str] = []

    def _walk(node: Any) -> None:
        if not isinstance(node, dict):
            return
        children = node.get("children") or []
        if children:
            for child in children:
                _walk(child)
            return
        name = str(node.get("name", "")).strip()
        if name:
            leaves.append(name)

    for node in tree:
        _walk(node)

    return deduplicate_task_variants(normalize_deduplicate_tasks(leaves))


def derive_tasks_from_messages(
    messages: Sequence[dict[str, str]],
    client: Any,
    derivation_model: str = "gpt-4.1-mini",
    organizer_model: str | None = None,
) -> dict[str, Any]:
    """Run transcript -> tasks derivation and task-quality screening pipeline."""
    transcript_text = format_transcript(messages)
    result: dict[str, Any] = {
        "message_count": len(messages),
        "user_turn_count": role_turn_count(messages, "User"),
        "interviewer_turn_count": role_turn_count(messages, "Interviewer"),
        "raw_extracted_tasks": [],
        "screened_task_tree": [],
        "derived_tasks": [],
        "reason": "",
        "ok": False,
    }

    if not is_valid_transcript(messages):
        result["reason"] = "invalid_transcript"
        return result

    try:
        raw_tasks = extract_tasks_with_llm(
            transcript_text,
            client=client,
            model=derivation_model,
        )
    except Exception as exc:  # pragma: no cover - defensive fallback path
        result["reason"] = f"llm_error:{type(exc).__name__}"
        return result

    if not raw_tasks:
        result["reason"] = "empty_task_extraction"
        return result

    tree = organize_tasks(
        raw_tasks,
        model_name=organizer_model or derivation_model,
        screen=True,
        append_uncovered_tasks=False,
    )
    derived_tasks = flatten_leaf_tasks(tree)

    result.update(
        {
            "raw_extracted_tasks": raw_tasks,
            "screened_task_tree": tree,
            "derived_tasks": derived_tasks,
            "reason": "" if derived_tasks else "empty_after_screening",
            "ok": bool(derived_tasks),
        }
    )
    return result


def derive_tasks_from_latest_valid_transcript(
    pilot_dir: str,
    user_id: str,
    client: Any,
    derivation_model: str = "gpt-4.1-mini",
    organizer_model: str | None = None,
) -> dict[str, Any]:
    """Derive task seeds from the latest valid pilot transcript for one user."""
    selected = select_latest_valid_chat_history_log(pilot_dir, user_id)
    base: dict[str, Any] = {
        "user_id": user_id,
        "source_session": None,
        "source_chat_history_log": None,
        "task_derivation_model": derivation_model,
        "ok": False,
        "reason": "no_valid_transcript",
        "raw_extracted_tasks": [],
        "screened_task_tree": [],
        "derived_tasks": [],
    }
    if not selected:
        return base

    derived = derive_tasks_from_messages(
        selected["messages"],
        client=client,
        derivation_model=derivation_model,
        organizer_model=organizer_model or derivation_model,
    )
    base.update(derived)
    base["source_session"] = selected["session_name"]
    base["source_chat_history_log"] = selected["log_path"]
    return base
