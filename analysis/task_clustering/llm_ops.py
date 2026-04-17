"""
LLM operations for the online task clustering pipeline.

Three operations, each backed by a structured-output LLM call:
  1. assign_task     — decide which cluster (if any) a new item belongs to
  2. update_leader   — refine a cluster's canonical statement after new members arrive
  3. split_cluster   — divisively split a heterogeneous cluster into sub-groups
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_ASSIGN_PROMPT = """\
You are grouping worker-reported task observations into clusters of similar work.

## New task
{task_text}

## Guidelines for "same cluster"
{criteria_block}

## Current clusters ({n_clusters} clusters)
{cluster_block}

---

## Your decision
Which cluster (if any) does the new task belong to?

- If it clearly fits an existing cluster: return ONLY the cluster number (e.g. "3").
- {no_match_instruction}

Do not force a match — if the task represents a genuinely different type of work, create a new cluster.
Return exactly one token. No explanation."""

_ASSIGN_PROMPT_VERBOSE = """\
You are grouping worker-reported task observations into clusters of similar work.
Err on the side of merging: if the new task represents essentially the same type of work
as an existing cluster, assign it there even if the wording differs.

## New task
{task_text}

## Guidelines for "same cluster"
{criteria_block}

## Current clusters ({n_clusters} clusters)
{cluster_block}

---

## Your decision
Which cluster (if any) does the new task belong to?

Respond in exactly this format:
REASON: <one sentence explaining your decision>
DECISION: <cluster number, or "{no_match_token}">"""

_NO_MATCH_NEW   = 'If it does not fit any cluster: return ONLY the word "new".'
_NO_MATCH_NONE  = 'If it does not fit any specific sub-type (it belongs at this general level): return ONLY the word "none".'
_NO_MATCH_FORCE = 'You MUST pick the single best-matching cluster. Do not return "new" or "none" — return only a cluster number.'

_UPDATE_LEADER_PROMPT = """\
You are a job analyst writing a short canonical label for a task cluster.

## Worker-reported observations (sample, up to 10)
{members_block}

## Instructions
Write a SHORT canonical task label (5–10 words) that:
- Names the core activity at the occupation level, not any one person's situation
- Drops domain-specific details, project names, tools, and outcomes unless universal to the role
- Reads like a task category heading — broad enough to cover all members, specific enough to be meaningful

Examples of good labels:
- "Review and synthesize research literature"
- "Design and run computational experiments"
- "Collaborate with cross-functional stakeholders"
- "Write and document technical findings"

Return ONLY the label. No explanation, no punctuation at the end."""

_SPLIT_PROMPT = """\
You are a job analyst deciding whether a task cluster should be split into sub-groups.

## Cluster canonical statement
{leader}

## Member observations ({n_members} total)
{members_block}

---

## Instructions
Examine the members carefully. If there are 2–4 **meaningfully distinct sub-types** of work
(different actions, different objects, or different tools that would make them separate bullet
points in a job description), split them into groups with short descriptive labels.

If the cluster is **homogeneous** (all members describe the same core activity), do not split.

## Output
Return a JSON array — one object per group:

[
  {{"label": "<2–6 word label for this sub-type>", "member_indices": [0, 3, 5, ...]}},
  ...
]

If no split: return a single-element array covering all indices.
Return ONLY the JSON array. No markdown fences."""


_SCREEN_PROMPT = """\
You are a job analyst screening worker-reported task statements for validity.

## Task statements ({n_tasks} statements)
{task_list}

---

## Screening checks (apply to each statement)

1. **Specific action?** — Names a concrete, observable verb.
   Pass: "Debug", "Review", "Write", "Analyze", "Build", "Design", "Present"
   Fail: "Support", "Manage", "Handle", "Facilitate", "Ensure", "Participate in"

2. **Concrete object?** — Acts on a specific artifact, system, document, or person.
   Pass: "unit tests", "research papers", "ML models", "stakeholders", "data pipelines"
   Fail: "tasks", "things", "work", "multiple priorities"

3. **Bounded activity?** — Has a start and end; not a standing trait or disposition.
   Pass: a recurring or one-time activity that happens and finishes
   Fail: "Be a good communicator", "Understand the codebase", "Know the domain"

4. **Single task?** — One coherent work unit, not 3+ unrelated activities bundled together.

If a statement fails but describes a real work activity, rewrite it to fix the issue.
If it is a personality trait, a role description, or too vague to salvage, reject it.

## Output
Return a JSON array — one object per input statement, in the same order:

[
  {{
    "index": <0-based index>,
    "status": "pass" | "rewritten" | "rejected",
    "rewritten": "<corrected statement if status=rewritten, else empty string>",
    "reason": "<one sentence: what was fixed or why rejected>"
  }},
  ...
]

Return ONLY the JSON array. No markdown fences."""


def screen_tasks(
    task_texts: list[str],
    llm: LLMClient,
    model: str = "gpt-4.1",
) -> list[dict]:
    """
    Screen a batch of raw task texts for validity.

    Returns a list of dicts with keys: index, status, rewritten, reason.
    status is one of: "pass", "rewritten", "rejected".
    """
    task_list = "\n".join(f"{i}. {t}" for i, t in enumerate(task_texts))
    prompt = _SCREEN_PROMPT.format(n_tasks=len(task_texts), task_list=task_list)
    response = llm.call(prompt, model=model, temperature=0.1, max_tokens=4096)
    try:
        return _parse_json(response)
    except Exception:
        # Fallback: pass everything through
        return [{"index": i, "status": "pass", "rewritten": "", "reason": ""} for i in range(len(task_texts))]


def _parse_json(text: str) -> list:
    text = text.strip()
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence):]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    return json.loads(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assign_task(
    task_text: str,
    leaders: list[str],
    llm: LLMClient,
    criteria: list[str] | None = None,
    model: str = "gpt-4.1",
    allow_none: bool = False,
    force: bool = False,
    return_reasoning: bool = False,
) -> str | tuple[str, str]:
    """
    Decide which cluster the new task belongs to, or "new".

    Parameters
    ----------
    task_text : str
        The raw text of the incoming task.
    leaders : list[str]
        Canonical leader statements for each current leaf cluster (1-indexed in prompt).
    llm : LLMClient
    criteria : list[str] | None
        Custom similarity criteria injected into the prompt.
        Uses DEFAULT_CRITERIA from models if None.
    model : str

    Returns
    -------
    "new" | "1" | "2" | ...
        "new" means start a new cluster; a digit string is the 1-based cluster index.
    """
    from .models import DEFAULT_CRITERIA
    crit = criteria if criteria is not None else DEFAULT_CRITERIA
    criteria_block = "\n".join(f"- {c}" for c in crit)
    cluster_block = "\n".join(f"{i+1}. {ldr}" for i, ldr in enumerate(leaders))
    no_match_instruction = _NO_MATCH_FORCE if force else (_NO_MATCH_NONE if allow_none else _NO_MATCH_NEW)

    prompt = _ASSIGN_PROMPT.format(
        task_text=task_text,
        criteria_block=criteria_block,
        n_clusters=len(leaders),
        cluster_block=cluster_block,
        no_match_instruction=no_match_instruction,
    )

    no_match_token = "none" if allow_none else "new"

    if return_reasoning:
        verbose_prompt = _ASSIGN_PROMPT_VERBOSE.format(
            task_text=task_text,
            criteria_block=criteria_block,
            n_clusters=len(leaders),
            cluster_block=cluster_block,
            no_match_token=no_match_token,
        )
        raw = llm.call(verbose_prompt, model=model, temperature=0.0, max_tokens=128).strip()
        reasoning = ""
        decision_raw = raw
        for line in raw.splitlines():
            if line.upper().startswith("REASON:"):
                reasoning = line[len("REASON:"):].strip()
            elif line.upper().startswith("DECISION:"):
                decision_raw = line[len("DECISION:"):].strip()
        choice = _parse_choice(decision_raw, leaders, no_match_token)
        return choice, reasoning

    response = llm.call(prompt, model=model, temperature=0.0, max_tokens=16).strip()
    return _parse_choice(response, leaders, no_match_token)


def _parse_choice(response: str, leaders: list[str], no_match_token: str) -> str:
    token = response.strip(" .'\"").lower()
    if token in (no_match_token, "new", "none"):
        return no_match_token
    if token.isdigit():
        idx = int(token)
        if 1 <= idx <= len(leaders):
            return token
    digits = [c for c in response if c.isdigit()]
    if digits:
        idx = int(digits[0])
        if 1 <= idx <= len(leaders):
            return str(idx)
    return no_match_token


def update_leader(
    leader: str,
    member_texts: list[str],
    llm: LLMClient,
    model: str = "gpt-4.1",
) -> str:
    """
    Refine a cluster's canonical leader statement after new members have joined.

    Returns the updated canonical statement (falls back to existing leader on error).
    """
    sample = member_texts[:10]
    members_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(sample))

    prompt = _UPDATE_LEADER_PROMPT.format(
        members_block=members_block,
    )

    try:
        response = llm.call(prompt, model=model, temperature=0.1, max_tokens=256).strip()
        return response if response else leader
    except Exception:
        return leader


def split_cluster(
    leader: str,
    member_texts: list[str],
    llm: LLMClient,
    model: str = "gpt-4.1",
) -> list[dict]:
    """
    Ask the LLM whether the cluster should be divisively split.

    Returns a list of group dicts:
        [{"label": str, "member_indices": [int, ...]}, ...]

    A single-element list means "no meaningful split found".
    """
    members_block = "\n".join(f"{i+1}. {t}" for i, t in enumerate(member_texts))

    prompt = _SPLIT_PROMPT.format(
        leader=leader,
        n_members=len(member_texts),
        members_block=members_block,
    )

    response = llm.call(prompt, model=model, temperature=0.0, max_tokens=1024)
    try:
        groups = _parse_json(response)
        # Convert 1-based indices in case the LLM returned them that way
        for g in groups:
            g["member_indices"] = [
                (i - 1) if i > 0 else i
                for i in g["member_indices"]
                if isinstance(i, int)
            ]
        return groups
    except Exception:
        # Fallback: no split
        return [{"label": leader, "member_indices": list(range(len(member_texts)))}]
