"""Organize a flat list of tasks into a hierarchical tree.

Three passes:
  0. Screen — drop entries that aren't actually tasks (personality traits,
     role descriptions, vague generalities). Uses the same 4 validity checks
     as analysis/onet/3_aggregate_canonical_tasks.py (specific action,
     concrete object, bounded activity, single task). The 5th O*NET check
     ("common to this occupation") does not apply here since we have no
     occupation context for a single user.
  1. Dedup — merge tasks that describe the same activity (including a
     sub-aspect of another task, or "use AI tool to do X" vs. "X").
  2. Group — nest specific tasks under a parent task in the list, OR under a
     newly-introduced umbrella label when several leaves share an obvious
     theme.

Validation is merge-aware: every screened task must appear in the tree either
as a node name or in some node's `merged_from` list. Anything missing is
appended as a top-level leaf so we never silently drop a screened task.
"""

import json
import os
import re
from functools import lru_cache
from typing import List, Dict, Any, Set

from src.utils.llm.engines import get_engine, invoke_engine


# Reuse the same similarity rules used by the offline clustering pipeline so
# the chat widget and the analysis pipeline agree on what "same task" means.
_DEFAULT_CRITERIA: List[str] = [
    "Same core action and object domain (e.g. both involve reviewing code with teammates)",
    "One is a specific instance of the other (attending standups IS sharing progress updates with the team)",
    "One is a subtask of the other",
    "A job analyst would describe them with the same canonical task statement",
    "Tasks represent the same type of work even if phrased differently by different people",
]

_AI_TOOL_HINT_RE = re.compile(
    r"\b(ai|ai tools?|genai|generative ai|llm|llms|chatgpt|gpt(?:-[\w.]+)?|claude|copilot|gemini|cursor)\b"
)
_AI_USAGE_MARKER_RE = re.compile(r"\b(using|with|via|leveraging|utilizing)\b")
_AI_PREFIX_RE = re.compile(
    r"^(?:using|leveraging|utilizing)\s+"
    r"(?:ai|ai tools?|genai|generative ai|llm|llms|chatgpt|gpt(?:-[\w.]+)?|claude|copilot|gemini|cursor)"
    r"(?:\s+tools?)?\s+to\s+",
    re.IGNORECASE,
)
_AI_INLINE_RE = re.compile(
    r"\s+(?:using|with|via)\s+"
    r"(?:ai|ai tools?|genai|generative ai|llm|llms|chatgpt|gpt(?:-[\w.]+)?|claude|copilot|gemini|cursor)"
    r"(?:\s+tools?)?\b",
    re.IGNORECASE,
)
_BLANK_SLATE_RE = re.compile(
    r"\b(?:starting from|from)\s+(?:a\s+)?blank\s+slate\b|\bfrom scratch\b",
    re.IGNORECASE,
)
# Trailing adjunct clauses stripped by the core extractor when the remaining
# stem still reads as a complete action+object. Purpose ("to <rest>"),
# beneficiary ("for <rest>"), and means ("by/via/through/using <rest>").
_PURPOSE_TAIL_RE = re.compile(
    r"\s+(?:to|in order to|so as to)\s+\S.*$",
    re.IGNORECASE,
)
_BENEFICIARY_TAIL_RE = re.compile(r"\s+for\s+\S.*$", re.IGNORECASE)
_MEANS_TAIL_RE = re.compile(
    r"\s+(?:by|via|through|using|leveraging|utilizing)\s+\S.*$",
    re.IGNORECASE,
)
# Small curated verb-synonym map. Keys are lowercase phrase prefixes; value is
# the canonical form. Only includes cases where paraphrase is unambiguous.
_VERB_SYNONYMS: Dict[str, str] = {
    "coming up with": "generating",
    "come up with": "generating",
    "creating": "generating",
    "drafting": "writing",
    "composing": "writing",
    "authoring": "writing",
    "having meetings with": "meeting with",
    "have meetings with": "meeting with",
    "participating in": "attending",
    "taking part in": "attending",
}
# Drop the particle "for" between a task's verb (first token) and its object,
# so "<verb> for <X>" collapses to "<verb> <X>" — e.g. "preparing for
# presentations" ≡ "preparing presentations". This runs only on single-word
# verbs at the very start of the stem.
_VERB_FOR_PARTICLE_RE = re.compile(r"^(\S+)\s+for\s+(?=\S)", re.IGNORECASE)
# Sort longest-first so multi-word phrases win over single-word prefixes.
_VERB_SYNONYM_KEYS: List[str] = sorted(_VERB_SYNONYMS, key=len, reverse=True)
_RUN_REFINE_PASS = os.getenv("TASK_HIERARCHY_ENABLE_PASS3", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
_FIRST_PERSON_CONTRACTIONS_RE = re.compile(
    r"\b(i['’](?:m|ve|d|ll)|we['’](?:re|ve|d|ll))\b",
    re.IGNORECASE,
)
_FIRST_PERSON_PRONOUNS_RE = re.compile(
    r"\b(i|me|my|mine|myself|we|us|our|ours|ourselves)\b",
    re.IGNORECASE,
)
_LEADING_AUX_RE = re.compile(
    r"^(?:am|are|is|was|were|have|has|had|being|been)\s+",
    re.IGNORECASE,
)

_CRITERIA_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "analysis", "task_clustering", "criteria.txt",
)


@lru_cache(maxsize=1)
def _load_criteria() -> List[str]:
    """Load similarity rules from analysis/task_clustering/criteria.txt.

    Falls back to the in-module defaults when the file is missing.
    """
    try:
        with open(_CRITERIA_FILE, "r", encoding="utf-8") as f:
            rules = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        return rules or list(_DEFAULT_CRITERIA)
    except OSError:
        return list(_DEFAULT_CRITERIA)


def _build_prompt(
    tasks: List[str],
    screen: bool = True,
    grouping_feedback: str = "",
) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    screen_section = """
**Screen** — for each task, check: does it name a specific recurring action with a concrete object?
- Rewrite vague verbs ("working on", "handling") to the specific action implied.
- Split if it bundles 2+ unrelated activities.
- Reject aspirations, goals, and role descriptions with no specific action.

""" if screen else ""
    rejected_note = "- Rejected tasks are omitted entirely.\n" if screen else ""
    feedback = re.sub(r"\s+", " ", str(grouping_feedback or "").strip())
    feedback_section = (
        f"""
**User feedback on grouping** — apply this while clustering:
- {feedback}
- Prioritize this guidance when deciding which tasks should be siblings and how to label group nodes.
- Keep all leaf-task fidelity rules unchanged (no leaf rewrites, no dropped tasks).
"""
        if feedback
        else ""
    )
    return f"""Organize this flat list of work tasks into a clean, deduplicated hierarchy.

Show your reasoning in <thinking>...</thinking>, then output JSON in <hierarchy>...</hierarchy>.

## In <thinking>: {("Screen, " if screen else "")}Cluster, then Deduplicate

{screen_section}**Cluster** — for each task, extract (a) action verb, (b) object domain, and (c) objective (the purpose or outcome). Then group by shared action + object domain:
- Tasks sharing the same specific object belong together regardless of verb.
- Tasks sharing the same broad object domain belong together.
- ALL meeting/talk/seminar/check-in activities → ONE cluster, regardless of verb or audience.
- Group workflow-lifecycle siblings under one parent when they act on the same artifact class (e.g., running experiments + analyzing experiment results).
- "reading X" and "writing X" are always separate even if the same object class.
- For each cluster, derive the shared objective from the extracted (c) values of its members.
- Name each cluster as a full task statement in the exact shape "<action> <object> to <purpose>".
- The label MUST contain all three parts: a concrete action verb, a concrete object/domain, and a purpose after "to".
- Use descriptive labels (typically 8-18 words). Avoid short noun phrases like "Meetings and talks", "Experimentation", or "Writing".
- Write high-level labels as impersonal task statements: remove personal association words (e.g., "I", "my", "we", "our").
- Good example: "participating in meetings and talks to exchange updates and feedback".

**Deduplicate** — merge when tasks share the same action AND the same direct object (the thing being acted on). Trailing clauses — purpose ("to ..."), beneficiary ("for ..."), or tool qualifiers ("using AI ...") — do NOT make tasks distinct. When in doubt, prefer merging; keep the single most informative phrasing.

Merge patterns (all of these MUST be merged):
- Same action+object, one has a purpose clause, the other does not:
  "Writing grants to secure funding" + "Writing grants" → MERGE (keep "Writing grants to secure funding")
- Same action+object, different purpose clauses:
  "Running experiments to test hypotheses" + "Running experiments to advance research" → MERGE
- Same action+object, one uses "for <beneficiary>", the other uses "to <purpose>":
  "Providing career development guidance for trainees" + "Providing career development guidance to support trainee growth" → MERGE
- Verb + "for" + object vs verb + object (same activity, stylistic variant):
  "Preparing for presentations" + "Preparing presentations to communicate work" → MERGE
- Tool qualifier on one variant only:
  "Running experiments using AI tools" + "Running experiments" → MERGE. Keep tool context as a trailing qualifier: "<action> <object> to <purpose> with <tool>".

Do NOT merge:
- Different objects (writing grants vs writing papers).
- Different actions on the same object (reading papers vs writing papers).
- Different document types (proposal vs manuscript).
- Preparing for X vs attending X (different actions, not stylistic variants).

Pre-flight check before emitting the tree: for every pair of top-level leaves whose first 2–3 words match, verify they are NOT the same action+object with just a trailing clause. If they are, merge them.

{feedback_section}

## Output format

Return a JSON array in <hierarchy>...</hierarchy>. Each cluster with 2+ tasks becomes a group node; single-task clusters are top-level leaves. Max 2 levels — no grandchildren.
Prefer fewer coherent top-level buckets over many flat leaves. If the input has 6+ tasks, aim for roughly 3–6 top-level buckets unless tasks are genuinely unrelated.

Before finalizing output, do a self-check:
- If you created multiple meeting-like buckets, merge them into one descriptive meetings bucket.
- If any invented group label is not in "<action> <object> to <purpose>" form, rewrite it.
- If you left many near-duplicate top-level leaves, re-cluster them before emitting JSON.

Two kinds of parent nodes:
- **Existing task as umbrella**: use its verbatim wording as `name`, omit `is_group`.
- **Invented label**: use the cluster name you derived above as `name`, set `"is_group": true`.

Node shape:
{{
  "name": "<verbatim task or invented label>",
  "is_group": true,       // only for invented labels
  "merged_from": [...],   // verbatim tasks absorbed into this node (dedup only)
  "children": [...]       // leaf nodes; omit or leave empty for leaves
}}

Example (illustrative — do not copy task names):
<hierarchy>
[
  {{
    "name": "Attending meetings to exchange updates and feedback",
    "is_group": true,
    "children": [
      {{"name": "attending advisor meetings to get feedback", "children": []}},
      {{"name": "attending lab meetings to present work", "children": []}},
      {{"name": "attending project meetings to sync on status", "children": []}}
    ]
  }},
  {{
    "name": "running experiments to test hypotheses",
    "merged_from": ["running experiments for PhD work using AI tools to advance research"],
    "children": []
  }},
  {{
    "name": "reading papers to track the field",
    "children": []
  }}
]
</hierarchy>

{rejected_note}Every task must appear exactly once (as a `name` or in `merged_from`).

Input tasks:
{task_lines}"""


def _json_parse_candidates(payload: str) -> List[str]:
    """Generate progressively more forgiving JSON payload candidates."""
    seen: Set[str] = set()
    out: List[str] = []

    def _add(candidate: str):
        cand = str(candidate or "").strip()
        if not cand or cand in seen:
            return
        seen.add(cand)
        out.append(cand)

    _add(payload)

    # Common case: model wrapped JSON with extra prose.
    m_arr = re.search(r"(\[[\s\S]*\])", payload)
    if m_arr:
        _add(m_arr.group(1))
    m_obj = re.search(r"(\{[\s\S]*\})", payload)
    if m_obj:
        _add(m_obj.group(1))

    # Best-effort slice from first JSON opener to last closer.
    starts = [idx for idx in (payload.find("["), payload.find("{")) if idx != -1]
    end = max(payload.rfind("]"), payload.rfind("}"))
    if starts and end != -1 and end > min(starts):
        _add(payload[min(starts):end + 1])

    # Remove trailing commas before closing braces/brackets.
    for candidate in list(out):
        _add(re.sub(r",(\s*[}\]])", r"\1", candidate))

    return out


def _extract_json(text: str) -> Any:
    m = re.search(r"<hierarchy>(.*?)</hierarchy>", text, re.DOTALL)
    payload = m.group(1).strip() if m else text.strip()
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.DOTALL).strip()

    last_error: Exception | None = None
    for candidate in _json_parse_candidates(payload):
        try:
            return json.loads(candidate)
        except Exception as e:
            last_error = e
            continue
    if last_error:
        raise last_error
    raise ValueError("No JSON payload found")


def _build_json_repair_prompt(raw_output: str, tasks: List[str]) -> str:
    """Ask the model to repair malformed hierarchy output into strict JSON."""
    task_lines = "\n".join(f"- {t}" for t in tasks)
    clipped_raw = str(raw_output or "").strip()
    if len(clipped_raw) > 12000:
        clipped_raw = clipped_raw[:12000] + "\n...[truncated]..."

    return f"""Fix the malformed hierarchy output below and return valid JSON only.

Output requirements:
- Return ONLY a JSON array wrapped in <hierarchy>...</hierarchy>.
- Max depth 2: group -> children.
- Keep leaf task wording exactly as given in the original tasks.
- Keep every original task exactly once, either as a leaf `name` or in `merged_from`.
- Keep `is_group: true` only for invented umbrella labels.
- Do not add commentary or markdown fences.

Original tasks:
{task_lines}

Malformed output:
<raw_output>
{clipped_raw}
</raw_output>
"""


# =============================================================================
# Dedicated LLM dedup pass (runs before the combined cluster+group pass)
# =============================================================================

def _build_dedup_prompt(tasks: List[str]) -> str:
    task_lines = "\n".join(f"{i}. {t}" for i, t in enumerate(tasks))
    return f"""You are a job analyst reviewing a worker's self-reported task statements.

## Tasks ({len(tasks)} total)
{task_lines}

---

## Your Job
Identify tasks that describe the **same core work activity** and should be merged.

Merge tasks when:
- They share the same action verb AND object domain (e.g. both are about writing grants)
- One is a specific instance of the other (attending standups IS sharing progress updates)
- Trailing purpose ("to <X>"), beneficiary ("for <X>"), or tool qualifiers ("using AI") do not make them distinct
- Separating them would not be meaningful to a job analyst or worker

Do NOT merge tasks that:
- Differ substantially in object (e.g. "review code" vs "review documentation")
- Use different core actions (e.g. "reading papers" vs "writing papers")
- Serve different audiences or have substantially different outcomes

For merged tasks, write a NEW statement that covers all merged sources without losing specificity.
Use the structure: <Action> <object> to <immediate outcome>.

## Output

Return ONLY a JSON array in <duplicates>...</duplicates>. One entry per OUTPUT task:
- Single (no merge):  {{"indices": [i]}}
- Merged group:       {{"indices": [i, j, ...], "merged_statement": "<new statement>", "merge_reason": "<brief>"}}

Indices are 0-based. Every input index must appear in exactly one entry. If there are no duplicates, list every task as its own singleton entry.

<duplicates>
[
  {{"indices": [0]}},
  {{"indices": [1, 4], "merged_statement": "Writing grants to secure research funding", "merge_reason": "bare form vs. 'to <purpose>'"}}
]
</duplicates>

No commentary outside the tags."""


def _extract_dedup_json(text: str) -> Any:
    m = re.search(r"<duplicates>(.*?)</duplicates>", text, re.DOTALL)
    payload = m.group(1).strip() if m else text.strip()
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.DOTALL).strip()
    return json.loads(payload)


def _dedup_tasks_llm(tasks: List[str], engine):
    """Run an LLM dedup pass using the 3b-style grouping format.

    The model returns one entry per OUTPUT task: singleton `{"indices":[i]}`
    or merged `{"indices":[i,j,...], "merged_statement": "..."}` — in the
    merged case the winner is a newly-synthesized canonical statement rather
    than picking the longest input. Returns `(kept_tasks, merge_map)` with the
    same shape as `_pre_dedup_by_core` so the maps compose cleanly.

    On any failure, returns the input unchanged and an empty map so downstream
    grouping still runs.
    """
    if len(tasks) <= 1:
        return list(tasks), {}
    try:
        response = invoke_engine(engine, _build_dedup_prompt(tasks))
        text = response.content if hasattr(response, "content") else str(response)
        groups = _extract_dedup_json(text)
        if not isinstance(groups, list):
            return list(tasks), {}

        kept: List[str] = []
        merge_map: Dict[str, List[str]] = {}
        seen_indices: Set[int] = set()

        for g in groups:
            if not isinstance(g, dict):
                continue
            raw_indices = g.get("indices") or []
            indices = [int(i) for i in raw_indices if isinstance(i, (int, float))]
            indices = [i for i in indices if 0 <= i < len(tasks) and i not in seen_indices]
            if not indices:
                continue

            if len(indices) == 1:
                kept.append(tasks[indices[0]])
                seen_indices.add(indices[0])
                continue

            merged_statement = str(g.get("merged_statement") or "").strip()
            if not merged_statement:
                # Model returned a merge group without a new statement —
                # fall back to the longest input variant in that group.
                merged_statement = max((tasks[i] for i in indices), key=len)
            kept.append(merged_statement)
            absorbed = [tasks[i] for i in indices if tasks[i] != merged_statement]
            if absorbed:
                merge_map.setdefault(merged_statement, []).extend(absorbed)
            seen_indices.update(indices)

        # Safety net: any input index the model omitted — treat as singleton.
        for i, t in enumerate(tasks):
            if i not in seen_indices:
                kept.append(t)

        if merge_map:
            print("[task_hierarchy] DEDUP pass 2 (LLM):")
            for keep_name, merged in merge_map.items():
                print(f"  kept: {keep_name}")
                for m in merged:
                    print(f"    ← merged: {m}")
        return kept, merge_map
    except Exception as e:
        print(f"[task_hierarchy] dedup pass failed: {e}")
        return list(tasks), {}


def _apply_dedup_merges_to_tree(
    tree: List[Dict[str, Any]],
    merge_map: Dict[str, List[str]],
) -> List[Dict[str, Any]]:
    """Inject pre-dedup merges into each matching leaf's `merged_from` list
    so the UI still shows what was absorbed."""
    if not merge_map or not isinstance(tree, list):
        return tree
    norm_map = {_norm(k): v for k, v in merge_map.items()}
    for node in tree:
        if not isinstance(node, dict):
            continue
        children = node.get("children") or []
        if children:
            _apply_dedup_merges_to_tree(children, merge_map)
        if node.get("is_group"):
            continue
        dropped = norm_map.get(_norm(str(node.get("name") or "")))
        if not dropped:
            continue
        existing = list(node.get("merged_from") or [])
        existing_norm = {_norm(m) for m in existing}
        for d in dropped:
            if _norm(d) not in existing_norm:
                existing.append(d)
                existing_norm.add(_norm(d))
        if existing:
            node["merged_from"] = existing
    return tree


# =============================================================================
# Validity screening (adapted from analysis/onet/3_aggregate_canonical_tasks.py)
# =============================================================================

def _build_screen_prompt(tasks: List[str]) -> str:
    task_list = "\n".join(f"{i}. {t}" for i, t in enumerate(tasks))
    return f"""You are a job analyst screening worker-reported task statements for validity.
Your goal is to keep only concrete, recurring activities the person CURRENTLY PERFORMS —
not goals, research agendas, aspirations, or descriptions of a domain they work in.

## Task statements ({len(tasks)} statements)
{task_list}

---

## Instructions

For each statement, apply these five checks:

1. **Currently performed?** — Is the activity itself something the person does in a typical week
   or month? Judge the action, not the outcome — writing a proposal about future work passes
   because writing is the current activity. Reject only when the whole statement describes a
   mission, research agenda, or aspiration rather than a discrete, recurring activity.
   Pass: "writing project proposals", "running experiments", "attending lab meetings",
         "designing and building language models to improve AI", "reading research papers"
   Fail: "advancing AI alignment" (pure aspiration — no observable action),
         "conducting research on human-centered AI" (ongoing mission, not a specific activity),
         "improving language models for populations" (outcome goal, not an activity)
   IMPORTANT: "designing and building [concrete artifact like models, systems, tools]" is a
   PASS — it has a specific action (design+build) and a concrete object (the artifact).
   "investigating/exploring [domain]" is borderline — rewrite it to "running experiments
   and analyzing results to understand [domain]" rather than rejecting outright.

2. **Specific action?** — Does it name a concrete, observable verb?
   Pass: "Debug", "Review", "Write", "Run", "Analyze", "Draft", "Read", "Attend", "Listen to",
         "Design", "Build", "Train", "Deploy", "Investigate" (when paired with concrete object)
   Fail: verbs so broad that a watching stranger couldn't tell what the person is physically doing.
   Common vague verbs that MUST be rewritten: "working on", "doing", "handling", "managing",
   "dealing with", "helping with", "leveraging" — always rewrite these to the specific action implied
   (e.g. "working on proposals" → "writing project proposals to secure funding";
    "leveraging human knowledge to improve AI" → reject: this describes a method, not an activity)

3. **Concrete object?** — Does it act on a specific artifact, system, document, or person —
   something you could point to or hand to someone?
   Pass: "experiment results", "research papers", "project proposal draft", "lab presentations",
         "language models", "AI systems", "codebases", "datasets"
   Fail: topic areas ("AI alignment", "human-centered systems"), abstract goals ("the field",
         "populations"), vague methods ("human knowledge", "AI capabilities")

4. **Bounded activity?** — Does it have a start and end, not a standing trait or disposition?
   Pass: a recurring or one-time activity that happens and finishes
   Fail: "Be a good communicator", "Understand the codebase", "Know the domain"

5. **Single task?** — One coherent unit of work, not 3+ unrelated activities bundled together.
   Pass: one meaningful unit (closely related sub-steps like "read and annotate" count as one)
   Fail: "Handle all aspects of onboarding, recruitment, and event planning"

## Rewriting

If a statement FAILS one or more checks but describes a real current work activity, fix it:
- **Preserve the user's own words as much as possible.** Keep their exact phrasing for the action and object wherever it already works — only change the minimum needed to satisfy the failing check.
- Replace vague verbs with the most specific action implied by the statement; prefer the user's verb if it is specific enough
- Narrow a vague object to the concrete artifact or person actually involved; use the user's own noun phrase if it already names something concrete
- Keep the purpose/objective the user stated; use format: <Action> <object> to <purpose>
- Use only details present in the original — do NOT fabricate, generalize, or substitute synonyms
- Keep under 20 words

If it bundles multiple distinct current activities, **split** it — return one object per activity.

If the statement describes a goal, aspiration, domain, or is truly unsalvageable, reject it.

## Output

Return ONLY a JSON array wrapped in <screen>...</screen>. Each input produces one or more objects:

[
  {{
    "index": <0-based index of original>,
    "status": "pass" | "rewritten" | "split" | "rejected",
    "rewritten": "<fixed statement if status is rewritten or split; empty string otherwise>",
    "reason": "<one sentence — what was fixed, split, or why rejected; empty if pass>"
  }},
  ...
]

No commentary outside the <screen> tags."""


def _extract_screen_json(text: str) -> Any:
    m = re.search(r"<screen>(.*?)</screen>", text, re.DOTALL)
    payload = m.group(1).strip() if m else text.strip()
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.DOTALL).strip()
    return json.loads(payload)


def _screen_tasks(tasks: List[str], engine) -> List[str]:
    """Screen and inline-rewrite task statements (O*NET-style single pass).

    Returns the list of valid task strings — originals that passed plus
    rewritten versions of those that could be fixed. Truly unsalvageable
    statements are dropped.  On any LLM/parse error returns the input list
    unchanged so the grouping pass still runs.
    """
    if len(tasks) <= 1:
        return list(tasks)
    try:
        response = invoke_engine(engine, _build_screen_prompt(tasks))
        text = response.content if hasattr(response, "content") else str(response)
        results = _extract_screen_json(text)
    except Exception:
        return list(tasks)

    if not isinstance(results, list):
        return list(tasks)

    kept: List[str] = []
    print("[task_hierarchy] SCREEN pass:")
    for r in results:
        if not isinstance(r, dict):
            continue
        idx = r.get("index")
        if not isinstance(idx, int) or idx >= len(tasks):
            continue
        status = str(r.get("status", "")).strip().lower()
        reason = r.get("reason", "")
        original = tasks[idx]
        rewritten = str(r.get("rewritten") or "").strip()

        if status == "pass":
            kept.append(original)
            print(f"  PASS: {original}")
        elif status in ("rewritten", "split") and rewritten:
            kept.append(rewritten)
            print(f"  REWRITE: {original!r}  →  {rewritten!r}")
            if reason:
                print(f"    ({reason})")
        else:
            suffix = f" — {reason}" if reason else ""
            print(f"  REJECT: {original}{suffix}")

    # Defensive: if everything got rejected, fall back to originals
    return kept if kept else list(tasks)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s or "").strip().lower())


def _to_impersonal_task_statement(name: str) -> str:
    """Rewrite a task string into an impersonal statement (no first-person words)."""
    original = re.sub(r"\s+", " ", str(name or "").strip())
    if not original:
        return ""
    s = original
    s = _FIRST_PERSON_CONTRACTIONS_RE.sub("", s)
    s = _FIRST_PERSON_PRONOUNS_RE.sub("", s)
    s = _LEADING_AUX_RE.sub("", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = re.sub(r"\s+", " ", s).strip(" ,.;:-")
    return s if s else original


def _rewrite_high_level_task_names(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize only top-level node names to impersonal task statements."""
    if not isinstance(tree, list):
        return tree
    for node in tree:
        if not isinstance(node, dict):
            continue
        raw_name = str(node.get("name") or "").strip()
        if not raw_name:
            continue
        node["name"] = _to_impersonal_task_statement(raw_name)
    return tree


def _normalize_grouping_feedback(text: str, max_len: int = 600) -> str:
    """Normalize free-text regrouping guidance to a compact prompt snippet."""
    compact = re.sub(r"\s+", " ", str(text or "").strip())
    return compact[:max_len]


def _is_ai_tool_task(name: str) -> bool:
    n = _norm(name)
    if not n:
        return False
    return bool(_AI_TOOL_HINT_RE.search(n) and _AI_USAGE_MARKER_RE.search(n))


def _strip_adjunct_tail(stem: str, pattern: "re.Pattern[str]", min_words: int = 2) -> str:
    """Strip a trailing adjunct clause only when the remaining stem still has
    at least `min_words` words — protects against "preparing for X" where the
    "for X" IS the object, not an adjunct."""
    candidate = pattern.sub("", stem).strip()
    if len(candidate.split()) >= min_words:
        return candidate
    return stem


def _normalize_verb_prefix(stem: str) -> str:
    """Replace a known verb-synonym prefix with its canonical form."""
    for key in _VERB_SYNONYM_KEYS:
        if stem == key:
            return _VERB_SYNONYMS[key]
        prefix = key + " "
        if stem.startswith(prefix):
            return _VERB_SYNONYMS[key] + " " + stem[len(prefix):]
    return stem


def _tool_agnostic_task_core(name: str) -> str:
    """Collapse mechanical variants of the same task to a shared canonical key.

    Deterministic transforms (applied in order):
      1. Lowercase, collapse whitespace.
      2. Strip leading/inline AI-tool qualifiers ("using Claude to ...").
      3. Drop "from scratch" / "from a blank slate" filler.
      4. Strip trailing purpose clause ("... to <X>").
      5. Strip trailing beneficiary clause ("... for <X>").
      6. Strip trailing means clause ("... using/via/by/through <X>").
      7. Normalize a small set of unambiguous verb synonyms.

    Each tail strip is guarded: the remaining stem must keep at least 2 words,
    so "preparing for presentations" does NOT collapse to "preparing".
    """
    n = _norm(name)
    if not n:
        return ""
    n = _AI_PREFIX_RE.sub("", n)
    n = _AI_INLINE_RE.sub("", n)
    n = _BLANK_SLATE_RE.sub("", n)
    n = re.sub(r"\s+", " ", n).strip()
    # Iterate because a task can have both a "to X" and a "for Y" suffix in
    # arbitrary order; strip whichever applies until stable.
    for _ in range(3):
        prev = n
        n = _strip_adjunct_tail(n, _PURPOSE_TAIL_RE)
        n = _strip_adjunct_tail(n, _BENEFICIARY_TAIL_RE)
        n = _strip_adjunct_tail(n, _MEANS_TAIL_RE)
        if n == prev:
            break
    n = _normalize_verb_prefix(n)
    # "<verb> for <object>" → "<verb> <object>" (grammatical-particle variant)
    n = _VERB_FOR_PARTICLE_RE.sub(r"\1 ", n).strip()
    return n


def _pre_dedup_by_core(tasks: List[str]):
    """Merge exact action+object duplicates deterministically, before any LLM pass.

    For tasks sharing the same tool-agnostic core, keep the longest variant
    (usually retains the most useful purpose context). Returns
    `(kept_tasks, merge_map)` where `merge_map[kept]` is the list of raw strings
    absorbed into it — same shape as `_dedup_tasks_llm`.
    """
    if len(tasks) <= 1:
        return list(tasks), {}
    groups: Dict[str, List[str]] = {}
    order: List[str] = []
    for t in tasks:
        core = _tool_agnostic_task_core(t) or _norm(t)
        if core not in groups:
            groups[core] = []
            order.append(core)
        groups[core].append(t)
    kept: List[str] = []
    merge_map: Dict[str, List[str]] = {}
    for core in order:
        members = groups[core]
        winner = max(members, key=len)
        kept.append(winner)
        dropped = [m for m in members if m is not winner]
        if dropped:
            merge_map.setdefault(winner, []).extend(dropped)
    return kept, merge_map


def _compose_merge_maps(
    first: Dict[str, List[str]],
    second: Dict[str, List[str]],
) -> Dict[str, List[str]]:
    """Compose two sequential merge passes. If `second` absorbs a string that
    was a winner in `first`, fold `first`'s absorbed list up to the new winner.
    """
    result: Dict[str, List[str]] = {k: list(v) for k, v in (first or {}).items()}
    for keep, dropped in (second or {}).items():
        bucket = result.setdefault(keep, [])
        for d in dropped or []:
            if d not in bucket:
                bucket.append(d)
            if d in result and d != keep:
                for x in result.pop(d):
                    if x not in bucket:
                        bucket.append(x)
    return result


def _cores_equivalent(a: str, b: str) -> bool:
    a = _norm(a)
    b = _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    # Conservative containment check (avoid tiny accidental overlaps).
    if min(len(a), len(b)) >= 10 and (a in b or b in a):
        return True
    ta, tb = a.split(), b.split()
    # Allow simple verb inflection differences ("write"/"writing") when object tail matches.
    if len(ta) == len(tb) and len(ta) >= 2 and ta[1:] == tb[1:]:
        return ta[0][:3] == tb[0][:3]
    return False


def _extract_tool_phrase(name: str) -> str:
    n = _norm(name)
    labels: List[str] = []
    if "chatgpt" in n:
        labels.append("ChatGPT")
    if "claude" in n:
        labels.append("Claude")
    if "copilot" in n:
        labels.append("GitHub Copilot")
    if "gemini" in n:
        labels.append("Gemini")
    if "cursor" in n:
        labels.append("Cursor")
    if "llm" in n or "llms" in n:
        labels.append("LLMs")
    if not labels:
        return "AI tools"
    # If specific tools were mentioned, keep only specific labels.
    return " and ".join(dict.fromkeys(labels))


def _merge_tool_into_task_name(base_name: str, ai_name: str) -> str:
    base = str(base_name or "").strip()
    if not base:
        return base
    if _AI_TOOL_HINT_RE.search(_norm(base)):
        return base
    tool = _extract_tool_phrase(ai_name)
    return f"{base} with {tool}"


def _dedup_list_preserve_order(items: List[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for it in items:
        key = _norm(it)
        if key and key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _merge_ai_tool_variants(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse AI-as-tool duplicates into a single leaf with tool qualifier.

    Example:
      - "writing code for experiments to support research"
      - "using AI tools to write code for experiments from scratch"
    becomes:
      - "writing code for experiments to support research with AI tools"
        merged_from includes both originals.
    """
    if not isinstance(nodes, list) or not nodes:
        return nodes

    # Recurse first so children are normalized before sibling-level merges.
    for n in nodes:
        if isinstance(n, dict):
            children = n.get("children")
            if isinstance(children, list) and children:
                n["children"] = _merge_ai_tool_variants(children)

    i = 0
    while i < len(nodes):
        a = nodes[i]
        if not isinstance(a, dict) or a.get("is_group"):
            i += 1
            continue
        j = i + 1
        while j < len(nodes):
            b = nodes[j]
            if not isinstance(b, dict) or b.get("is_group"):
                j += 1
                continue
            name_a = str(a.get("name") or "").strip()
            name_b = str(b.get("name") or "").strip()
            ai_a = _is_ai_tool_task(name_a)
            ai_b = _is_ai_tool_task(name_b)
            if ai_a == ai_b:
                j += 1
                continue

            core_a = _tool_agnostic_task_core(name_a)
            core_b = _tool_agnostic_task_core(name_b)
            if not _cores_equivalent(core_a, core_b):
                j += 1
                continue

            # Keep the non-AI leaf as the canonical base task; absorb the AI phrasing.
            winner, loser = (b, a) if ai_a else (a, b)
            winner_name_before = str(winner.get("name") or "").strip()
            loser_name = str(loser.get("name") or "").strip()

            merged_name = _merge_tool_into_task_name(winner_name_before, loser_name)
            if merged_name and merged_name != winner_name_before:
                winner["name"] = merged_name

            merged_from = [str(x).strip() for x in (winner.get("merged_from") or []) if str(x).strip()]
            # Preserve all source phrasings for coverage accounting.
            if winner_name_before and _norm(winner_name_before) != _norm(str(winner.get("name") or "")):
                merged_from.append(winner_name_before)
            if loser_name:
                merged_from.append(loser_name)
            merged_from.extend(
                str(x).strip()
                for x in (loser.get("merged_from") or [])
                if str(x).strip()
            )
            merged_from = _dedup_list_preserve_order(merged_from)
            if merged_from:
                winner["merged_from"] = merged_from

            if loser is a:
                nodes.pop(i)
                a = winner
                j = i + 1
                continue
            nodes.pop(j)
        i += 1

    return nodes


def _collect_covered(nodes: List[Dict[str, Any]]) -> Set[str]:
    """Return the set of normalized input-task strings the tree claims to cover.

    Group nodes contribute only via their `merged_from` (their `name` is
    invented and not an input task).
    """
    covered: Set[str] = set()
    for n in nodes or []:
        if not isinstance(n, dict):
            continue
        if not n.get("is_group"):
            name = n.get("name")
            if name:
                covered.add(_norm(name))
        for merged in n.get("merged_from") or []:
            if merged:
                covered.add(_norm(merged))
        covered |= _collect_covered(n.get("children") or [])
    return covered


def _sanitize(
    nodes: Any,
    verbatim: Dict[str, str] = None,
    depth: int = 0,
) -> List[Dict[str, Any]]:
    """Coerce LLM output into the shape the frontend expects.

    - Restores verbatim casing on leaf names using the `verbatim` lookup
      (normalised key → original string), so Title-Casing by the LLM is undone.
    - Demotes invented group nodes that end up with fewer than 2 children:
      their children are promoted to top-level leaves instead.
    """
    if verbatim is None:
        verbatim = {}
    out: List[Dict[str, Any]] = []
    if not isinstance(nodes, list):
        return out
    for n in nodes:
        if not isinstance(n, dict):
            continue
        raw_name = str(n.get("name") or "").strip()
        if not raw_name:
            continue

        is_group = bool(n.get("is_group"))
        # Enforce max depth of 2 (root -> children). Any deeper structure is flattened.
        children = [] if depth >= 1 else _sanitize(n.get("children"), verbatim, depth + 1)

        # Demote single-child invented groups — promote the child to top level
        if is_group and len(children) < 2:
            out.extend(children)
            continue

        # Restore verbatim casing for non-group leaves
        if not is_group:
            raw_name = verbatim.get(_norm(raw_name), raw_name)

        node: Dict[str, Any] = {"name": raw_name, "children": children}
        if is_group:
            node["is_group"] = True
        merged = n.get("merged_from") or []
        if isinstance(merged, list):
            cleaned_merged = [
                verbatim.get(_norm(str(m).strip()), str(m).strip())
                for m in merged if str(m).strip()
            ]
            cleaned_merged = _dedup_list_preserve_order(cleaned_merged)
            if cleaned_merged:
                node["merged_from"] = cleaned_merged
        out.append(node)
    return out


def _flat_fallback(tasks: List[str]) -> List[Dict[str, Any]]:
    return [{"name": t, "children": []} for t in tasks]


def _dedupe_leaf_nodes(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate identical leaves globally while preserving first occurrence."""
    seen: Dict[str, Dict[str, Any]] = {}

    def _walk(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for node in nodes or []:
            if not isinstance(node, dict):
                continue
            if node.get("is_group"):
                children = _walk(node.get("children") or [])
                if len(children) >= 2:
                    node["children"] = children
                    out.append(node)
                else:
                    out.extend(children)
                continue

            name = str(node.get("name") or "").strip()
            key = _norm(name)
            if not key:
                continue
            merged = [
                str(x).strip()
                for x in (node.get("merged_from") or [])
                if str(x).strip() and _norm(str(x)) != key
            ]
            merged = _dedup_list_preserve_order(merged)
            if merged:
                node["merged_from"] = merged
            elif "merged_from" in node:
                node.pop("merged_from", None)

            winner = seen.get(key)
            if winner is None:
                seen[key] = node
                out.append(node)
                continue

            winner_merged = [
                str(x).strip() for x in (winner.get("merged_from") or []) if str(x).strip()
            ]
            winner_merged.append(name)
            winner_merged.extend(merged)
            winner_merged = [
                x for x in _dedup_list_preserve_order(winner_merged)
                if _norm(x) != key
            ]
            if winner_merged:
                winner["merged_from"] = winner_merged
        return out

    return _walk(tree)


def _collapse_core_duplicates(tree: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Post-clustering backstop: collapse parent↔child and sibling↔sibling
    near-duplicates detected by core equivalence (same action+object after
    stripping purpose/beneficiary/means clauses).

    Catches cases both dedup passes missed — e.g. when the clustering LLM puts
    "writing grants to secure funding" and "writing grants" into a parent/child
    relationship instead of merging them. Groups (invented umbrella labels)
    are left alone: their name is intentionally abstract.
    """
    def _merge_sibling_bucket(siblings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        order: List[str] = []
        for node in siblings or []:
            if not isinstance(node, dict):
                continue
            name = str(node.get("name") or "").strip()
            if not name or node.get("is_group"):
                key = f"__uniq_{id(node)}__"
            else:
                key = _tool_agnostic_task_core(name) or _norm(name)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(node)

        out: List[Dict[str, Any]] = []
        for key in order:
            bucket = groups[key]
            if len(bucket) == 1:
                out.append(bucket[0])
                continue
            winner = max(bucket, key=lambda n: len(str(n.get("name") or "")))
            absorbed_names: List[str] = list(winner.get("merged_from") or [])
            winner_children = list(winner.get("children") or [])
            for node in bucket:
                if node is winner:
                    continue
                n_name = str(node.get("name") or "").strip()
                if n_name and n_name != winner.get("name") and n_name not in absorbed_names:
                    absorbed_names.append(n_name)
                for mf in node.get("merged_from") or []:
                    s = str(mf).strip()
                    if s and s != winner.get("name") and s not in absorbed_names:
                        absorbed_names.append(s)
                winner_children.extend(node.get("children") or [])
            if absorbed_names:
                winner["merged_from"] = absorbed_names
            if winner_children:
                winner["children"] = winner_children
            out.append(winner)
        return out

    def _walk(nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not isinstance(nodes, list):
            return nodes
        nodes = _merge_sibling_bucket(nodes)
        for node in nodes:
            if not isinstance(node, dict):
                continue
            children = node.get("children") or []
            if not children:
                continue
            children = _walk(children)

            parent_name = str(node.get("name") or "").strip()
            if parent_name and not node.get("is_group"):
                parent_core = _tool_agnostic_task_core(parent_name) or _norm(parent_name)
                survivors: List[Dict[str, Any]] = []
                absorbed = list(node.get("merged_from") or [])
                for child in children:
                    c_name = str(child.get("name") or "").strip()
                    if child.get("is_group") or not c_name:
                        survivors.append(child)
                        continue
                    c_core = _tool_agnostic_task_core(c_name) or _norm(c_name)
                    if c_core == parent_core:
                        if c_name != parent_name and c_name not in absorbed:
                            absorbed.append(c_name)
                        for mf in child.get("merged_from") or []:
                            s = str(mf).strip()
                            if s and s != parent_name and s not in absorbed:
                                absorbed.append(s)
                        for gc in child.get("children") or []:
                            survivors.append(gc)
                    else:
                        survivors.append(child)
                if absorbed:
                    node["merged_from"] = absorbed
                children = survivors
            node["children"] = children
        return nodes

    return _walk(tree)


def _apply_coverage_safety_net(tree: List[Dict[str, Any]], tasks: List[str]) -> List[Dict[str, Any]]:
    covered = _collect_covered(tree)
    for task in tasks:
        if _norm(task) not in covered:
            tree.append({"name": task, "children": []})
            covered.add(_norm(task))
    return tree


def _grouping_quality_score(tree: List[Dict[str, Any]]) -> float:
    """Simple structural score to compare two candidate hierarchies."""
    if not isinstance(tree, list):
        return -1.0
    groups = [
        n for n in tree
        if isinstance(n, dict) and n.get("is_group") and len(n.get("children") or []) >= 2
    ]
    top_level_leaves = [
        n for n in tree if isinstance(n, dict) and not n.get("is_group")
    ]
    # Prefer meaningful grouping and avoid too many top-level loose leaves.
    return float(len(groups) * 3 - max(0, len(top_level_leaves) - 4))


def _is_descriptive_group_label(name: str) -> bool:
    """Heuristic quality check for invented group labels.

    Group labels should read like task statements with action+object+purpose.
    """
    label = _norm(name)
    if not label:
        return False
    if " to " not in label:
        return False
    left, right = label.split(" to ", 1)
    if not left.strip() or not right.strip():
        return False
    # Avoid very short noun-phrase labels (e.g., "meetings and talks").
    if len(label.split()) < 6:
        return False
    # Require at least one action-like token before the purpose clause.
    left_tokens = left.split()
    if not any(t.endswith("ing") for t in left_tokens):
        return False
    return True


def _has_underspecified_group_labels(tree: List[Dict[str, Any]]) -> bool:
    for n in tree:
        if not isinstance(n, dict):
            continue
        if n.get("is_group") and not _is_descriptive_group_label(str(n.get("name") or "")):
            return True
    return False


def _needs_smart_refine(tree: List[Dict[str, Any]], tasks: List[str]) -> bool:
    """Decide when to run an extra smart regrouping pass."""
    if not isinstance(tree, list):
        return False
    # Always attempt one repair pass for vague group labels.
    if _has_underspecified_group_labels(tree):
        return True
    if len(tasks) < 5:
        return False
    groups = sum(1 for n in tree if isinstance(n, dict) and n.get("is_group"))
    top_level_leaves = sum(1 for n in tree if isinstance(n, dict) and not n.get("is_group"))
    # Trigger only when the result looks under-grouped.
    return top_level_leaves >= 5 or (groups <= 1 and top_level_leaves >= 3)


def _build_refine_prompt(
    tasks: List[str],
    tree: List[Dict[str, Any]],
    grouping_feedback: str = "",
) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    tree_json = json.dumps(tree, ensure_ascii=False, indent=2)
    feedback = _normalize_grouping_feedback(grouping_feedback)
    feedback_section = (
        f"""
User regrouping feedback:
- {feedback}
- Respect this feedback when reclustering and naming groups.
"""
        if feedback
        else ""
    )
    return f"""Improve this task hierarchy for better bucket quality while preserving task fidelity.

Return ONLY JSON wrapped in <hierarchy>...</hierarchy>.

Rules:
- Keep every original task exactly once (as a node `name` or in `merged_from`).
- Do NOT invent new leaf tasks and do NOT rewrite leaf wording.
- Max depth 2 (group -> children). No grandchildren.
- Create groups only when they have at least 2 children.
- Prefer coherent top-level buckets over many flat leaves.
- Group same-workflow lifecycle siblings (e.g., running experiments + analyzing experiment results).
- Meeting/talk activities, including meeting preparation, should generally share one meetings bucket unless clearly unrelated.
- Different document types (proposal vs submission) remain separate leaves, but can share a parent writing bucket.
- Every invented group label (`is_group: true`) MUST be a descriptive task statement with action + object + purpose in this exact format: "<action> <object> to <purpose>".
- Group labels must be specific and readable (typically 8-18 words) and must not be short noun phrases like "Meetings and talks", "Experimentation", or "Writing".
- High-level labels must be impersonal task statements with no first-person wording (remove "I", "my", "we", "our").
- You may rewrite group labels to satisfy the rule above, but do not rewrite leaf task wording.
- For AI-tool variants of the same underlying work, keep ONE task and encode tool usage as a suffix qualifier: "<action> <object> to <purpose> with <tool>".

{feedback_section}

Original tasks:
{task_lines}

Current hierarchy:
<current_hierarchy>
{tree_json}
</current_hierarchy>"""


def _smart_refine_grouping(
    tree: List[Dict[str, Any]],
    tasks: List[str],
    engine,
    verbatim: Dict[str, str],
    grouping_feedback: str = "",
) -> List[Dict[str, Any]]:
    """Ask the model to repair under-grouped structures and keep the better tree."""
    try:
        response = invoke_engine(
            engine,
            _build_refine_prompt(tasks, tree, grouping_feedback=grouping_feedback),
        )
        text = response.content if hasattr(response, "content") else str(response)
        refined = _sanitize(_extract_json(text), verbatim)
        if not refined:
            return tree

        # Same cleanup used on first-pass output.
        child_names = {
            _norm(c["name"])
            for n in refined if n.get("is_group")
            for c in n.get("children") or []
        }
        refined = [
            n for n in refined
            if n.get("is_group") or _norm(n["name"]) not in child_names
        ]

        covered = _collect_covered(refined)
        for task in tasks:
            if _norm(task) not in covered:
                refined.append({"name": task, "children": []})

        return refined if _grouping_quality_score(refined) >= _grouping_quality_score(tree) else tree
    except Exception:
        return tree


def _invoke_and_parse_hierarchy(
    engine,
    prompt: str,
    tasks: List[str],
) -> Any:
    """Invoke hierarchy model and repair once if initial JSON is malformed."""
    response = invoke_engine(engine, prompt)
    text = response.content if hasattr(response, "content") else str(response)
    try:
        return _extract_json(text)
    except Exception:
        repair_response = invoke_engine(
            engine,
            _build_json_repair_prompt(text, tasks),
        )
        repair_text = (
            repair_response.content
            if hasattr(repair_response, "content")
            else str(repair_response)
        )
        return _extract_json(repair_text)


def organize_tasks(
    tasks: List[str],
    model_name: str = "gpt-5.1",
    screen: bool = False,
    grouping_feedback: str = "",
) -> List[Dict[str, Any]]:
    """Return a tree of nodes (dedup + group, optionally with screening in a single LLM call).

    screen=False (default): tasks are assumed already clean (e.g. from portrait generation).
    screen=True: adds an O*NET-style validity/rewrite pass before grouping.
    On any failure, returns the input as a flat list of leaves.
    """
    cleaned = [str(t).strip() for t in tasks if str(t).strip()]
    if len(cleaned) <= 1:
        return _flat_fallback(cleaned)

    try:
        engine = get_engine(model_name)
    except Exception:
        return _flat_fallback(cleaned)

    # Dedup in two passes. Preserve verbatim strings for all inputs (including
    # dropped) so downstream cache/merged_from still round-trip correctly.
    verbatim = {_norm(t): t for t in cleaned}
    # Pass 1: deterministic core-based dedup — catches mechanical variants
    # ("writing grants to secure funding" ≡ "writing grants") without needing
    # the LLM to cooperate.
    cleaned, det_merge_map = _pre_dedup_by_core(cleaned)
    if det_merge_map:
        print("[task_hierarchy] DEDUP pass 1 (deterministic):")
        for keep, merged in det_merge_map.items():
            print(f"  kept: {keep}")
            for m in merged:
                print(f"    ← merged: {m}")
    # Pass 2: LLM dedup — catches true paraphrase cases the deterministic pass
    # can't reach (synonymous verbs, paraphrased objects).
    cleaned, llm_merge_map = _dedup_tasks_llm(cleaned, engine)
    dedup_merge_map = _compose_merge_maps(det_merge_map, llm_merge_map)

    try:
        normalized_feedback = _normalize_grouping_feedback(grouping_feedback)
        if len(cleaned) <= 1:
            tree = _flat_fallback(cleaned)
            tree = _apply_dedup_merges_to_tree(tree, dedup_merge_map)
            return tree
        tree = _sanitize(
            _invoke_and_parse_hierarchy(
                engine,
                _build_prompt(
                    cleaned,
                    screen=screen,
                    grouping_feedback=normalized_feedback,
                ),
                cleaned,
            ),
            verbatim,
        )
        if not tree:
            # Model-first fallback: ask for a clean regroup starting from a flat seed.
            flat_seed = _flat_fallback(cleaned)
            tree = _sanitize(
                _invoke_and_parse_hierarchy(
                    engine,
                    _build_refine_prompt(
                        cleaned,
                        flat_seed,
                        grouping_feedback=normalized_feedback,
                    ),
                    cleaned,
                ),
                verbatim,
            )
        if not tree:
            return _flat_fallback(cleaned)

        # Remove top-level leaves already present as children of a group node.
        child_names = {
            _norm(c["name"])
            for n in tree if n.get("is_group")
            for c in n.get("children") or []
        }
        tree = [
            n for n in tree if n.get("is_group") or _norm(n["name"]) not in child_names
        ]

        # Structural cleanup before optional refine.
        tree = _dedupe_leaf_nodes(tree)
        tree = _apply_coverage_safety_net(tree, cleaned)

        # Optional pass-3 smart repair (disabled by default for latency).
        if _RUN_REFINE_PASS and _needs_smart_refine(tree, cleaned):
            tree = _smart_refine_grouping(
                tree=tree,
                tasks=cleaned,
                engine=engine,
                verbatim=verbatim,
                grouping_feedback=normalized_feedback,
            )

        # Final cleanup for AI-tool variants and duplicate leaves.
        tree = _merge_ai_tool_variants(tree)
        tree = _dedupe_leaf_nodes(tree)
        # Backstop: collapse parent↔child / sibling↔sibling near-duplicates
        # that both dedup passes missed (e.g. clustering LLM nested a task
        # under its own rephrasing).
        tree = _collapse_core_duplicates(tree)
        tree = _apply_coverage_safety_net(tree, cleaned)

        # Fold pre-dedup drops into the matching leaf's merged_from so the UI
        # still reflects what was absorbed.
        tree = _apply_dedup_merges_to_tree(tree, dedup_merge_map)

        # Presentation normalization: top-level nodes should be written as
        # impersonal task statements (no first-person phrasing).
        tree = _rewrite_high_level_task_names(tree)

        print("[task_hierarchy] GROUP pass:")
        for n in tree:
            if n.get("is_group"):
                print(f"  [group] {n['name']}")
                for c in n.get("children") or []:
                    merged = c.get("merged_from") or []
                    suffix = f"  ← merged: {merged}" if merged else ""
                    print(f"    {c['name']}{suffix}")
            else:
                merged = n.get("merged_from") or []
                suffix = f"  ← merged: {merged}" if merged else ""
                print(f"  {n['name']}{suffix}")

        return tree
    except Exception:
        return _flat_fallback(cleaned)
