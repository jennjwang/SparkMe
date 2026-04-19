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


def _build_prompt(tasks: List[str]) -> str:
    task_lines = "\n".join(f"- {t}" for t in tasks)
    criteria_block = "\n".join(f"- {c}" for c in _load_criteria())
    return f"""You are organizing a flat list of work tasks into a clean, deduplicated hierarchy.

Do THREE things, in order:

================================================================
STEP 0 — IDENTIFY CLUSTERS (do this in the <thinking> block before writing output)
================================================================
For each task extract:
  (a) Primary action verb  ("running", "analyzing", "attending", "writing", …)
  (b) Object domain        (the general category of thing being acted on:
                            "experiments/data", "meetings", "documents", …)

Then form clusters using this hybrid rule:

  1. Start with action-verb clusters: tasks that share the same verb are candidates
     for one group.

  2. Merge verb-clusters whose object domains significantly overlap. Two verb-clusters
     belong together when their objects are different facets of the same underlying
     thing — not just loosely related. Two triggers for merging:

     (a) Same specific object — tasks that act on the exact same artifact or event
         belong in one cluster regardless of action verb.
         Example: "preparing for advisor meetings" and "attending advisor meetings"
         → same specific object (advisor meetings) → merge into one cluster.
         Example: "running experiments" and "analyzing experimental data"
         → same underlying artifact (experiments/data) → merge into one cluster.

     (b) Same object domain across verbs — tasks whose objects are different instances
         of the same category belong together.
         Example: "attending advisor meetings" and "meeting with peers"
         → both involve human gatherings → merge into one cluster.

     Counter-example: "reading papers" and "writing papers" → same object class but
     opposite directions of work → keep separate.

  3. For each final cluster, identify three things:
       - Shared action(s): the verb(s) that cover the cluster
       - Shared object domain: the general thing being acted on
       - Shared objective: the common purpose or outcome across all tasks in the cluster

     Name the cluster: "<action(s)> <object domain> to <shared objective>"
     Examples:
       "running"+"analyzing" / "experiments" / "generate and interpret findings"
         → "Running and analyzing experiments to generate findings"
       "attending"+"meeting with" / "meetings and peers" / "exchange updates and feedback"
         → "Attending meetings to exchange updates and feedback"
       "writing" / "documents" / "communicate and plan research"
         → "Writing research documents to communicate and plan work"
     Keep it under 10 words, sentence case.

================================================================
STEP 1 — DEDUPLICATE
================================================================
Merge tasks that describe the same underlying activity. For each merged group,
keep ONE canonical task (the most descriptive verbatim entry from the input)
and record the others under `merged_from`.

CORE MERGE RULE: two tasks are the same activity only when they share BOTH the
same action AND the same object. Sharing only one is NOT sufficient to merge.
  - Same action, different object → SEPARATE tasks (keep both).
    e.g. "attending advisor meetings" vs "attending lab meetings" — both attend,
    but different objects (advisor meeting vs lab meeting) → DO NOT merge.
  - Same object, different action → SEPARATE tasks (keep both).
    e.g. "preparing for advisor meetings" vs "attending advisor meetings" —
    same object, but preparing ≠ attending → DO NOT merge.
  - Same action AND same object → merge, keep the more specific phrasing.

Additional patterns to merge (only when action+object both match):
  (a) Pure synonyms / rephrasings of the same activity.
  (b) One task is a sub-aspect or stage of another with the same action+object.
      Example: "Give presentations" absorbs "Convincingly communicate findings
      during presentations" — same action (give/present), same object (findings
      in a presentation).
  (c) "Use <AI tool> to assist with X" → merge INTO "X" (the tool is a how,
      not a separate task).
  (d) Two tasks describe the same action+object at different specificities —
      keep the more specific verbatim entry.

Do NOT merge tasks that share a topic or category but have different
action+object pairs (e.g. "read papers" and "write papers", or "attending
advisor meetings" and "attending talks"). In particular: preparing for X and
attending X are ALWAYS different tasks — different actions — never merge them.
(They can be grouped together in STEP 2 because they share the same object,
but they must remain separate leaf nodes.)

================================================================
STEP 2 — GROUP (nest into a hierarchy)
================================================================
Use the clusters you identified in STEP 0 directly as groups. Each cluster with
2+ tasks becomes one group node; single-task clusters stay as top-level leaves.

Group label format:
  - Use the cluster name derived in STEP 0: "<action(s)> <object domain> to <shared objective>"
  - Sentence case (only first word capitalised), under 10 words
  - Must contain action, object domain, and objective — never a pure noun phrase

Two kinds of group parents:
  (i)  An existing input task that is clearly the umbrella for the cluster.
       Use its verbatim wording as the parent name.
  (ii) A NEW invented label when no input task is the natural umbrella.
       Mark with `"is_group": true`.

Rules:
- Only create a group when it has at least 2 children.
- Each leaf must use the verbatim input wording (after dedup).
- At most 2 levels (group → children). No grandchildren.

================================================================
OUTPUT FORMAT
================================================================
Return ONLY a JSON array of nodes wrapped in <hierarchy>...</hierarchy>.
Each node has this shape:

{{
  "name": "<verbatim input task, OR a short invented group label>",
  "is_group": <true if this is an invented umbrella label, else omit or false>,
  "merged_from": ["<other input tasks merged into this one>", ...],
  "children": [<zero or more nodes, leaves only — no further nesting>]
}}

Coverage requirement: every input task must appear EXACTLY ONCE across the
whole tree, either as some node's `name` or inside some node's `merged_from`.

Input tasks:
{task_lines}

Before writing the JSON, show your STEP 0 action-cluster analysis inside
<thinking>...</thinking> tags, then output the JSON inside <hierarchy>...</hierarchy>."""


def _extract_json(text: str) -> Any:
    m = re.search(r"<hierarchy>(.*?)</hierarchy>", text, re.DOTALL)
    payload = m.group(1).strip() if m else text.strip()
    payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.DOTALL).strip()
    return json.loads(payload)


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
   Pass: "writing project proposals", "running experiments", "attending lab meetings"
   Fail: "advancing AI alignment", "conducting research on human-centered AI" (ongoing mission,
   not a specific activity), "improving language models for populations" (aspiration)

2. **Specific action?** — Does it name a concrete, observable verb?
   Pass: "Debug", "Review", "Write", "Run", "Analyze", "Draft", "Read", "Attend", "Listen to"
   Fail: verbs so broad that a watching stranger couldn't tell what the person is physically doing.
   Common vague verbs that MUST be rewritten: "working on", "doing", "handling", "managing",
   "dealing with", "helping with" — always rewrite these to the specific action implied
   (e.g. "working on proposals" → "writing project proposals to secure funding")

3. **Concrete object?** — Does it act on a specific artifact, system, document, or person —
   something you could point to or hand to someone?
   Pass: "experiment results", "research papers", "project proposal draft", "lab presentations"
   Fail: topic areas ("AI", "human-centered systems"), abstract goals ("the field", "populations")

4. **Bounded activity?** — Does it have a start and end, not a standing trait or disposition?
   Pass: a recurring or one-time activity that happens and finishes
   Fail: "Be a good communicator", "Understand the codebase", "Know the domain"

5. **Single task?** — One coherent unit of work, not 3+ unrelated activities bundled together.
   Pass: one meaningful unit (closely related sub-steps like "read and annotate" count as one)
   Fail: "Handle all aspects of onboarding, recruitment, and event planning"

## Rewriting

If a statement FAILS one or more checks but describes a real current work activity, fix it:
- Replace vague verbs with the most specific action implied by the statement
- Narrow a vague object to the concrete artifact or person actually involved
- Keep the purpose/objective; use format: <Action> <object> to <purpose>
- Use only details present in the original — do NOT fabricate
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


def _sanitize(nodes: Any, verbatim: Dict[str, str] = None) -> List[Dict[str, Any]]:
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
        children = _sanitize(n.get("children"), verbatim)

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
            if cleaned_merged:
                node["merged_from"] = cleaned_merged
        out.append(node)
    return out


def _flat_fallback(tasks: List[str]) -> List[Dict[str, Any]]:
    return [{"name": t, "children": []} for t in tasks]


def organize_tasks(tasks: List[str], model_name: str = "gpt-4.1-mini") -> List[Dict[str, Any]]:
    """Return a tree of nodes.

    Pipeline: ONET-style validity screen → dedup + group via single LLM call.
    On any failure, returns the (screened) input as a flat list of leaves.
    """
    cleaned = [str(t).strip() for t in tasks if str(t).strip()]
    if len(cleaned) <= 1:
        return _flat_fallback(cleaned)

    try:
        engine = get_engine(model_name)
    except Exception:
        return _flat_fallback(cleaned)

    screened = _screen_tasks(cleaned, engine)
    if len(screened) <= 1:
        return _flat_fallback(screened)

    try:
        verbatim = {_norm(t): t for t in screened}
        response = invoke_engine(engine, _build_prompt(screened))
        text = response.content if hasattr(response, 'content') else str(response)
        tree = _sanitize(_extract_json(text), verbatim)
        if not tree:
            return _flat_fallback(screened)

        # Remove top-level leaves already present as children of a group node
        child_names = {
            _norm(c["name"])
            for n in tree if n.get("is_group")
            for c in n.get("children") or []
        }
        tree = [n for n in tree if n.get("is_group") or _norm(n["name"]) not in child_names]

        covered = _collect_covered(tree)
        appended = []
        for task in screened:
            if _norm(task) not in covered:
                tree.append({"name": task, "children": []})
                appended.append(task)

        print("[task_hierarchy] GROUP pass:")
        def _print_tree(nodes, indent=0):
            for n in nodes:
                prefix = "  " * indent
                label = f"[group] {n['name']}" if n.get("is_group") else n["name"]
                merged = n.get("merged_from")
                merged_str = f"  ← merged: {merged}" if merged else ""
                print(f"{prefix}  {label}{merged_str}")
                _print_tree(n.get("children") or [], indent + 1)
        _print_tree(tree)
        if appended:
            print(f"[task_hierarchy] Appended as uncovered leaves: {appended}")

        return tree
    except Exception:
        return _flat_fallback(screened)
