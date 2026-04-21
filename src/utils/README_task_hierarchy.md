# task_hierarchy

Converts a flat list of task strings (from the user portrait's `Task Inventory`) into a two-level hierarchy for display in the time-split widget and profile panel.

## Entry point

```python
from src.utils.task_hierarchy import organize_tasks

tree = organize_tasks(tasks, model_name="gpt-4.1-mini")
```

`tasks` — list of strings from `lastPortrait["Task Inventory"]`  
Returns a list of node dicts (see [Output shape](#output-shape)).

## Pipeline

### Pass 0 — Screen

Filters the raw task list before grouping. Each statement is evaluated against five checks adapted from the O*NET task validity criteria:

1. **Currently performed** — a discrete recurring activity, not a mission statement or aspiration
2. **Specific action** — concrete observable verb (write, run, attend, analyze…), not "working on" / "handling" / "leveraging"
3. **Concrete object** — a specific artifact, system, or person; not a topic area or abstract goal
4. **Bounded activity** — has a start and end, not a standing trait
5. **Single task** — one coherent unit of work, not three unrelated things bundled together

Outcomes per statement:
- **pass** — kept verbatim
- **rewritten** — vague verb or object fixed; kept as rewritten form
- **split** — bundled statement broken into multiple tasks; each kept separately
- **rejected** — mission statement, aspiration, or unsalvageable; dropped

If the LLM call fails or rejects everything, the original list is passed through unchanged so grouping still runs.

Console output prefix: `[task_hierarchy] SCREEN pass:`

### Pass 1 — Deduplicate (inside the group LLM call)

Merge rule: two tasks are the same activity only when they share **both** the same action **and** the same object.

- Same action, different object → keep both (`attending advisor meetings` ≠ `attending lab meetings`)
- Same object, different action → keep both (`preparing for advisor meetings` ≠ `attending advisor meetings`)
- `Use <AI tool> to do X` → merged into `X` (tool is a *how*, not a separate task)
- Different document types are different objects: `writing project proposals` ≠ `writing submission manuscripts` — do not merge, only group

The canonical task kept is the most descriptive verbatim input entry. Merged entries are recorded in `merged_from`.

### Pass 2 — Group

Uses the clusters identified in Pass 1's thinking step. Two clustering triggers:

**(a) Same specific object** — tasks acting on the exact same artifact cluster together regardless of verb.  
Example: `preparing for advisor meetings` + `attending advisor meetings` → one group.

**(b) Same object domain** — tasks whose objects are different instances of the same category cluster together.  
Example: `attending lab meetings` + `attending project meetings` + `meeting with peers` + `attending research talks` → one group (all are gatherings).

Note: forming a specific-object cluster (a) does not exhaust the domain. Remaining tasks in the same broad domain still form their own cluster under (b).

Counter-example: `reading papers` and `writing papers` share an object class but are opposite directions of work — kept separate.

Group label format: `"<action(s)> <object domain> to <shared objective>"` — sentence case, under 10 words, always includes action + object + objective (never a pure noun phrase).

Groups with fewer than 2 children are demoted (children promoted to top-level leaves).

### Validation

After grouping, every screened task must appear exactly once in the tree — either as a node name or in a node's `merged_from` list. Any task not covered by the LLM output is appended as a top-level leaf.

Console output prefix: `[task_hierarchy] GROUP pass:`

## Output shape

```json
[
  {
    "name": "Running and analyzing experiments to generate findings",
    "is_group": true,
    "children": [
      { "name": "running experiments to generate research data", "merged_from": ["using Claude Code to run experiments"], "children": [] },
      { "name": "analyzing experiment results to understand model behavior", "children": [] }
    ]
  },
  {
    "name": "reading research papers to understand the state of the field",
    "children": []
  }
]
```

| Field | Description |
|---|---|
| `name` | Verbatim input task (leaf) or invented group label |
| `is_group` | `true` only on invented umbrella labels; omitted on real input tasks |
| `merged_from` | Other input tasks merged into this leaf; omitted when empty |
| `children` | Leaf nodes under a group; empty list on leaves |

Maximum depth is 2 (group → children). No grandchildren.

## Similarity criteria

The dedup merge rules are loaded from `analysis/task_clustering/criteria.txt` at runtime (cached). If that file is missing, a set of defaults is used. This keeps the widget and the offline analysis pipeline aligned on what counts as the "same task."

## Error handling

Every pass has a fallback:
- Screen LLM error → pass all inputs through unchanged
- Group LLM error or parse error → return screened inputs as a flat list of leaves
- Coverage gap → missing tasks appended as top-level leaves

The function never raises; callers always get a usable list.
