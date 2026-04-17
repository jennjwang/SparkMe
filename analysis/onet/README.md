# O*NET Analysis Pipeline

Compares worker-reported task statements from the SparkMe user study against O*NET 30.2 task databases to classify how much of what workers actually do is captured by existing occupational frameworks — and what is genuinely new.

---

## Pipeline Overview

```
user_study/<id>/memory_bank_content.json
        │
        ▼
analysis/onet/1_generate_tasks_from_study.py
        │  extracts occupation, sector, industry + structured task objects from interview memories
        ▼
analysis/onet/data/study_tasks.json
        │
        ▼
analysis/onet/2_map_onet_occupations.py
        │  maps each participant to an O*NET-SOC code
        ▼
analysis/onet/data/study_tasks.json  (+onet_code, onet_title)
        │
        ▼
analysis/onet/3_aggregate_canonical_tasks.py
        │  aggregates raw task statements into a canonical list
        │  (calls 3a_rewrite_task_statements.py, 3b_dedup_canonical_tasks.py)
        ▼
analysis/onet/data/canonical_tasks.json
        │
        ▼
analysis/onet/4_compare_onet_tasks.py
        │  classifies each canonical task against O*NET
        ▼
analysis/onet/data/onet_comparison.json
        │
        ▼
analysis/onet/5_visualize_onet_comparison.py
        │
        ▼
analysis/onet/data/onet_comparison_viewer.html
```

---

## Step 1 — Task Extraction

**Script:** `analysis/onet/1_generate_tasks_from_study.py`  
**Reads:** `user_study/<id>/memory_bank_content.json` (per participant)  
**Writes:** `analysis/onet/data/study_tasks.json`

Each participant's memory bank is built by the SparkMe interview system as the interview runs — the session scribe agent converts the interviewee's responses into structured memory notes, each tagged to a subtopic. Task extraction filters to only memories tagged to subtopics **2.x** (core responsibilities) and **3.x** (task proficiency, challenge, engagement):


| Subtopic | Content                                                           |
| -------- | ----------------------------------------------------------------- |
| 2.1      | Primary job responsibilities and regular daily tasks              |
| 2.2      | Approximate proportion of time spent on core activities           |
| 2.3      | Level of autonomy and scope of decision-making                    |
| 2.4      | Additional responsibilities or tools relevant to decision-making  |
| 3.1      | Tasks that feel easiest or most natural                           |
| 3.2      | Tasks perceived as most challenging or complex                    |
| 3.3      | Tasks that are repetitive, data-heavy, or suitable for automation |
| 3.4      | Tasks that are most enjoyable vs boring or tedious                |
| 3.5      | Common pain points or inefficiencies                              |
| 3.6      | How enjoyment, skill level, and productivity relate               |
| 3.7      | Strategies or workarounds used for difficult tasks                |


Subtopics 1.x (background/demographics) are used separately to extract job title, sector, and industry via LLM, but are excluded from task generation.

**LLM input (per participant, GPT-4.1, temperature=0.4):**

- Job title, sector, and industry (extracted from 1.x background memories)
- All 2.x/3.x memory notes, each labeled with its subtopic and including the verbatim interview quote that produced it

**Occupation extraction (1.x memories → GPT-4.1, temperature=0):**

A separate LLM call on 1.x background memories extracts three fields:

- `occupation` — standardised job title as a short noun phrase (e.g. `"Operations Manager (Healthcare)"`)
- `sector` — broad economic sector (e.g. `"Healthcare"`, `"Technology"`, `"Finance"`, `"Education"`, `"Research & Academia"`)
- `industry` — specific industry within that sector (e.g. `"orthopedic physician practice"`, `"electric vehicle / automotive"`)

These fields are written directly to `study_tasks.json` at extraction time — before O*NET mapping.

**LLM output:** 3–6 structured task objects per participant. Each task has:

- `task_statement` — a single sentence combining action, object, and purpose
- `action`, `object`, `purpose`, `tools`, `frequency`, `method`, `judgment`, `work_context`, and other schema fields
- `sources` — a dict mapping each populated field name to the verbatim interview quote that supports it

The LLM is instructed to populate fields only from information explicitly stated or directly implied in the memories — no inference or fabrication.

---

## Step 2 — Occupation Mapping

**Script:** `2_map_onet_occupations.py`  
**Reads:** `analysis/onet/data/study_tasks.json`, `Occupation Data O*NET.xlsx`  
**Writes:** `analysis/onet/data/study_tasks.json` (adds `onet_code`, `onet_title`, `onet_industry`, `onet_match_notes`)

Each participant is mapped to an O*NET-SOC code using GPT-5.4 (temperature=0, batches of 6).

**LLM input (per batch):**

- Up to 6 participants, each with: `user_id`, job title, occupation category, and their full list of extracted task statements
- The complete O*NET occupation list: every SOC code, title, and first sentence of the occupation description

**LLM output (per participant):**

- `industry` — the sector identified in Step 1 (e.g. "electric vehicle / automotive")
- `onet_code` — matched SOC code (e.g. "15-2051.00")
- `onet_title` — matched O*NET title
- `match_notes` — one sentence explaining the industry + role match

**Two-step reasoning the LLM applies:**

**Step 1 — Identify industry/sector**  
Job title is the primary signal. Task statements are used only for confirmation — an incidental or secondary task (e.g. a sales rep who also translates) does not override the industry inferred from the title.

**Step 2 — Identify role within that industry**  
The most specific O*NET code available for that industry + role combination is selected. If a worker performs a secondary function alongside their primary role, the primary role wins.

Special cases:

- PhD students doing technical research → research scientist code for their field
- Students who also teach → primary research role, not teaching
- Entrepreneurs → the functional role performed day-to-day (e.g. buyer, manager)
- Interns → the occupation they are training for

---

## Step 3 — Canonical Task Aggregation

**Script:** `3_aggregate_canonical_tasks.py`  
**Reads:** `analysis/onet/data/study_tasks.json`  
**Writes:** `analysis/onet/data/canonical_tasks.json`, `analysis/onet/data/pipeline_trace.json`

Four stages run sequentially per occupation category (grouped by O*NET title). Each stage refines the output of the previous one. A full pipeline trace is saved to `pipeline_trace.json` for inspection via `visualize_pipeline_trace.py`.

---

### Stage 3a — Validity Screening (GPT-5.4, temperature=0.1)

Screens each raw task statement against five concrete checks. Statements that fail any check are removed before grouping.

**Input (per category):**

- All raw task statements for the category, each with its structured fields:
  ```
  [Occupation] Task statement | action: ... | object: ... | purpose: ... | tools: ... | frequency: ...
  ```

**Output:** One object per input statement:

- `statement` — the original text
- `occupation` — the participant's occupation
- `checks` — five booleans:
  - `specific_action` — concrete observable verb, not vague
  - `concrete_object` — specific artifact/system/document, not generic
  - `bounded_activity` — has a start and end, not a standing trait
  - `single_task` — one coherent unit, not 3+ bundled activities
  - `common_to_occupation` — typical task for this occupation, not idiosyncratic
- `status` — `pass`, `rewritten`, or `rejected`
- `rewritten` — revised statement (when status is `rewritten`, else empty string)
- `reason` — explanation of what was fixed or why it was rejected

Statements that fail one or more checks are not immediately dropped — the LLM first attempts to rewrite them into a valid form. Only statements that cannot be salvaged (personality traits, role descriptions, or too vague to extract any concrete activity) are rejected.

**Decision criteria:**


| Check                    | Pass                                                | Fail                                                           |
| ------------------------ | --------------------------------------------------- | -------------------------------------------------------------- |
| Specific action?         | "Debug", "Review", "Write", "Schedule"              | "Support", "Manage", "Handle", "Facilitate", "Ensure"          |
| Concrete object?         | "billing reports", "unit tests", "client contracts" | "tasks", "things", "work", "multiple priorities"               |
| Bounded activity?        | Something that happens and finishes                 | "Be a good communicator", "Understand the business"            |
| Single task?             | One meaningful work unit                            | "Handle all aspects of onboarding and relationship management" |
| Common to occupation?    | Recognizable, typical activity for the role         | Idiosyncratic to one person (e.g. SWE who plans office events) |


---

### Stage 3b — Grouping and Merging (GPT-5.4, temperature=0.2)

Groups valid statements from multiple participants into a deduplicated canonical list.

**Input:**

- Only the statements that passed Stage 3a
- List of participant occupations in the category

**Output:** A JSON array — one object per distinct activity identified:

- `canonical_statement` — a draft single-sentence description
- `abbreviated_phrase` — 2–5 word label (e.g. "billing review", "code debugging")
- `source_count` — how many participants contributed
- `source_statements` — the original raw statements that were merged
- `merge_type` — how this task was produced (see below)
- `notes` — analyst notes explaining what was combined and why

**Merge types:**

| Value            | Meaning                                                                        |
| ---------------- | ------------------------------------------------------------------------------ |
| `identical`      | Merged statements describing the same task in different words                  |
| `partial_overlap`| Merged statements that overlap but each added distinct details (tool, context) |
| `subsumes`       | A broader statement absorbed a narrower sub-step                               |
| `single`         | No merge — statement stood on its own                                          |

**Decision criteria:**

| Merge when                                                           | Keep separate when                              |
| -------------------------------------------------------------------- | ----------------------------------------------- |
| Same core activity described in different words                      | Could be assigned to a different worker         |
| Overlapping activities where one adds a distinct tool or context     | Require meaningfully different tools or skills  |
| One statement is a sub-step or component of a broader task           | Differ substantially in frequency or importance |
| Multiple participants describe the same core task in different words |                                                 |


---

### Stage 3c — Rewrite (GPT-4.1, temperature=0.1)

**Script:** `3a_rewrite_task_statements.py` (called from `3_aggregate_canonical_tasks.py`)

Rewrites each canonical statement to enforce a consistent three-part structure. Processed in chunks of 4 to stay within token limits.

**Input (per chunk):**

- Up to 4 canonical statements from Stage 3b

**Output (per statement):**

- `rewritten` — the statement reformatted as: `<Action> <object> [using <tools>] to <immediate outcome>.`
- `action` — extracted action verb
- `object` — extracted object
- `outcome` — extracted outcome

These extracted components are stored as `statement_parts` for validation in Stage 3c.

**Decision criteria — component quality:**


| Component | Good                                                                                               | Bad                                                                                              |
| --------- | -------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Action    | Specific observable verb: "Debug", "Draft", "Schedule"                                             | Vague: "use", "support", "facilitate", "ensure", "manage", "handle", "perform", "participate in" |
| Object    | Concrete artifact, system, person, or material: "monthly billing reports", "backend API endpoints" | Generic placeholder: "AI tools", "multiple tasks", "work"                                        |
| Outcome   | Immediate result: "to flag billing errors", "to catch regressions before deployment"               | Downstream goal: "to improve performance", "to meet requirements", "to optimize"                 |


Tool references are included only when a specific named tool appears in the source statements. Specifics not present in the sources are never fabricated.

---

### Stage 3d — Deduplication (GPT-5.4, temperature=0.1)

**Script:** `3b_dedup_canonical_tasks.py` (called from `3_aggregate_canonical_tasks.py`, or run standalone)

Catches canonical tasks that survived grouping as separate items but describe the same core activity from different angles (e.g. "Attend team stand-ups to provide progress updates" and "Discuss results with the team or management").

**Input:** The full list of canonical tasks for one O*NET title category.

**Output (per group):**

- Single task (no merge): `{"indices": [i]}`
- Merged group: `{"indices": [i, j, ...], "merged_statement": "...", "merged_parts": {"action": "...", "object": "...", "outcome": "..."}, "merge_reason": "..."}`

Merged tasks produce a single canonical entry with:
- Combined `source_statements` from all merged tasks
- Summed `source_count`
- A new `canonical_statement` covering all merged sources
- A `notes` field recording the merge reason

**Decision criteria — merge when:**

- Same action verb AND same object domain (e.g. both are about communicating work status)
- One is a specific instance of the other (attending stand-ups IS sharing progress updates)
- Separating them would not be meaningful to a job analyst or worker

**Decision criteria — keep separate when:**

- Different tools or methods that create genuinely distinct work
- Substantially different objects (e.g. "review code" vs "review documentation")
- Different audiences or substantially different outcomes

**Standalone usage:**

```bash
python analysis/onet/3b_dedup_canonical_tasks.py
```

Reads `canonical_tasks.json`, deduplicates all categories in parallel (4 workers), and overwrites the file in place.

---

## Step 4 — O*NET Comparison (GPT-5.4, temperature=0.1)

**Script:** `4_compare_onet_tasks.py`  
**Reads:** `analysis/onet/data/canonical_tasks.json`, `analysis/onet/data/study_tasks.json`, `Task Statements O*NET 30.2.xlsx`  
**Writes:** `analysis/onet/data/onet_comparison.json`

Compares each canonical task against the O*NET task database to determine how much of what workers reported is already captured by existing occupational frameworks.

**Input (per category, one LLM call):**

- All canonical tasks for the category (from Step 3), numbered
- All **Core** O*NET task statements for every SOC code mapped to participants in that category, each prefixed with its code and title:
  ```
  [15-1252.00 Software Developers] Modify existing software to correct errors, adapt it to new hardware, or upgrade interfaces and improve performance.
  ```
  Supplemental O*NET tasks are excluded to keep context manageable.

**Output (per category):**

- `canonical_coverage` — one entry per canonical task:
  - `canonical_task` — the statement text
  - `coverage` — `exact`, `partial`, or `novel`
  - `novelty_type` — `ai_augmented`, `ai_new`, `new_non_ai`, or `null`
  - `best_onet_match` — the matched O*NET task text, or `null` if no genuine overlap
  - `notes` — brief explanation of the classification
- `onet_absent` — O*NET tasks with no match in the canonical list:
  - `onet_task` — the O*NET task text
  - `onet_code` — the SOC code it belongs to
  - `status` — `not_reported` or `out_of_scope`
  - `notes` — explanation

### Decision criteria — Coverage Classification

For each canonical task, the LLM compares it against the full set of O*NET tasks and classifies the best match:


| Label     | Criteria                                                                                                                                                                                                                                                                                                                             | Example                                                                                                                                                                                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `exact`   | O*NET has a task with the **same core activity, object, and intent**                                                                                                                                                                                                                                                                 | Canonical: "Write and modify code for software features" ↔ O*NET: "Modify existing software to correct errors"                                                                                                                                                            |
| `partial` | The canonical task is a **recognizable instance** of a broader O*NET task, or the O*NET task is one component of what the canonical task describes. A worker doing one would immediately recognize the other as part of the same work. Being in the same domain is not enough — the tasks must be in a subset/superset relationship. | Canonical: "Debug failing CI pipeline tests using pytest logs to identify regressions" ↔ O*NET: "Modify existing software to correct errors, adapt it to new hardware, or upgrade interfaces" (debugging test failures is one specific way of correcting software errors) |
| `novel`   | **No O*NET task meaningfully covers this activity.**                                                                                                                                                                                                                                                                                 | Canonical: "Verify AI-generated answers for accuracy" ↔ no O*NET equivalent                                                                                                                                                                                               |
| `unsure`  | Cannot confidently distinguish between partial and novel — the match is ambiguous.                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                                            |


`best_onet_match` is set to the O*NET task text only when there is genuine overlap. If no task meaningfully matches, it is set to `null` — not forced to the closest distant match.

### Decision criteria — Novelty Classification

For `partial` and `novel` tasks, the LLM classifies **why** the task diverges from O*NET:


| Label          | Criteria                                                                                                                    | Example                                                      |
| -------------- | --------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `ai_augmented` | A **traditional task** now performed **with AI assistance** — the core activity exists in O*NET but the AI tooling does not | "Summarize papers using AI tools"                            |
| `ai_new`       | A **genuinely new task created by AI** that did not exist before                                                            | "Verify AI-generated answers for accuracy"                   |
| `new_non_ai`   | A task **absent from O*NET but unrelated to AI** — a gap in the framework, not an AI phenomenon                             | "Implement industry standard workflows in software projects" |


### Decision criteria — Absent Task Classification

For each O*NET task that has no match in the canonical list, the LLM classifies why it is missing:


| Label          | Criteria                                                                                                                                          |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `not_reported` | Workers likely perform this task but did not mention it in the study (e.g. routine tasks not worth noting, or outside the interview's focus)      |
| `out_of_scope` | The task falls **outside what study participants actually do** — it belongs to the O*NET occupation code but not to these specific workers' roles |


Absent tasks are scoped to the specific O*NET code they came from — each occupation block in the viewer only shows absent tasks from its own code, not from other codes in the same category.

---

## Step 5 — Visualization

**Script:** `5_visualize_onet_comparison.py`  
**Reads:** `analysis/onet/data/onet_comparison.json`, `analysis/onet/data/canonical_tasks.json`, `analysis/onet/data/study_tasks.json`  
**Writes:** `analysis/onet/data/onet_comparison_viewer.html`

Interactive HTML viewer organized by O*NET occupation. Each block shows:

- Participant job titles mapped to that code (`n=X` per occupation)
- Canonical tasks with coverage and novelty badges
- Best O*NET match text
- Participant count per task
- Verbatim source quotes from interviews (click to expand)
- O*NET tasks absent from the study, scoped to that occupation's code

Filters: coverage type (exact/partial/novel), novelty type (ai_augmented/ai_new/new_non_ai).

---

## Data Files


| File                                             | Description                                              |
| ------------------------------------------------ | -------------------------------------------------------- |
| `Occupation Data O*NET.xlsx`                     | O*NET occupation titles and descriptions (v30.2)         |
| `Task Statements O*NET 30.2.xlsx`                | O*NET task statements by SOC code (v30.2)                |
| `analysis/onet/data/study_tasks.json`            | Structured task objects per participant with occupation, sector, industry, and O*NET codes |
| `analysis/onet/data/canonical_tasks.json`        | Deduplicated canonical task list per occupation category |
| `analysis/onet/data/onet_comparison.json`        | Coverage classification results                          |
| `analysis/onet/data/onet_comparison_viewer.html` | Interactive HTML viewer                                  |


