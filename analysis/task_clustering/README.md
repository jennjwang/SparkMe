# Online Task Clustering

General-purpose online clustering pipeline that groups task observations into a hierarchical taxonomy. Domain-agnostic — not tied to O\*NET or any specific schema.

Four algorithm layers, applied in sequence on each ingested item:

1. **Outlier buffer** — new items that don't match any cluster are held in a buffer. A real cluster is only created when a second matching item arrives. Items that never accumulate support decay and are discarded.
2. **Leader algorithm** — assigns each item to an existing cluster (or promotes a buffered item to a new one). Uses an LLM as the similarity function, not embeddings.
3. **DenStream fading** — cluster weights decay exponentially over interviews. Stale clusters are pruned.
4. **Incremental Divisive Clustering** — after each assignment, the receiving cluster is checked: if it has crossed the size threshold, it is split into 2–4 sub-types immediately. Produces a two-level tree where leaves = specific subtask types, roots = broad categories.

---

## Algorithm

The pipeline applies three steps on every ingested item *t*. Steps execute in sequence; state is shared.

---

### Step 1 — Outlier Buffer (leader.py)

When the LLM judges that *t* doesn't fit any existing cluster, it is held in the outlier buffer rather than immediately creating a new cluster. A cluster is only promoted when a second item arrives that matches a buffered item.

```
CHECK_BUFFER(t, state):
  if state.outlier_buffer = ∅:
    state.outlier_buffer.append(OutlierItem(task=t, weight=1.0))
    return null                               # not yet assigned

  choice ← LLM("which buffer item does t match? or 'none'",
               leaders=[item.task.text for item in state.outlier_buffer])

  if choice = "none":
    state.outlier_buffer.append(OutlierItem(task=t, weight=1.0))
    return null

  matched ← state.outlier_buffer.pop(choice)
  label   ← LLM_GENERALIZE([matched.task.text, t.text])
  cluster ← Cluster(leader=label,
                    members=[matched.task.id, t.id],
                    weight=matched.weight + 1.0)
  state.clusters.add(cluster)
  return cluster.id
```

Items in the buffer decay with the same DenStream fading applied to clusters (Step 3). Items whose weight falls below ε are discarded without ever forming a cluster.

---

### Step 2 — Leader Assignment (leader.py)

Route *t* through the two-level cluster tree. Each routing decision is one LLM call. The buffer check above is only reached when no existing cluster matches.

```
ASSIGN(t, state):
  roots ← [c ∈ state.clusters | c.parent = null]

  if roots = ∅:
    return CHECK_BUFFER(t, state)             # no clusters yet

  # Level-0: pick a root or fall through to buffer
  choice ← LLM("which root does t fit? or 'new'", leaders=[c.label for c in roots])

  if choice = "new":
    return CHECK_BUFFER(t, state)

  root ← roots[choice]
  children ← root.children

  if children = ∅:
    MERGE(t, root)                            # root is a leaf — assign directly
    return root.id

  # Level-1: pick a child or stay at root
  choice2 ← LLM("which child does t fit? or 'none'", leaders=[c.label for c in children])

  if choice2 = "none":
    MERGE(t, root)
  else:
    MERGE(t, children[choice2])
```

```
MERGE(t, cluster):
  cluster.members.append(t)
  cluster.weight += 1
  if not cluster.anchored and |cluster.members| mod LEADER_REFRESH_EVERY = 0:
    cluster.leader ← LLM_GENERALIZE(sample(cluster.members, 10))
```

**LLM calls per item:** 1–3 — level-0 routing (1) + optional level-1 routing (1) + optional buffer check (1).
**No forced merging:** the LLM is instructed to output `"new"` / `"none"` when nothing fits.

---

### Step 3 — DenStream Fading (denstream.py)

Applies every `fade_interval_interviews` items. Exponential decay per interview count.

```
FADE_AND_PRUNE(state):
  delta ← state.total_processed - state.last_fade_count
  factor ← 2^(-λ · delta)

  for each cluster c:
    c.weight ← c.weight · factor
  for each item b in state.outlier_buffer:
    b.weight ← b.weight · factor              # buffer items decay too

  state.last_fade_count ← state.total_processed

  for each leaf cluster c where c.weight < ε and not c.anchored:
    DELETE(c)
  state.outlier_buffer ← [b for b in outlier_buffer where b.weight ≥ ε]
```

| Symbol | Parameter | Meaning |
|---|---|---|
| λ | `lambda` | Decay rate; λ=0.1 → weight halves every ~10 interviews |
| ε | `eps` | Prune threshold; clusters below this weight are removed |
| — | `fade_interval_interviews` | How often (in items) fading is applied |

Anchored (taxonomy) clusters are exempt from pruning. Trace mode sets `fade_interval_interviews = ∞` (no fading).

---

### Step 4 — Incremental Divisive Splitting (divisive.py)

After each assignment, only the cluster that just received *t* is checked. No full scan.

```
TRY_SPLIT(cluster, state):
  if cluster.level ≠ 0: return               # only roots split
  if |cluster.members| < max_split_size: return

  groups ← LLM_SPLIT(cluster.leader, cluster.members)
  # LLM proposes 2–4 labeled sub-groups with member index lists

  if |groups| ≤ 1: return                    # LLM says homogeneous

  for each group g:
    child ← Cluster(leader=g.label, members=g.members, level=1)
    cluster.children.append(child)

  cluster.members ← []                       # direct members redistributed to children
```

After a split, the root becomes an internal node. Future items routed to this root go to Level-1 (Step 1, `children ≠ ∅` branch). Old children from prior splits are preserved.

| Parameter | Config key | Default |
|---|---|---|
| Split trigger | `max_split_size` | 5 |
| Max depth | `split_depth_limit` | 1 |

---

### Full per-item loop

```
INGEST(t, state):
  state.total_processed += 1

  cluster_id ← ASSIGN(t, state)              # Steps 1–2: buffer check + leader routing
                                              # cluster_id may be null if t went to buffer

  if state.total_processed mod fade_interval_interviews = 0:
    FADE_AND_PRUNE(state)                     # Step 3: DenStream fading (clusters + buffer)

  if cluster_id ≠ null:
    TRY_SPLIT(state.clusters[cluster_id], state)  # Step 4: Divisive splitting
```

---

## Architecture

```
run_screening.py         Pre-screen raw study tasks using ONET validity checks
cli.py                   CLI entry point — reads config, runs pipeline, saves output
trace.py                 Trace mode — ingests item-by-item, saves snapshot per step
  │
  └─► pipeline.py        OnlineClusteringPipeline — orchestrator
        │
        ├─► leader.py    Outlier buffer + Leader algorithm
        │     └─► _check_buffer()           — hold item or promote buffered pair
        │     └─► llm_ops.assign_task()     — route item to cluster or "new"
        │     └─► llm_ops.update_leader()   — generalize cluster label
        │
        ├─► denstream.py DenStream fading + pruning (clusters and buffer)
        │
        └─► divisive.py  Incremental divisive splitting (per assignment)
              └─► llm_ops.split_cluster()

models.py                TaskItem, Cluster, OutlierItem, ClusterState dataclasses
llm_ops.py               LLM prompt operations (assign, update_leader, split, screen)
visualize.py             Generate interactive HTML visualizer from state or trace JSON
```

All LLM calls go through `dataset_gen/llm_client.py` (`LLMClient`), which wraps the OpenAI API with retry logic.

---

## Typical workflow

```bash
# 1. Pre-screen raw study tasks using the ONET pipeline's validity checks
python analysis/task_clustering/run_screening.py
# → output/screened_study_tasks.json  (pass + rewritten tasks, per interview)

# 2a. Run the full pipeline
python analysis/task_clustering/cli.py --config config_cirs.json

# 2b. OR run in trace mode (item-by-item, saves viz automatically)
python analysis/task_clustering/trace.py --config config_cirs.json

# 3. View the interactive visualizer
open analysis/task_clustering/output/viz.html
```

---

## Data flow

```
                      ┌──────────────┐
                      │  Taxonomy    │  (optional warmup — e.g. O*NET task statements)
                      │  JSON file   │
                      └──────┬───────┘
                             │ load_taxonomy()
                             ▼
┌──────────┐      ┌──────────────────────┐      ┌───────────────────┐
│  Input   │─────►│  OnlineClustering    │─────►│  State JSON       │
│  JSON    │      │  Pipeline            │      │  (resumable)      │
└──────────┘      │                      │      └───────────────────┘
                  │  for each item:      │
                  │   1. Leader assign   │      ┌───────────────────┐
                  │   2. DenStream fade  │─────►│  Hierarchy JSON   │
                  │   3. Divisive split  │      │  (nested tree)    │
                  └──────────────────────┘      └───────────────────┘
```

---

## Inputs

### 1. Task data (required)

Any JSON file containing a list of records. Two layouts are supported:

**Flat** — top-level list of task records:
```json
[
  {"task_statement": "Review pull requests on GitHub", "occupation": "SWE"},
  {"task_statement": "Write unit tests using pytest",  "occupation": "SWE"}
]
```

**Nested** — outer records each contain an inner list of tasks:
```json
[
  {
    "occupation": "Software Engineer",
    "tasks": [
      {"task_statement": "Review pull requests on GitHub"},
      {"task_statement": "Write unit tests using pytest"}
    ]
  }
]
```

| Config field | What it maps to | Required | Default |
|---|---|---|---|
| `text_field` | JSON key holding the task text | Yes | `text` |
| `source_field` | Participant / session id | No | `""` |
| `timestamp_field` | ISO-8601 string or Unix epoch | No | wall-clock `now()` |
| `items_field` | Inner list key for nested layouts | No | — (flat assumed) |

### 2. Pre-screening (recommended)

Run `run_screening.py` before clustering to validate and rewrite raw task statements using the ONET pipeline's validity checks:

1. **Specific action verb** — names a concrete, observable verb (not "support", "manage")
2. **Concrete object** — acts on a specific artifact, system, or person
3. **Bounded activity** — has a start and end; not a standing trait
4. **Single task** — one coherent work unit

Tasks that fail are either rewritten (salvageable) or rejected. Output includes both kept and rejected tasks with reasons.

```bash
python analysis/task_clustering/run_screening.py
# Input:  analysis/onet/data/study_tasks.json
# Output: analysis/task_clustering/output/screened_study_tasks.json
```

### 3. Criteria file (optional)

A plain text file with one similarity rule per line. `#` comments are ignored. These rules are injected into the LLM prompt that decides whether a new item matches an existing cluster.

```
# criteria.txt — guidelines for "same cluster"
Same core action and object domain
One is a specific instance of the other
A job analyst would describe them with the same canonical task statement
Tasks represent the same type of work even if phrased differently
```

If not provided, default rules are used. Set `"criteria": "path/to/criteria.txt"` in config.

### 4. Taxonomy (optional)

A JSON file of pre-defined cluster nodes loaded as anchored seeds before ingestion. Useful for warm-starting with a known domain structure (e.g. O\*NET task statements).

```json
[
  {"id": "2258", "label": "Analyze problems", "description": "Analyze problems to develop solutions involving computer hardware and software.", "parent": null},
  {"id": "2259", "label": "Apply new technology", "description": "Apply theoretical expertise and innovation to create or apply new technology.", "parent": null}
]
```

| Field | What it does | Required |
|---|---|---|
| `id` | Cluster id (string) | No — auto UUID |
| `label` | Short display name | Yes |
| `description` | Used as the cluster's leader statement | No — falls back to `label` |
| `parent` | Id of parent node | No — `null` = root |

Taxonomy behaviour:
- All taxonomy clusters have `anchored = True` — **never pruned** by DenStream.
- Taxonomy cluster labels are **never overwritten** by `update_leader` — they stay as defined.
- New items are routed to taxonomy clusters first. If none match, a new dynamic cluster is created alongside them (shown with a **NEW** badge in the visualizer).
- Build the taxonomy from O\*NET: filter `Task Statements O*NET 30.2.xlsx` by occupation and convert Core tasks to JSON.

### 5. Saved state (resume)

Pass a previously saved state JSON back via `"resume"` in config to continue ingestion where you left off.

---

## Outputs

### 1. State file

Full serialised `ClusterState`. Pass back via `"resume"` to continue.

```json
{
  "clusters": {
    "<id>": {
      "leader": "Review and synthesize research literature",
      "members": ["<item_id>", "..."],
      "weight": 4.23,
      "parent_id": null,
      "children": ["<child_id>"],
      "level": 0,
      "anchored": false
    }
  },
  "items": { "<id>": {"text": "...", "source": "...", "timestamp": "..."} },
  "lambda_": 0.1,
  "eps": 0.5,
  "max_split_size": 5,
  "split_depth_limit": 1,
  "total_processed": 30,
  "last_fade_count": 25
}
```

### 2. Hierarchy file

Human-readable nested tree (one entry per root, children nested recursively).

### 3. Trace file (`<state_stem>_trace.json`)

Generated by `trace.py`. An array of snapshots — one per interview — each containing the full cluster tree state, the task just added, where it was assigned, and the LLM's reasoning. Used by the visualizer's timeline slider.

### 4. Visualizer (`viz.html`)

Auto-generated by `trace.py`. Interactive HTML with:
- **Timeline slider** — step through interviews one at a time (← → arrow keys)
- **LLM reasoning** — shows why each task was merged or kept separate
- **Participant bars** — color-coded breakdown per cluster
- **Expandable cards** — click a cluster to see its tasks
- **NEW badge** — green border + badge on dynamically created clusters (only shown when a taxonomy is present)

```bash
# Regenerate from existing trace data (no LLM calls):
python analysis/task_clustering/visualize.py \
  --state output/cirs_clusters.json \
  --trace output/cirs_clusters_trace.json \
  --output output/viz.html
```

---

## Decision points

### 1. Outlier buffer

When the LLM says a new item doesn't fit any existing cluster, the item is added to the outlier buffer instead of immediately creating a cluster. The buffer prevents one-off or noisy tasks from polluting the taxonomy.

A real cluster is created only when a second incoming item matches a buffered item. The LLM is called once to check whether the new item matches any buffer entry (treating each buffer item's raw text as a candidate leader).

Buffer items decay with the same DenStream fading applied to clusters. Items that fall below `eps` are discarded. This means that if a truly unique task type never recurs within the active window, it is silently dropped — intentional behaviour for cross-participant clustering where rare mentions should not become canonical clusters.

| Parameter | Effect |
|---|---|
| `lambda` | Controls how quickly buffer items decay between interviews |
| `eps` | Weight threshold below which a buffer item is discarded |
| `fade_interval_interviews` | How often decay is applied |

---

### 2. Assign: hierarchical routing


The LLM routes each item **top-down** through the cluster tree, one level at a time.

- **Level 0** (roots): LLM picks a root or says `"new"` → creates a new root cluster.
- **Level 1** (children): LLM picks a child or says `"none"` → item stays at the root.

This is O(depth) LLM calls (typically 1–2) rather than one call with all leaves.

**No forced merging** — the LLM is instructed to create a new cluster if the task is genuinely different from all existing ones. Criteria rules define what counts as "same cluster".

| Parameter | Config key | Default |
|---|---|---|
| Similarity rules | `criteria` (file path) | 4 default rules |
| Temperature | hardcoded | 0.0 (deterministic) |

### 3. Leader generalization

When a **new cluster is created**, `update_leader` runs immediately on the founding task to produce an occupation-level label (e.g. "Review and synthesize research literature" rather than the raw task text).

The leader is refreshed again every `LEADER_REFRESH_EVERY = 5` new members, using up to 10 sampled member texts. **Anchored (taxonomy) clusters never have their leader updated.**

Leader format: a short 5–10 word label broad enough to cover all members, dropping person-specific details (project names, domain specifics, individual tools).

### 4. DenStream fading (interview-based)

```
weight *= 2^(-lambda_ * delta_interviews)
```

Fading is applied every `fade_interval_interviews` ingested items (not wall-clock hours). Leaf clusters with `weight < eps` are pruned. Anchored clusters are never pruned.

| Parameter | Config key | Default | Interpretation |
|---|---|---|---|
| Fading factor | `lambda` | 0.1 | Weight halves every ~10 interviews |
| Prune threshold | `eps` | 0.5 | Clusters below this weight are removed |
| Fade cadence | `fade_interval_interviews` | 10 | Apply fading every N ingested items |

**Trace mode disables fading** (`fade_interval_interviews = 999999`) so all clusters accumulate for inspection.

### 5. Incremental divisive splitting

After each assignment, only the cluster that just received the item is checked. It is split if:
- `len(members) >= max_split_size`, AND
- `level < split_depth_limit` (currently capped at 1 — only roots split)

The LLM sees all member texts and proposes 2–4 sub-groups. If it returns a single group, no split happens. After splitting:
- Parent becomes an internal node (direct members cleared, new items routed to children)
- Children inherit the redistributed members
- Old children (from prior splits) are preserved

| Parameter | Config key | Default |
|---|---|---|
| Split trigger | `max_split_size` | 5 |
| Max depth | `split_depth_limit` | 1 |
| Disable splits | `no_splits` | false |

### 6. Taxonomy warm start

Load O\*NET task statements (or any domain taxonomy) before ingestion:

```python
pipe.load_taxonomy("taxonomy_cirs.json")
```

Taxonomy clusters are anchored (not pruned, not relabeled) and compete with incoming items on equal footing. Items that don't fit any taxonomy cluster create new dynamic clusters. The visualizer highlights dynamic clusters with a green **NEW** badge when a taxonomy is present.

---

## Config file

```jsonc
{
  "input": {
    "path": "output/screened_study_tasks.json",
    "text_field": "task_statement",
    "source_field": "occupation",
    "timestamp_field": null,
    "items_field": "tasks"
  },
  "output": {
    "state": "output/clusters.json",
    "hierarchy": "output/clusters_hierarchy.json"
  },
  "taxonomy": "taxonomy_cirs.json",   // null to skip
  "criteria": "criteria.txt",          // null = defaults
  "resume": null,
  "algorithm": {
    "model": "gpt-5.4",
    "lambda": 0.1,
    "eps": 0.5,
    "max_split_size": 5,
    "split_depth_limit": 1,
    "fade_interval_interviews": 10,
    "no_splits": false
  }
}
```

---

## Python API

```python
from analysis.task_clustering import OnlineClusteringPipeline, TaskItem
from datetime import datetime

pipe = OnlineClusteringPipeline(
    lambda_=0.1,
    eps=0.5,
    max_split_size=5,
    split_depth_limit=1,
    fade_interval_interviews=10,
    criteria=["Same core work activity", "Same tools and methods"],
    model="gpt-5.4",
)

# Optional: O*NET taxonomy warmup
pipe.load_taxonomy("taxonomy_cirs.json")

# Ingest
cluster_id = pipe.ingest(TaskItem(
    id="t1", text="Review pull requests on GitHub",
    source="P01", timestamp=datetime.now(),
))

assignments = pipe.ingest_batch(items)   # {task_id: cluster_id}

# Manual controls
pipe.force_fade()           # Apply fading + pruning immediately
pipe.force_split_check()    # Trigger divisive splits on all eligible clusters

# Output
hierarchy = pipe.get_hierarchy()
leaves    = pipe.get_leaf_clusters()
stats     = pipe.summary()

# Persistence
pipe.save("state.json")
pipe = OnlineClusteringPipeline.load("state.json")
```
