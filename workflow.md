# SparkMe Workflow

## Original SparkMe — Single Deep Interview

```
╔══════════════════════════════════════════════════════════════════════════╗
║  STARTUP  (once per session)                                             ║
║                                                                          ║
║  topics.json ──► InterviewTopicManager    user_portrait.json             ║
║                  10 topics                     │                         ║
║                  48 subtopics                  │                         ║
║                  all is_covered=False          │                         ║
║                        └──────────────────► SessionAgenda                ║
║                                            (the shared scoreboard)       ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  TURN LOOP                                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌─────────────────────────────────────────────────────────┐            ║
║   │  INTERVIEWER  decides what to ask                       │            ║
║   │                                                         │            ║
║   │  reads from SessionAgenda:                              │            ║
║   │    • user_portrait                                      │            ║
║   │    • last_meeting_summary                               │            ║
║   │    • topic tree + coverage state + notes                │            ║
║   │    • strategic suggestions  ◄── from StrategicPlanner   │            ║
║   │                                                         │            ║
║   │  optionally calls Recall tool:                          │            ║
║   │    query ──► FAISS search ──► relevant past memories    │            ║
║   │                                                         │            ║
║   │  scores each subtopic 1-3 (STAR coverage)               │            ║
║   │  picks highest-priority uncovered subtopic              │            ║
║   │  calls respond_to_user(subtopic_id, question)           │            ║
║   └──────────────────────────┬──────────────────────────────┘            ║
║                              │ question + subtopic_id in metadata        ║
║                              ▼                                           ║
║   ┌──────────────────────────────────────────────────────────┐           ║
║   │  MESSAGE BUS  (asyncio pub/sub)                           │          ║
║   │                                                           │          ║
║   │  Interviewer publishes ──► [SessionScribe, User]          │          ║
║   │  User publishes        ──► [Interviewer,                  │          ║
║   │                             SessionScribe,                │          ║
║   │                             StrategicPlanner]             │          ║
║   └───────┬───────────────────────┬──────────────────────────┘           ║
║           │                       │                                      ║
║           ▼                       ▼                                      ║
║   ┌───────────────┐     ┌─────────────────┐                              ║
║   │     USER      │     │  SESSION SCRIBE │ (async background)           ║
║   │               │     │                 │                              ║
║   │ reads question│     │ receives the    │                              ║
║   │ types answer  │     │ Q&A pair        │                              ║
║   └───────┬───────┘     │                 │                              ║
║           │             │ ┌─────────────┐ │                              ║
║           │    answer   │ │ extract     │ │  LLM call:                   ║
║           │             │ │ memories    │ │  "what facts did             ║
║           │             │ └──────┬──────┘ │   the user share?"           ║
║           │             │        │        │                              ║
║           │             │        ▼        │                              ║
║           │             │  MEMORY BANK   ◄─── add_memory()               ║
║           │             │  (FAISS index) │  title + text + embedding     ║
║           │             │                │  + source Q&A                 ║
║           │             │ ┌─────────────┐│                               ║
║           │             │ │ mark        ││  LLM call:                    ║
║           │             │ │ coverage    ││  "does this cover             ║
║           │             │ └──────┬──────┘│   the STAR elements?"         ║
║           │             │        │        │                              ║
║           │             │        ▼        │                              ║
║           │             │  SessionAgenda ◄─── update subtopic            ║
║           │             │  is_covered=True    notes + coverage           ║
║           │             └─────────────────┘                              ║
║           │                                                              ║
║           │  answer published ──► [Interviewer, SessionScribe,           ║
║           │                        StrategicPlanner]                     ║
║           │                                      │                       ║
║           │                                      ▼ (every 3 turns)       ║
║           │                          ┌───────────────────────┐           ║
║           │                          │   STRATEGIC PLANNER   │           ║
║           │                          │   (async background)  │           ║
║           │                          │                        │          ║
║           │                          │  1. brainstorm new     │          ║
║           │                          │     subtopics          │          ║
║           │                          │                        │          ║
║           │                          │  2. detect emergent    │          ║
║           │                          │     insights           │          ║
║           │                          │     (novelty 1-5)      │          ║
║           │                          │                        │          ║
║           │                          │  3. predict N rollouts │          ║
║           │                          │     score each:        │          ║
║           │                          │     U = αC - βCost     │          ║
║           │                          │         + γEmergence   │          ║
║           │                          │                        │          ║
║           │                          │  4. generate questions │          ║
║           │                          │     for best rollout   │          ║
║           │                          │         │              │          ║
║           │                          └─────────┼─────────────┘           ║
║           │                                    │ suggestions             ║
║           │                                    ▼                         ║
║           │                          strategic_state                     ║
║           │                          .strategic_question_suggestions     ║
║           │                          (read by Interviewer next turn)     ║
║           │                                                              ║
║           └──────────────────────────────────────────────────────────►   ║
║                                               back to top of turn loop   ║
╠══════════════════════════════════════════════════════════════════════════╣
║  session ends when:  all_core_topics_completed()  OR  max_turns hit      ║
║                      OR timeout (default 10 min inactivity)              ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  SHUTDOWN  (once per session)                                            ║
║                                                                          ║
║  ReportOrchestrator                                                      ║
║    └── UpdateUserPortrait  ──► fills in portrait fields from memories    ║
║                                                                          ║
║  memory_bank.save_to_file()                                              ║
║    ├── memory_bank_content.json     (all Memory objects)                 ║
║    └── memory_bank_embeddings.json  (all vectors, FAISS rebuilt on load) ║
║                                                                          ║
║  session_agenda saved                                                    ║
║    └── coverage state, notes, portrait  ──► loaded next session          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Three things to hold in your head:**

1. **SessionAgenda is the shared scoreboard** — all three agents read from it and write to it. The Interviewer uses it to pick questions, the Scribe updates it after each answer, the Planner uses it to score rollouts.
2. **The message bus is fire-and-forget** — when the user answers, `asyncio.create_task()` is called for each subscriber. They all run concurrently. The Interviewer doesn't wait for the Scribe to finish before forming the next question.
3. **The Scribe writes to memory, the Interviewer reads from it** — they never talk directly. The memory bank is the indirect channel between past sessions and the current one.

---

## New SparkMe — Weekly Work Tracking

Same architecture, with changes marked `◄── NEW`:

### Data model

```
user_portrait.json (stable, LLM-updated)     WeeklySnapshot (volatile, overwritten each week)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│ role:                                │     │ week_number: 14                      │
│   functional_role: "..."             │     │ tasks:                               │
│   collaboration_structure:           │     │   - description: "client deck prep"  │
│     key_collaborators: []            │     │     time_share: 0.30                 │
│     delegates_to: []                 │     │     ai_involved: true                │
│     receives_from: []                │     │     ai_tool: "ChatGPT"               │
│   primary_tools: []                  │     │     ai_purpose: "first drafts"       │
│   motivations_and_goals: []          │     │   - description: "data analysis"     │
│                                      │     │     time_share: 0.25                 │
│ ai_relationship:                     │     │     ai_involved: false               │
│   trust_by_task_type:                │     │ collaborators_this_week: []           │
│     - task_type: "drafting"          │     │ notable_events: "..."                │
│       trust_level: "high"            │     └──────────────────────────────────────┘
│       reason: "..."                  │       saved to: snapshot_week_N.json
│   preferred_interaction_style: "..."  │       loaded next week for Scribe comparison
│   known_pain_points: []              │
│   known_bright_spots: []             │
└──────────────────────────────────────┘
  updated by: SessionCoordinator (LLM)
  evolves slowly across sessions
```

### Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║  STARTUP                                                                ║
║                                                                         ║
║  ▶ topics_intake.json  OR  topics_weekly.json   ◄── NEW: two files,     ║
║    (selected by session_type="intake"|"weekly")      selected at init   ║
║                                                                         ║
║  ▶ user_portrait.json  ◄── NEW: stable profile only                     ║
║    { role: {functional_role, collaboration_structure,                    ║
║             primary_tools, motivations_and_goals},                      ║
║      ai_relationship: {trust_by_task_type,                              ║
║             preferred_interaction_style,                                 ║
║             known_pain_points, known_bright_spots} }                    ║
║                                                                         ║
║  [weekly only] ────────────────────────────────────────────────────┐    ║
║  │  SnapshotManager.load_latest_snapshot()                         │    ║
║  │    └── snapshot_week_N-1.json                                   │    ║
║  │          │                                                      │    ║
║  │          ▼                                                      │    ║
║  │  session_agenda.last_week_snapshot  (raw structured data)       │    ║
║  │    Scribe compares user responses against this each turn        │    ║
║  └─────────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  TURN LOOP                                                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                         ║
║   ┌───────────────────────────────────────────────────────────┐         ║
║   │  INTERVIEWER                                               │        ║
║   │                                                            │        ║
║   │  ▶ "weekly_introduction" / "weekly_normal" prompt ◄── NEW  │        ║
║   │     vs. "introduction" / "normal"                          │        ║
║   │                                                            │        ║
║   │  ▶ covers subtopics like normal — including any            │        ║
║   │    snapshot-driven subtopics added by the Scribe           │        ║
║   │                                                            │        ║
║   │  ▶ aims to wrap ~10 min, no hard turn cap                  │        ║
║   │                                                            │        ║
║   │  everything else unchanged:                                │        ║
║   │    reads SessionAgenda, calls Recall, STAR scoring         │        ║
║   └──────────────────────────┬────────────────────────────────┘         ║
║                              │ question                                 ║
║                              ▼                                          ║
║                   [MESSAGE BUS — unchanged]                             ║
║                              │                                          ║
║           ┌──────────────────┴──────────────────┐                       ║
║           │                                     │                       ║
║   ┌───────────────┐                  ┌─────────────────────────────┐    ║
║   │     USER      │                  │  SESSION SCRIBE             │    ║
║   │               │                  │                             │    ║
║   └───────┬───────┘                  │  unchanged:                 │    ║
║           │                          │    extract memories          │    ║
║           │                          │    mark coverage             │    ║
║           │                          │    find insights             │    ║
║           │  answer                  │                             │    ║
║           │                          │  NEW (weekly only):         │    ║
║           │                          │  ┌────────────────────────┐ │    ║
║           │                          │  │ _compare_against_      │ │    ║
║           │                          │  │   snapshot()           │ │    ║
║           │                          │  │                        │ │    ║
║           │                          │  │ Q&A vs last_week_      │ │    ║
║           │                          │  │   snapshot → LLM       │ │    ║
║           │                          │  │                        │ │    ║
║           │                          │  │ uses recall tool for   │ │    ║
║           │                          │  │   memory grounding     │ │    ║
║           │                          │  │                        │ │    ║
║           │                          │  │ adds emergent sub-     │ │    ║
║           │                          │  │   topics for:          │ │    ║
║           │                          │  │  • inconsistencies     │ │    ║
║           │                          │  │  • unmentioned items   │ │    ║
║           │                          │  │                        │ │    ║
║           │                          │  │ (runs in parallel      │ │    ║
║           │                          │  │  with coverage update) │ │    ║
║           │                          │  └────────────────────────┘ │    ║
║           │                          │                             │    ║
║           │                          │  Interviewer sees new sub-  │    ║
║           │                          │  topics next turn, covers   │    ║
║           │                          │  them like any other        │    ║
║           │                          └─────────────────────────────┘    ║
║           │                                                             ║
║           └──────────────────────────────────────────────────────────►  ║
║                               [STRATEGIC PLANNER — unchanged]           ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  SHUTDOWN                                                               ║
║                                                                         ║
║  1. SessionCoordinator / UpdateUserPortrait  ◄── updated                ║
║     ▶ LLM reads session memories                                        ║
║     ▶ updates stable portrait fields (role, ai_relationship)            ║
║       merges dicts + deduplicates lists  ◄── NEW                        ║
║       (was: always overwrote with cleaned string)                       ║
║     ▶ does NOT touch snapshot — that's separate                         ║
║                                                                         ║
║  2. memory_bank.save_to_file()  ──── unchanged                          ║
║     session_agenda saved  ──── unchanged                                ║
║                                                                         ║
║  3. [weekly only] ─────────────────────────────────────────────────┐    ║
║     │  _generate_and_save_weekly_snapshot()             ◄── NEW    │    ║
║     │                                                              │    ║
║     │  session memories ──► LLM extraction                         │    ║
║     │                                                              │    ║
║     │  WeeklySnapshot {                                            │    ║
║     │    tasks: [{description, time_share,                         │    ║
║     │             ai_involved, ai_tool, ai_purpose}],              │    ║
║     │    collaborators_this_week: [],                               │    ║
║     │    notable_events: "..."                                     │    ║
║     │  }                                                           │    ║
║     │         │                                                    │    ║
║     │         ▼                                                    │    ║
║     │  logs/{user_id}/weekly_snapshots/snapshot_week_N.json        │    ║
║     │         │                                                    │    ║
║     │         └──► loaded at STARTUP next week for Scribe          │    ║
║     └──────────────────────────────────────────────────────────────┘    ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**What changed, by phase:**

| Phase            | What changed                                                                             |
| ---------------- | ---------------------------------------------------------------------------------------- |
| Startup          | Two topic files, new portrait schema (stable only), raw snapshot loaded for Scribe       |
| Interviewer      | Weekly prompts, covers snapshot-driven subtopics alongside regular ones                   |
| SessionScribe    | NEW `_compare_against_snapshot()`: adds emergent subtopics for inconsistencies/gaps       |
| StrategicPlanner | Completely unchanged                                                                      |
| Memory bank      | Completely unchanged                                                                      |
| Shutdown         | Portrait updates stable fields only; snapshot generated separately as its own file        |

**Key design decisions:**

1. **Portrait and snapshot are separate concerns.** The portrait holds stable, slowly-evolving profile data (role, AI trust, tools). The snapshot holds volatile weekly data (tasks, time shares, collaborators this week). They never overlap — the portrait is LLM-updated, the snapshot is LLM-extracted and overwritten each week.

2. **Snapshot-driven items are subtopics, not a separate data path.** When the Scribe detects an inconsistency or unmentioned item vs. last week's snapshot, it adds an emergent subtopic. The Interviewer covers it like any other uncovered subtopic. Coverage tracking handles "addressed" status.

3. **Scribe uses recall for memory grounding.** Before adding a subtopic, the Scribe can search prior memories to include the user's original words, making the subtopic description specific (e.g., "user said 'the quarterly deck was eating my Fridays' but hasn't mentioned it").

4. **Snapshot comparison runs in parallel** with subtopic coverage updates (separate locks, `asyncio.gather`). One-turn delay — same pattern as memory extraction.
