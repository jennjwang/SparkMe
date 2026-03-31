# SparkMe Workflow

## Original SparkMe — Single Deep Interview

```
╔══════════════════════════════════════════════════════════════════════════╗
║  STARTUP  (once per session)                                             ║
║                                                                          ║
║  topics.json ──► InterviewTopicManager    user_portrait.json             ║
║                  (topics + subtopics,          │                         ║
║                   all is_covered=False)        │                         ║
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
║   │  reads SessionAgenda:                                   │            ║
║   │    • user_portrait, last_meeting_summary                │            ║
║   │    • topic tree + coverage state + notes                │            ║
║   │    • strategic suggestions  ◄── from StrategicPlanner   │            ║
║   │                                                         │            ║
║   │  optionally: Recall tool → FAISS search → past memories │            ║
║   │                                                         │            ║
║   │  picks highest-priority uncovered subtopic              │            ║
║   │  calls respond_to_user(subtopic_id, question)           │            ║
║   └──────────────────────────┬──────────────────────────────┘            ║
║                              │ question + subtopic_id in metadata        ║
║                              ▼                                           ║
║   ┌──────────────────────────────────────────────────────────┐           ║
║   │  MESSAGE BUS  (asyncio pub/sub)                          │           ║
║   │  Interviewer ──► [SessionScribe, User]                   │           ║
║   │  User        ──► [Interviewer, SessionScribe, Planner]   │           ║
║   └───────┬───────────────────────┬──────────────────────────┘           ║
║           │                       │                                      ║
║           ▼                       ▼                                      ║
║   ┌───────────────┐     ┌──────────────────────────────────┐             ║
║   │     USER      │     │  SESSION SCRIBE  (async)         │             ║
║   │               │     │                                  │             ║
║   │ reads question│     │  • extract memories  (LLM call)  │             ║
║   │ types answer  │     │    → MEMORY BANK (FAISS index)   │             ║
║   └───────┬───────┘     │                                  │             ║
║           │             │  • mark coverage  (LLM call)     │             ║
║           │    answer   │    → SessionAgenda               │             ║
║           │             │      is_covered=True + notes     │             ║
║           │             └──────────────────────────────────┘             ║
║           │                                                              ║
║           │                                      ▼ (every 3 turns)       ║
║           │                          ┌───────────────────────┐           ║
║           │                          │   STRATEGIC PLANNER   │           ║
║           │                          │   (async background)  │           ║
║           │                          │  1. brainstorm subtopics          ║
║           │                          │  2. detect insights (novelty 1-5) ║
║           │                          │  3. score rollouts:               ║
║           │                          │     U = αC - βCost + γEmergence   ║
║           │                          │  4. generate questions            ║
║           │                          └─────────┼─────────────┘           ║
║           │                                    │ → strategic_state       ║
║           │                                    ▼   (read next turn)      ║
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
║  ReportOrchestrator → UpdateUserPortrait  (fills portrait from memories) ║
║  memory_bank.save_to_file()  (content + embeddings JSON)                 ║
║  session_agenda saved  (coverage state, notes, portrait)                 ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**Three things to hold in your head:**

1. **SessionAgenda is the shared scoreboard** — Interviewer picks questions from it, Scribe updates it after each answer, Planner scores rollouts against it.
2. **The message bus is fire-and-forget** — subscribers run via `asyncio.create_task()`; the Interviewer doesn't wait for the Scribe before forming the next question.
3. **Scribe writes to memory, Interviewer reads from it** — they never talk directly; the memory bank is the channel between past sessions and the current one.

---

## New SparkMe — Weekly Work Tracking

Same architecture, with changes marked `◄── NEW`:

### Data model

```
user_portrait.json (stable, LLM-updated)     WeeklySnapshot (volatile, overwritten each week)
┌──────────────────────────────────────┐     ┌──────────────────────────────────────┐
│ role:                                │     │ week_number: 14                      │
│   functional_role: "..."             │     │ tasks:                               │
│   collaboration_structure: {...}     │     │   - description: "client deck prep"  │
│   primary_tools: []                  │     │     time_share: 0.30                 │
│   motivations_and_goals: []          │     │     ai_involved: true                │
│                                      │     │     ai_tool: "ChatGPT"               │
│ ai_relationship:                     │     │     ai_purpose: "first drafts"       │
│   trust_by_task_type: [...]          │     │ collaborators_this_week: []          │
│   preferred_interaction_style: "..." │     │ notable_events: "..."                │
│   known_pain_points: []              │     └──────────────────────────────────────┘
│   known_bright_spots: []             │       saved to: snapshot_week_N.json
└──────────────────────────────────────┘       loaded next week for Scribe comparison
  updated by: SessionCoordinator (LLM)
  evolves slowly across sessions
```

### Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║  STARTUP                                                                 ║
║                                                                          ║
║  ▶ topics_intake.json OR topics_weekly.json  ◄── NEW: selected by        ║
║    session_type="intake"|"weekly"                session_type at init    ║
║                                                                          ║
║  ▶ user_portrait.json  ◄── NEW: stable profile only                      ║
║                                                                          ║
║  [weekly only]  SnapshotManager.load_latest_snapshot()                   ║
║    └── snapshot_week_N-1.json → session_agenda.last_week_snapshot        ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  TURN LOOP                                                               ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌───────────────────────────────────────────────────────────┐          ║
║   │  INTERVIEWER                                              │          ║
║   │  ▶ "weekly_introduction" / "weekly_normal" prompt ◄── NEW │          ║
║   │  ▶ covers snapshot-driven subtopics like any other        │          ║
║   │  ▶ aims to wrap ~10 min, no hard turn cap                 │          ║
║   └──────────────────────────┬────────────────────────────────┘          ║
║                              │                                           ║
║                   [MESSAGE BUS — unchanged]                              ║
║                              │                                           ║
║           ┌──────────────────┴──────────────────┐                        ║
║           │                                     │                        ║
║   ┌───────────────┐     ┌─────────────────────────────────────┐          ║
║   │     USER      │     │  SESSION SCRIBE                     │          ║
║   └───────┬───────┘     │  • extract memories  (unchanged)    │          ║
║           │             │  • mark coverage     (unchanged)    │          ║
║           │  answer     │                                     │          ║
║           │             │  NEW (weekly only):                 │          ║
║           │             │  • _compare_against_snapshot()      │          ║
║           │             │    Q&A vs last_week_snapshot → LLM  │          ║
║           │             │    uses Recall for memory grounding │          ║
║           │             │    adds emergent subtopics for:     │          ║
║           │             │      - inconsistencies              │          ║
║           │             │      - unmentioned items            │          ║
║           │             │    (runs parallel with coverage)    │          ║
║           │             └─────────────────────────────────────┘          ║
║           │                                                              ║
║           └──────────────────────────────────────────────────────────►   ║
║                               [STRATEGIC PLANNER — unchanged]            ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  SHUTDOWN                                                                ║
║                                                                          ║
║  1. UpdateUserPortrait  ◄── updated                                      ║
║     merges dicts + deduplicates lists  ◄── NEW                           ║
║     (was: always overwrote with cleaned string)                          ║
║                                                                          ║
║  2. memory_bank.save_to_file()  — unchanged                              ║
║     session_agenda saved         — unchanged                             ║
║                                                                          ║
║  3. [weekly only]  _generate_and_save_weekly_snapshot()  ◄── NEW         ║
║     session memories → LLM → WeeklySnapshot                              ║
║     → logs/{user_id}/weekly_snapshots/snapshot_week_N.json               ║
║       (loaded at STARTUP next week)                                      ║
╚══════════════════════════════════════════════════════════════════════════╝
```

**What changed, by phase:**

| Phase            | What changed                                                                        |
| ---------------- | ----------------------------------------------------------------------------------- |
| Startup          | Two topic files, new portrait schema (stable only), raw snapshot loaded for Scribe  |
| Interviewer      | Weekly prompts, covers snapshot-driven subtopics alongside regular ones             |
| SessionScribe    | NEW `_compare_against_snapshot()`: adds emergent subtopics for inconsistencies/gaps |
| StrategicPlanner | Completely unchanged                                                                |
| Memory bank      | Completely unchanged                                                                |
| Shutdown         | Portrait updates stable fields only; snapshot generated separately as its own file  |

**Key design decisions:**

1. **Portrait and snapshot are separate concerns.** The portrait holds stable, slowly-evolving profile data (role, AI trust, tools). The snapshot holds volatile weekly data (tasks, time shares, collaborators this week). They never overlap — the portrait is LLM-updated, the snapshot is LLM-extracted and overwritten each week.
2. **Snapshot-driven items are subtopics, not a separate data path.** When the Scribe detects an inconsistency or unmentioned item vs. last week's snapshot, it adds an emergent subtopic. The Interviewer covers it like any other uncovered subtopic. Coverage tracking handles "addressed" status.
3. **Scribe uses recall for memory grounding.** Before adding a subtopic, the Scribe can search prior memories to include the user's original words, making the subtopic description specific (e.g., "user said 'the quarterly deck was eating my Fridays' but hasn't mentioned it").
4. **Snapshot comparison runs in parallel** with subtopic coverage updates (separate locks, `asyncio.gather`). One-turn delay — same pattern as memory extraction.
