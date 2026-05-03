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

## SparkMe — Intake-Only Work Profiling

SparkMe now runs intake sessions only. Runtime entry points always load
`topics_intake.json`; the older recurring check-in and snapshot comparison flow
is not part of active session creation.

**Active session design decisions:**

1. Sessions are intake-only and always use the intake topic plan.
2. The portrait is the durable cross-session profile and is refreshed from memories.
3. The task-validation and AI-task widgets handle late-session task review and probing.

---

## Current SparkMe — Task Profiling Web App (`simple` branch)

Extends the intake interview with explicit task collection and AI usage probing. The internal agent architecture (Interviewer + SessionScribe + StrategicPlanner) is unchanged; new phases are layered on top via widget messages injected into the chat stream.

### Session flow

```
╔══════════════════════════════════════════════════════════════════════════╗
║  STARTUP                                                                 ║
║                                                                          ║
║  User logs in → / (landing page, enters available_time in minutes)      ║
║  POST /api/start-session  →  session_token                               ║
║  session_type = "intake"                                                 ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 1 — INTERVIEW                                                     ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Same agent loop as original SparkMe:                                   ║
║    Interviewer → message bus → User + SessionScribe                     ║
║    StrategicPlanner runs in background every 3 turns                    ║
║                                                                          ║
║  Frontend polls GET /api/get-messages (500ms interval)                  ║
║  User sends via POST /api/send-message or /api/send-voice               ║
║                                                                          ║
║  When Interviewer decides the conversation is ready:                    ║
║    → emits  profile_confirm_widget  message                             ║
║      (frontend shows "building your profile" indicator,                 ║
║       opens profile panel in background)                                ║
║                                                                          ║
║  Session continues until all_core_topics_completed() OR timeout         ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 2 — TASK VALIDATION WIDGET (TVW)                                  ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Interviewer emits  task_validation_widget  message                     ║
║                                                                          ║
║  TVW runs in the chat pane (not a separate page):                       ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  Regular task batches (3 tasks at a time, up to 18) │                ║
║  │  POST /api/generate-tasks  ← prefetched in bg       │                ║
║  │  POST /api/regenerate-task-description              │                ║
║  │  for each task: confirm / edit / remove             │                ║
║  ├─────────────────────────────────────────────────────┤                ║
║  │  Attention check  (8 clearly unrelated tasks)       │                ║
║  │  POST /api/attention-check-tasks                    │                ║
║  ├─────────────────────────────────────────────────────┤                ║
║  │  AI-era tasks  (capabilities + governance, up to 18)│                ║
║  │  POST /api/ai-era-tasks   ← 3 per batch, prefetched │                ║
║  │  POST /api/ai-attention-check-tasks  (3 distractors)│                ║
║  └─────────────────────────────────────────────────────┘                ║
║                                                                          ║
║  User submits → POST /api/submit-task-validation                        ║
║    → if attention check failed: early exit                              ║
║    → else: portrait updated with Task Inventory, probing loop starts    ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 3 — POST-TVW PROBING                                              ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  For each validated task the Interviewer asks a clarifying question:    ║
║    POST /api/task-followup  {task_text, phase, recent_dialogue}         ║
║    returns {reply, done}                                                ║
║                                                                          ║
║  Phase: "probing" — clarify regular work tasks                          ║
║  When done → immediately moves to Phase 4                               ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  PHASE 4 — AI-RELATED TASK WIDGET (AITW)                                 ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║  Triggered client-side when probing loop completes:                     ║
║                                                                          ║
║  ┌─────────────────────────────────────────────────────┐                ║
║  │  Transition message (bot): "With AI becoming part   │                ║
║  │  of so many workflows..."                           │                ║
║  │  → user answers open question                       │                ║
║  ├─────────────────────────────────────────────────────┤                ║
║  │  AI task widget (AITW):                             │                ║
║  │    POST /api/ai-era-tasks  ← 3 per batch, up to 18  │                ║
║  │    POST /api/ai-attention-check-tasks (3 distractors│                ║
║  │    User selects AI-related tasks                    │                ║
║  ├─────────────────────────────────────────────────────┤                ║
║  │  "Are there any other AI-related tasks...?"         │                ║
║  │    POST /api/task-followup  {phase: "ai_extras"}    │                ║
║  │    done=true here ends the session server-side      │                ║
║  └─────────────────────────────────────────────────────┘                ║
║                                                                          ║
║  Server also emits  ai_usage_widget  after user answers an AI adoption  ║
║  question during the interview phase; handled in parallel.              ║
╚══════════════════════════════════════════════════════════════════════════╝
                                    │
                                    ▼
╔══════════════════════════════════════════════════════════════════════════╗
║  SHUTDOWN                                                                ║
║                                                                          ║
║  POST /api/end-session  →  feedback_widget emitted                      ║
║  User submits → POST /api/submit-feedback                               ║
║                                                                          ║
║  ReportOrchestrator → updates portrait (Task Inventory, Task Grouping   ║
║    Tree) from session memories                                          ║
║  memory_bank.save_to_file()                                             ║
║  session_agenda saved  (end_reason: "completed" | "timeout")           ║
╚══════════════════════════════════════════════════════════════════════════╝
```

### Portrait structure (new fields)

```
user_portrait.json
┌─────────────────────────────────────────────────────────┐
│ ... (stable fields from original portrait schema)       │
│                                                         │
│ Task Inventory: [                                       │
│   "implementing and debugging ML models",               │
│   "writing and revising research papers",               │
│   ...                                                   │
│ ]                                                       │
│                                                         │
│ Task Grouping Tree: {                                   │
│   "group name": ["task A", "task B"],                   │
│   ...                                                   │
│ }                                                       │
│                                                         │
│ AI Task Inventory: [                                    │
│   "reviewing AI-generated code suggestions",            │
│   ...                                                   │
│ ]                                                       │
└─────────────────────────────────────────────────────────┘
```

### Active API endpoints (web app only)

| Endpoint | Phase | Purpose |
|----------|-------|---------|
| `POST /api/start-session` | Startup | Create/reuse session |
| `POST /api/send-message` | Interview | User text turn |
| `POST /api/send-voice` | Interview | User voice turn (transcribe + route) |
| `GET /api/get-messages` | All | Poll for new messages / widgets |
| `POST /api/acknowledge-messages` | All | Clear delivered message buffer |
| `GET /api/get-voice-response` | Interview | Poll TTS readiness (202 = pending) |
| `GET /api/stream-voice-response` | Interview | Stream TTS audio chunks |
| `POST /api/generate-tasks` | TVW | Generate task batch (3 at a time) |
| `POST /api/regenerate-task-description` | TVW | Rewrite task descriptions |
| `POST /api/attention-check-tasks` | TVW | 4 unrelated distractor tasks |
| `POST /api/ai-era-tasks` | AITW | AI-era task batches |
| `POST /api/ai-attention-check-tasks` | AITW | 3 non-AI distractors |
| `POST /api/submit-task-validation` | TVW | Store tasks, start probing |
| `POST /api/task-followup` | Probing / AI extras | Interviewer clarifier reply |
| `POST /api/organize-tasks` | Profile panel | LLM re-group task tree |
| `POST /api/update-portrait` | Profile panel | Persist drag-drop edits |
| `GET /api/session-state` | Visualizer | Full live session state |
| `POST /api/end-session` | Shutdown | Emit feedback widget |
| `POST /api/submit-feedback` | Shutdown | Finalize session |

### Interviewer behavior (current prompt)

**Persona**: Terry Gross-style — warm, curious, conversational. Contractions, short sentences, relaxed cadence. Never sounds like a survey, chatbot, or HR form.

**Task extraction sequence** (for Task Inventory topics):
1. Goals first: "What were your goals for last week?"
2. Deliverables: "What did you actually deliver or complete?"
3. Work backwards: "What did you need to do to get that done?" — tasks emerge from deliverables, not from a generic walk-through.

**Per-turn rules:**
- **One question only.** No double-barreled questions, no compound forms.
- **Anchor to specific content.** Every follow-up must pick a concrete noun, task, or detail from the user's last answer. Generic structural questions after specific tasks have been named are duplicates.
- **No acknowledgment preambles** on routine answers. Jump straight to the next question. Only a brief (~5 word) neutral acknowledgment when the user shared something effortful, uncertain, or emotionally charged.
- **Never evaluate, praise, summarize, or restate.** No "got it", "interesting", "makes sense", "that's helpful context", no paraphrasing openers.
- **Never use a topic-closing summary.** Do NOT say "Thanks so much for walking me through everything!", "That covers a lot of ground", "Great, we've got a solid picture" — these falsely signal the session is wrapping up.
- **Convert instances to recurring patterns**: confirm "Is that something you do regularly, or was that a one-time thing?" before treating a deliverable as a base task.
- **Target length**: one sentence (just the question) when no acknowledgment and no topic transition; two sentences max when either is warranted; never more.

**Transition rules:**
- Subtopic → subtopic within the same parent topic: **no bridge**, ask directly.
- Topic → topic (parent changes): **brief bridge required** (e.g., "Shifting gears a bit —").
  - Role/Context → Task Inventory: two-sentence collaborative goal-framing, then goals-first question.
  - All other topic moves: one natural phrase; avoid stiff pivots ("moving on to section two", "pivoting to").

**Duplicate detection:**
- Before asking, extract the core information goal of each recent interviewer question.
- Do not ask any question whose information goal overlaps — even if reworded, narrowed to an instance, or broadened to a pattern.
- Attempted = covered: if a question was asked and got a partial answer, accept it and move on.

**End condition**: `end_conversation` only when all subtopic `coverage_criteria` are satisfied AND no strategic questions remain. Do NOT improvise new questions or re-ask covered topics in different wording.

---

**What changed vs. earlier variants:**

| Component | Status |
|-----------|--------|
| Interviewer prompt / behavior | **UPDATED** — task extraction sequence, strict no-acknowledgment rules, no topic-closing summaries, anchor-to-specific-content requirement |
| SessionScribe + StrategicPlanner | Unchanged |
| Memory bank + question bank | Unchanged |
| Shutdown / ReportOrchestrator | Extended to write Task Inventory + Task Grouping Tree |
| Task Validation Widget | **NEW** — explicit task elicitation outside interview conversation |
| Post-TVW probing loop | **NEW** — lightweight per-task follow-up via `/api/task-followup` |
| AI-related task widget (AITW) | **NEW** — open question → AI task selection → AI extras probing |
| Profile panel | **NEW** — live drag-and-drop task tree, persisted via portrait updates |
| AI-era task collection | **NEW** — separate batch generation for AI capability/governance tasks |
