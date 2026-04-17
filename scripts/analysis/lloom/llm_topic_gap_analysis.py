"""
LLM-Driven Topic Gap Analysis

Replicates the manual cross-session analysis process:
1. Load all memories across participants
2. Compress into per-participant summaries
3. Ask an LLM to identify themes NOT covered by existing topics
4. Aggregate theme proposals across batches
5. Final LLM pass to consolidate and propose new topics/subtopics

Usage:
    python scripts/llm_topic_gap_analysis.py

Outputs:
    - data/configs/topics_proposed_llm.json
    - user_study/llm_topic_gap_analysis_report.md
"""

import json
import os
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
EXISTING_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
PROPOSED_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics_proposed_llm.json"
REPORT_PATH = USER_STUDY_DIR / "llm_topic_gap_analysis_report.md"

load_dotenv(BASE_DIR / ".env")

MODEL = "gpt-4o"
BATCH_SIZE = 8  # participants per LLM call (for batch mode)
TWO_PASS = True  # True = 2-pass (titles-first), False = batch mode


# ─── Data Loading ────────────────────────────────────────────────────────────

def load_existing_topics():
    with open(EXISTING_TOPICS_PATH) as f:
        return json.load(f)


def format_existing_topics(topics):
    """Format existing topics as a readable string for prompts."""
    lines = []
    for i, t in enumerate(topics):
        tid = i + 1
        lines.append(f"Topic {tid}: {t['topic']}")
        for j, sub in enumerate(t["subtopics"]):
            lines.append(f"  {tid}.{j+1}: {sub}")
    return "\n".join(lines)


def load_all_memories():
    """Load all memories grouped by participant."""
    participants = {}
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        if not mem_path.is_file():
            continue
        with open(mem_path) as f:
            data = json.load(f)
        memories = data.get("memories", [])
        if memories:
            participants[pid] = memories
    return participants


def compress_participant(pid, memories, max_memories=30):
    """
    Compress a participant's memories into a concise summary for the LLM.
    Keeps the most informative fields, truncates if too many.
    """
    # Sort by timestamp, take a spread if too many
    if len(memories) > max_memories:
        step = len(memories) / max_memories
        indices = [int(i * step) for i in range(max_memories)]
        memories = [memories[i] for i in indices]

    lines = [f"## Participant {pid[:12]}... ({len(memories)} memories)"]
    for m in memories:
        title = m.get("title", "")
        text = m.get("text", "")[:200]
        response = m.get("source_interview_response", "")[:200]
        links = m.get("subtopic_links", [])
        link_str = ", ".join(f"{l['subtopic_id']}(imp={l['importance']})" for l in links[:3])

        lines.append(f"- **{title}**")
        lines.append(f"  Text: {text}")
        if response:
            lines.append(f"  Response: {response}")
        if link_str:
            lines.append(f"  Links: {link_str}")
        lines.append("")

    return "\n".join(lines)


# ─── LLM Calls ──────────────────────────────────────────────────────────────

def call_llm(client, system_prompt, user_prompt, temperature=0.3):
    """Make an LLM call with retry."""
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=4096,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"  LLM call failed (attempt {attempt+1}): {e}")
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
    return None


def identify_gaps_batch(client, existing_topics_str, participant_summaries):
    """
    Ask the LLM to identify themes in a batch of participants that are
    NOT adequately covered by the existing topic framework.
    """
    system_prompt = """You are an expert qualitative researcher analyzing interview data about AI's impact on the workforce.

You will be given:
1. An existing interview topic framework (10 topics with subtopics)
2. Memory summaries from several interview participants

Your task: Identify recurring THEMES in the memories that are NOT adequately captured by the existing topics and subtopics.

Focus on themes that:
- Appear across multiple participants (not idiosyncratic to one person)
- Represent genuinely distinct concepts, not just rewordings of existing subtopics
- Would yield qualitatively different interview questions if added

For each gap theme you find, provide:
- A concise theme name
- A 1-2 sentence description of what it covers
- Which participants show evidence of it (by their ID prefix)
- 2-3 specific memory titles as evidence
- Why this is NOT covered by existing topics (which topic is closest and why it falls short)

Output as JSON array:
[
  {
    "theme": "...",
    "description": "...",
    "participants_with_evidence": ["pid1...", "pid2..."],
    "evidence_titles": ["...", "..."],
    "closest_existing_topic": "Topic X: ...",
    "why_gap": "..."
  }
]

Be selective — only report themes with clear evidence from 2+ participants. Quality over quantity."""

    user_prompt = f"""## Existing Topic Framework

{existing_topics_str}

## Participant Memory Summaries

{participant_summaries}

Identify themes in these memories that are NOT adequately covered by the existing framework. Return JSON only."""

    return call_llm(client, system_prompt, user_prompt)


def consolidate_themes(client, existing_topics_str, all_batch_themes):
    """
    Final LLM pass: consolidate themes from all batches into a coherent
    set of proposed new topics and subtopics.
    """
    system_prompt = """You are an expert qualitative researcher finalizing a topic framework analysis.

You will be given:
1. The existing 10-topic interview framework
2. Gap themes identified across multiple batches of participant data

Your task: Consolidate these into a final proposal of:
A) NEW TOPICS (entirely new top-level topics with subtopics)
B) NEW SUBTOPICS for existing topics

Rules:
- Merge similar/overlapping themes into coherent topics
- A new topic should be supported by evidence from at least 5 different participants
- A new subtopic should be supported by at least 3 participants
- Clearly distinguish what's NEW vs what overlaps with existing coverage
- Provide participant count and example evidence for each proposal
- Be specific about why each proposal represents a genuine gap

Output as JSON:
{
  "new_topics": [
    {
      "topic_name": "...",
      "description": "Why this deserves to be a standalone topic",
      "subtopics": [
        {
          "name": "...",
          "description": "...",
          "participant_count": N,
          "evidence_titles": ["...", "..."]
        }
      ],
      "total_participant_count": N,
      "why_not_covered": "Which existing topic is closest and why this is distinct"
    }
  ],
  "new_subtopics_for_existing": [
    {
      "existing_topic": "Topic N: Name",
      "subtopic_name": "...",
      "description": "...",
      "participant_count": N,
      "evidence_titles": ["...", "..."],
      "why_gap": "Why existing subtopics don't cover this"
    }
  ]
}"""

    themes_text = json.dumps(all_batch_themes, indent=2)

    user_prompt = f"""## Existing Topic Framework

{existing_topics_str}

## Gap Themes from All Batches ({len(all_batch_themes)} themes total)

{themes_text}

Consolidate these into a final proposal of new topics and subtopics. Merge overlapping themes. Only keep proposals with strong cross-participant evidence. Return JSON only."""

    return call_llm(client, system_prompt, user_prompt, temperature=0.2)


# ─── Output Generation ───────────────────────────────────────────────────────

def generate_proposed_json(consolidated, existing_topics):
    """Generate topics_proposed_llm.json from consolidated results."""
    proposed = json.loads(json.dumps(existing_topics))  # deep copy

    # Add new subtopics to existing topics
    topic_name_to_idx = {}
    for i, t in enumerate(proposed):
        topic_name_to_idx[f"Topic {i+1}"] = i
        topic_name_to_idx[t["topic"]] = i

    for new_sub in consolidated.get("new_subtopics_for_existing", []):
        # Try to match existing topic
        existing = new_sub.get("existing_topic", "")
        idx = None
        for key, i in topic_name_to_idx.items():
            if key in existing or existing in key:
                idx = i
                break
        # Try matching by number
        if idx is None:
            import re
            match = re.search(r"Topic\s*(\d+)", existing)
            if match:
                idx = int(match.group(1)) - 1

        if idx is not None and 0 <= idx < len(proposed):
            proposed[idx]["subtopics"].append(new_sub["subtopic_name"])

    # Add new topics
    for new_topic in consolidated.get("new_topics", []):
        proposed.append({
            "topic": new_topic["topic_name"],
            "subtopics": [s["name"] for s in new_topic.get("subtopics", [])],
        })

    return proposed


def generate_report(consolidated, all_batch_themes, n_participants, n_memories, n_batches):
    """Generate the markdown report."""
    lines = [
        "# LLM-Driven Topic Gap Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Model:** {MODEL}",
        f"**Participants analyzed:** {n_participants}",
        f"**Total memories analyzed:** {n_memories}",
        f"**Analysis batches:** {n_batches}",
        "",
        "## Method",
        "",
        "This analysis replicates manual qualitative coding using an LLM:",
        f"1. Compressed {n_participants} participants' memories into summaries",
        f"2. Sent {n_batches} batches of {BATCH_SIZE} participants to {MODEL}",
        "3. Asked the LLM to identify themes NOT covered by existing 10-topic framework",
        f"4. Collected {len(all_batch_themes)} raw gap themes across batches",
        "5. Final consolidation pass merged overlapping themes into proposals",
        "",
        "---",
        "",
    ]

    # Raw themes from batches
    lines.append(f"## Raw Gap Themes ({len(all_batch_themes)} from {n_batches} batches)")
    lines.append("")
    for i, theme in enumerate(all_batch_themes):
        lines.append(f"### {i+1}. {theme.get('theme', 'Unknown')}")
        lines.append(f"_{theme.get('description', '')}_")
        lines.append("")
        pids = theme.get("participants_with_evidence", [])
        lines.append(f"- **Participants:** {len(pids)} ({', '.join(str(p)[:8] + '...' for p in pids)})")
        evidence = theme.get("evidence_titles", [])
        if evidence:
            lines.append(f"- **Evidence:** {'; '.join(evidence[:3])}")
        closest = theme.get("closest_existing_topic", "")
        why = theme.get("why_gap", "")
        if closest:
            lines.append(f"- **Closest existing:** {closest}")
        if why:
            lines.append(f"- **Why gap:** {why}")
        lines.append("")

    lines.extend(["---", ""])

    # Consolidated proposals
    lines.append("## Consolidated Proposals")
    lines.append("")

    lines.append("### New Topics")
    lines.append("")
    for nt in consolidated.get("new_topics", []):
        lines.append(f"#### {nt['topic_name']}")
        lines.append(f"_{nt.get('description', '')}_")
        lines.append("")
        lines.append(f"- **Total participants with evidence:** {nt.get('total_participant_count', '?')}")
        lines.append(f"- **Why not covered:** {nt.get('why_not_covered', '')}")
        lines.append("")
        lines.append("| Subtopic | Description | Participants | Evidence |")
        lines.append("|----------|-------------|-------------|----------|")
        for sub in nt.get("subtopics", []):
            ev = "; ".join(sub.get("evidence_titles", [])[:2])
            lines.append(f"| {sub['name']} | {sub.get('description', '')} | "
                         f"{sub.get('participant_count', '?')} | {ev} |")
        lines.append("")

    lines.append("### New Subtopics for Existing Topics")
    lines.append("")
    lines.append("| Existing Topic | New Subtopic | Description | Participants | Why Gap |")
    lines.append("|---------------|-------------|-------------|-------------|---------|")
    for ns in consolidated.get("new_subtopics_for_existing", []):
        lines.append(f"| {ns.get('existing_topic', '')} | {ns['subtopic_name']} | "
                     f"{ns.get('description', '')} | {ns.get('participant_count', '?')} | "
                     f"{ns.get('why_gap', '')} |")
    lines.append("")

    return "\n".join(lines)


# ─── Two-Pass Mode Functions ─────────────────────────────────────────────────

def build_titles_summary(participants):
    """Build a compact summary using titles + short text for all participants."""
    lines = []
    for pid, memories in participants.items():
        mem_lines = []
        for m in memories:
            title = m.get("title", "")
            text = m.get("text", "")[:120]
            mem_lines.append(f"  - {title}: {text}")
        lines.append(f"### {pid[:12]}... ({len(memories)} memories)")
        lines.extend(mem_lines)
        lines.append("")
    return "\n".join(lines)


def identify_gaps_all_at_once(client, existing_topics_str, titles_summary):
    """
    Single-pass: send ALL participant titles to the LLM and ask for gap themes.
    Works because titles-only fits within context window.
    """
    system_prompt = """You are an expert qualitative researcher analyzing interview data about AI's impact on the workforce.

You will be given:
1. An existing interview topic framework (10 topics, 48 subtopics)
2. Memory titles and short texts from ALL interview participants (37 people)

Your task: Identify 10-20 recurring THEMES across participants that are NOT adequately captured by the existing topics and subtopics.

Think like a researcher doing open coding:
- Read through all the memory titles looking for patterns
- Group related concepts into themes
- Check each theme against the existing framework — is it truly a gap?
- Count how many different participants show evidence of each theme

Focus on:
- Themes that appear across MANY participants (5+), not just one or two
- Concepts that would require genuinely NEW interview questions to explore
- Behavioral patterns, not just attitudes (e.g., "how people verify AI outputs" vs. "whether they trust AI")
- Emotional/social dimensions the framework misses
- Practical workflow details the framework glosses over

Do NOT report:
- Themes already well-covered by existing subtopics (even if many memories touch them)
- Participant-specific details (one person's job description is not a theme)
- Generic observations ("people use AI differently" — be specific about HOW)

For each gap theme, provide:
- theme: A concise name (5-10 words)
- description: What this theme covers and why it matters (2-3 sentences)
- participant_count: Approximate number of participants showing this theme
- evidence_titles: 3-5 specific memory titles as evidence (from different participants)
- closest_existing: Which existing topic/subtopic is nearest
- why_gap: Why the existing coverage falls short (be specific)

Output as JSON array. Be thorough — aim for 10-20 themes."""

    user_prompt = f"""## Existing Topic Framework

{existing_topics_str}

## All Participant Memories (titles + short text)

{titles_summary}

Identify 10-20 gap themes. Return JSON array only."""

    return call_llm(client, system_prompt, user_prompt, temperature=0.2)


def consolidate_into_topics(client, existing_topics_str, gap_themes):
    """
    Pass 2: Take the identified gap themes and organize them into
    proposed new topics and subtopics.
    """
    system_prompt = """You are an expert qualitative researcher designing an interview topic framework.

You will be given:
1. The existing 10-topic interview framework
2. 10-20 gap themes identified from cross-participant analysis

Your task: Organize these gap themes into a proposal for:
A) NEW TOPICS — group related themes into coherent new top-level topics, each with 3-5 subtopics
B) NEW SUBTOPICS — themes that fit better as additions to existing topics

Guidelines:
- Each new topic should represent a distinct conceptual area, not overlap heavily with existing topics
- Subtopics within a new topic should be specific enough to drive distinct interview questions
- Include participant evidence counts — a new topic should have 10+ participants across its subtopics
- Write subtopic descriptions as you would for an interviewer: clear, actionable, specific
- If a gap theme is really just an expansion of an existing subtopic, propose it as a new subtopic, not a new topic
- Be willing to propose 3-5 new topics if the evidence supports it

Output as JSON:
{
  "new_topics": [
    {
      "topic_name": "Short Topic Name",
      "description": "Why this is a distinct area worth exploring in interviews",
      "subtopics": [
        {
          "name": "Subtopic description (written for an interviewer)",
          "description": "What this subtopic covers",
          "participant_count": N,
          "evidence_titles": ["...", "..."]
        }
      ],
      "total_participant_count": N,
      "why_not_covered": "Which existing topics are closest and why they fall short"
    }
  ],
  "new_subtopics_for_existing": [
    {
      "existing_topic": "Topic N: Name",
      "subtopic_name": "New subtopic description",
      "description": "What this covers that existing subtopics miss",
      "participant_count": N,
      "evidence_titles": ["...", "..."],
      "why_gap": "Specific gap explanation"
    }
  ]
}"""

    themes_text = json.dumps(gap_themes, indent=2)

    user_prompt = f"""## Existing Topic Framework

{existing_topics_str}

## Identified Gap Themes

{themes_text}

Organize these into proposed new topics and new subtopics for existing topics. Return JSON only."""

    return call_llm(client, system_prompt, user_prompt, temperature=0.2)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("LLM-DRIVEN TOPIC GAP ANALYSIS")
    print("=" * 60)

    client = OpenAI()

    # Load data
    print("\n[1/5] Loading data...")
    existing_topics = load_existing_topics()
    existing_topics_str = format_existing_topics(existing_topics)
    participants = load_all_memories()
    total_memories = sum(len(mems) for mems in participants.values())
    print(f"  {len(participants)} participants, {total_memories} memories")
    print(f"  {len(existing_topics)} existing topics")

    if TWO_PASS:
        # ── Two-pass mode: titles summary → gap identification → consolidation ──

        print("\n[2/5] Building titles summary...")
        titles_summary = build_titles_summary(participants)
        print(f"  Summary size: {len(titles_summary):,} chars (~{len(titles_summary)//4:,} tokens)")

        print(f"\n[3/5] Identifying gap themes (single pass, all participants)...")
        result = identify_gaps_all_at_once(client, existing_topics_str, titles_summary)
        all_batch_themes = []
        if result:
            result = result.strip()
            if result.startswith("```"):
                result = result.split("\n", 1)[1]
                result = result.rsplit("```", 1)[0]
            try:
                all_batch_themes = json.loads(result)
                print(f"  Found {len(all_batch_themes)} gap themes")
            except json.JSONDecodeError:
                print(f"  JSON parse error")
                print(f"  Raw: {result[:300]}...")

        n_batches = 1
        print(f"\n[4/5] Consolidating {len(all_batch_themes)} themes into topics...")
        consolidation_result = consolidate_into_topics(
            client, existing_topics_str, all_batch_themes
        )

    else:
        # ── Batch mode: compress → batch analyze → consolidate ──

        print("\n[2/5] Compressing participant memories...")
        compressed = {}
        for pid, memories in participants.items():
            compressed[pid] = compress_participant(pid, memories)
        print(f"  Compressed {len(compressed)} participants")

        print(f"\n[3/5] Running batch gap analysis ({BATCH_SIZE} participants/batch)...")
        pids = list(compressed.keys())
        batches = [pids[i:i + BATCH_SIZE] for i in range(0, len(pids), BATCH_SIZE)]
        all_batch_themes = []

        for batch_idx, batch_pids in enumerate(batches):
            batch_text = "\n\n".join(compressed[pid] for pid in batch_pids)
            print(f"  Batch {batch_idx + 1}/{len(batches)} "
                  f"({len(batch_pids)} participants)...", end=" ", flush=True)

            result = identify_gaps_batch(client, existing_topics_str, batch_text)
            if result:
                result = result.strip()
                if result.startswith("```"):
                    result = result.split("\n", 1)[1]
                    result = result.rsplit("```", 1)[0]
                try:
                    themes = json.loads(result)
                    all_batch_themes.extend(themes)
                    print(f"found {len(themes)} themes")
                except json.JSONDecodeError:
                    print(f"JSON parse error, skipping")
                    print(f"    Raw: {result[:200]}...")
            else:
                print("no response")

        n_batches = len(batches)
        print(f"  Total raw themes: {len(all_batch_themes)}")

        print(f"\n[4/5] Consolidating {len(all_batch_themes)} themes...")
        consolidation_result = consolidate_themes(client, existing_topics_str, all_batch_themes)

    consolidated = {}
    if consolidation_result:
        consolidation_result = consolidation_result.strip()
        if consolidation_result.startswith("```"):
            consolidation_result = consolidation_result.split("\n", 1)[1]
            consolidation_result = consolidation_result.rsplit("```", 1)[0]
        try:
            consolidated = json.loads(consolidation_result)
            n_topics = len(consolidated.get("new_topics", []))
            n_subs = len(consolidated.get("new_subtopics_for_existing", []))
            print(f"  Proposed: {n_topics} new topics, {n_subs} new subtopics")
        except json.JSONDecodeError:
            print(f"  JSON parse error in consolidation")
            print(f"  Raw: {consolidation_result[:300]}...")
            consolidated = {"new_topics": [], "new_subtopics_for_existing": []}

    # Generate outputs
    print("\n[5/5] Generating outputs...")

    proposed = generate_proposed_json(consolidated, existing_topics)
    with open(PROPOSED_TOPICS_PATH, "w") as f:
        json.dump(proposed, f, indent=4)
    print(f"  Written: {PROPOSED_TOPICS_PATH}")

    report = generate_report(
        consolidated, all_batch_themes,
        len(participants), total_memories, n_batches
    )
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"  Written: {REPORT_PATH}")

    # Also save raw intermediate data for inspection
    raw_path = USER_STUDY_DIR / "llm_topic_gap_raw.json"
    with open(raw_path, "w") as f:
        json.dump({
            "batch_themes": all_batch_themes,
            "consolidated": consolidated,
        }, f, indent=2)
    print(f"  Written: {raw_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("PROPOSED NEW TOPICS")
    print("=" * 60)
    for nt in consolidated.get("new_topics", []):
        print(f"\n  {nt['topic_name']} ({nt.get('total_participant_count', '?')} participants)")
        print(f"    {nt.get('description', '')}")
        for sub in nt.get("subtopics", []):
            print(f"    - {sub['name']} ({sub.get('participant_count', '?')} part)")

    print("\n" + "=" * 60)
    print("PROPOSED NEW SUBTOPICS FOR EXISTING TOPICS")
    print("=" * 60)
    for ns in consolidated.get("new_subtopics_for_existing", []):
        print(f"\n  [{ns.get('existing_topic', '')}]")
        print(f"    + {ns['subtopic_name']} ({ns.get('participant_count', '?')} part)")

    print("\nDone!")


if __name__ == "__main__":
    main()
