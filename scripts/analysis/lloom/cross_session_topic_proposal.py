"""
Cross-Session Topic & Subtopic Proposal Generator

Analyzes all memory banks across participants and interview sessions to identify
emergent themes and propose new topics/subtopics for the interview framework
using keyword-based thematic matching.

Usage:
    python scripts/analysis/cross_session_topic_proposal.py

Outputs:
    - data/configs/topics_proposed.json
    - user_study/cross_session_topic_analysis_report.md
"""

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
EXISTING_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
PROPOSED_TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics_proposed.json"
REPORT_PATH = USER_STUDY_DIR / "cross_session_topic_analysis_report.md"

MIN_PARTICIPANTS_FOR_TOPIC = 10
MIN_PARTICIPANTS_FOR_SUBTOPIC = 8

# ─── Theme Definitions ───────────────────────────────────────────────────────

NEW_TOPIC_THEMES = {
    "AI Quality Assurance and Human Oversight Practices": {
        "description": "How workers verify, validate, and maintain control over AI outputs in practice",
        "theme_groups": {
            "Personal verification strategies for AI outputs": {
                "keywords": ["verify", "validate", "review", "check", "side-by-side", "double-check",
                             "spot-check", "cross-check", "oversight", "manual review"],
                "description": "Individual practices for checking AI-generated results against domain knowledge and ground truth"
            },
            "Organizational AI review processes and policies": {
                "keywords": ["formal review", "organization.*review", "team.*review", "policy",
                             "guideline", "protocol", "standard operating", "SOP",
                             "formal.*process", "override mechanism"],
                "description": "Team and organizational-level mechanisms for reviewing and governing AI-driven work"
            },
            "Impact of AI errors on subsequent workflow changes": {
                "keywords": ["error.*chang", "mistake.*adjust", "failure.*learn",
                             "misinterpret.*adjust", "wrong.*change", "incorrect.*modified",
                             "AI.*wrong", "hallucin"],
                "description": "How encountering AI failures reshapes users' behaviors, trust levels, and workflows"
            },
            "Prompt engineering and input quality practices": {
                "keywords": ["prompt", "descriptive", "instruction", "input quality",
                             "how to ask", "wording", "specify", "clear.*input",
                             "detailed.*input", "craft.*query"],
                "description": "Skills and strategies for formulating effective inputs to AI tools"
            },
            "Criteria for trusting vs. overriding AI recommendations": {
                "keywords": ["trust.*override", "when to trust", "accept.*reject",
                             "confidence.*AI", "rely.*AI", "threshold.*trust",
                             "trust.*depend", "conditional.*trust"],
                "description": "Decision frameworks workers use to determine when AI output is trustworthy enough to act on"
            },
        }
    },
    "AI's Impact on Collaboration and Communication": {
        "description": "How AI tools reshape interpersonal dynamics, team communication, and stakeholder relationships",
        "theme_groups": {
            "AI's role in enhancing professional communication": {
                "keywords": ["email.*reword", "professional.*email", "summariz.*communicat",
                             "tone.*email", "brevity", "clarity.*email", "draft.*email",
                             "rewrite.*message", "polish.*communicat"],
                "description": "Using AI to improve writing quality, tone, and conciseness in professional communication"
            },
            "Impact of AI on team collaboration dynamics": {
                "keywords": ["team.*collaborat", "colleague.*AI", "shared.*tool",
                             "team.*dynamic", "group.*AI", "collaborat.*style", "team.*workflow"],
                "description": "How AI tools change the way teams work together, share knowledge, and coordinate"
            },
            "AI in client/stakeholder-facing work": {
                "keywords": ["client", "customer", "patient", "stakeholder",
                             "present.*data", "report.*stakeholder", "external.*communicat"],
                "description": "How AI shapes interactions with external parties"
            },
            "Cross-functional coordination driven by AI insights": {
                "keywords": ["cross-functional", "interdepartment", "across team",
                             "different department", "cross.*team", "bridg.*team", "multiple.*team"],
                "description": "New coordination needs that emerge when AI insights span organizational boundaries"
            },
            "Peer influence and social learning in AI adoption": {
                "keywords": ["peer.*influence", "colleague.*show", "peer.*learn",
                             "social.*learn", "peer.*tip", "colleague.*demonstrat",
                             "word of mouth", "peer.*recommend"],
                "description": "How workers learn about and adopt AI through informal social networks and peer demonstration"
            },
        }
    },
    "AI Productivity and Time Reallocation": {
        "description": "Concrete impacts of AI on how workers spend their time",
        "theme_groups": {
            "Quantified time savings from AI tool use": {
                "keywords": ["time sav", "faster", "quicker", "speed up", "hour.*sav",
                             "reduc.*time", "efficien", "cut.*time", "save.*time"],
                "description": "Specific examples and estimates of time saved through AI-assisted task completion"
            },
            "Reallocation of saved time to higher-value work": {
                "keywords": ["focus.*more", "freed.*time", "reallocat", "higher.*value",
                             "strategic.*work", "more time.*for", "shift.*focus", "priorit"],
                "description": "How workers redirect time freed by AI toward more complex, creative, or strategic tasks"
            },
            "Perceived vs. actual productivity gains from AI": {
                "keywords": ["productiv", "effective", "output.*quality", "more.*done",
                             "accomplish", "guilt", "feel.*productive", "actual.*benefit"],
                "description": "Gap between expected and realized productivity improvements"
            },
            "Overhead costs and diminishing returns of AI use": {
                "keywords": ["overhead", "time.*learn", "time.*prompt", "time.*verify",
                             "cost.*use", "effort.*AI", "diminish", "not worth", "extra.*step"],
                "description": "Time and cognitive costs of using AI tools"
            },
            "AI's impact on task prioritization and sequencing": {
                "keywords": ["prioriti", "sequenc", "order.*task", "workflow.*chang",
                             "reorder", "task.*first", "batch.*task", "schedul.*AI"],
                "description": "How AI availability changes which tasks get done first and how workflows are structured"
            },
        }
    },
    "Domain Knowledge and AI Limitations": {
        "description": "The interplay between specialized professional knowledge and the boundaries of AI capability",
        "theme_groups": {
            "Types of domain knowledge AI cannot access": {
                "keywords": ["tacit", "institutional", "contextual.*knowledge", "insider",
                             "unwritten", "implicit.*knowledge", "cannot.*know",
                             "doesn't.*know", "AI.*lack"],
                "description": "Categories of professional knowledge that remain inaccessible to AI systems"
            },
            "Examples where domain expertise corrected AI misinterpretation": {
                "keywords": ["correct.*AI", "AI.*wrong", "misinterpret", "AI.*miss",
                             "human.*caught", "override.*AI", "know.*AI.*didn't",
                             "context.*AI.*lack"],
                "description": "Concrete instances where human expertise identified and corrected AI errors"
            },
            "Tasks that remain fundamentally human due to contextual judgment": {
                "keywords": ["human.*judgment", "cannot.*automat", "require.*human",
                             "human.*essential", "human.*still", "judgment.*call",
                             "nuance.*human", "relationship.*human"],
                "description": "Work activities that inherently require human presence, judgment, or relational skills"
            },
            "Strategies for encoding domain knowledge into AI inputs": {
                "keywords": ["context.*provide", "background.*AI", "instruct.*AI",
                             "teach.*AI", "inform.*AI", "custom.*prompt",
                             "tailor.*input", "feed.*context"],
                "description": "Approaches workers use to bridge the domain knowledge gap when working with AI"
            },
            "Evolution of domain knowledge needs as AI grows": {
                "keywords": ["future.*knowledge", "evolv.*skill", "chang.*expertise",
                             "new.*knowledge", "adapt.*skill", "knowledge.*shift",
                             "expertise.*chang"],
                "description": "How the nature and importance of professional domain knowledge shifts with advancing AI"
            },
        }
    },
    "Emotional and Ethical Dimensions of AI Use": {
        "description": "Personal emotional responses, ethical concerns, and identity impacts of integrating AI into work",
        "theme_groups": {
            "Emotional responses to AI use at work": {
                "keywords": ["guilt", "anxiety", "excit", "relief", "frustrat.*AI",
                             "enjoy.*AI", "uncomfortable", "mixed feeling", "conflicted", "nervous"],
                "description": "The range of emotions workers experience when using AI"
            },
            "Ethical concerns about AI in specific professional contexts": {
                "keywords": ["ethic", "moral", "right.*wrong", "appropriate",
                             "responsible.*AI", "should.*AI", "concern.*ethic", "fair"],
                "description": "Industry-specific ethical dilemmas arising from AI use in sensitive domains"
            },
            "Data privacy and security concerns with AI tools": {
                "keywords": ["privacy", "confidential", "sensitive.*data", "HIPAA",
                             "security", "data.*leak", "proprietary", "upload.*sensitive"],
                "description": "Worries about data exposure, compliance, and information security"
            },
            "Impact of AI on creative satisfaction and professional identity": {
                "keywords": ["creative", "identity", "craft", "pride.*work",
                             "authentic", "originality", "ownership",
                             "feel.*replace", "professional.*identity"],
                "description": "How AI affects workers' sense of creative ownership and professional pride"
            },
            "AI's effect on workload perception and work-life balance": {
                "keywords": ["workload", "burnout", "stress", "work-life",
                             "overwhelm", "pressure", "expectation.*more", "always.*on"],
                "description": "Whether AI reduces or intensifies perceived workload"
            },
        }
    },
}

NEW_SUBTOPICS_FOR_EXISTING = {
    "2": {
        "Tasks newly created or transformed by AI integration": {
            "keywords": ["new.*task", "new.*responsib", "transform.*task", "created.*AI",
                         "didn't.*exist", "emerge.*task", "new.*role.*AI"],
            "description": "Work activities newly created or fundamentally reshaped by AI"
        },
    },
    "4": {
        "Role of UI/UX design in AI tool adoption": {
            "keywords": ["interface", "UI", "UX", "design.*tool", "user-friendly",
                         "intuitive", "menu.*driven", "visual.*interface", "easy.*use"],
            "description": "How the design and usability of AI tools influences willingness and speed of adoption"
        },
        "Social stigma or secrecy around AI use": {
            "keywords": ["stigma", "secre", "hide.*AI", "not.*admit",
                         "judg.*AI", "embarrass", "perception.*AI.*use", "cheating"],
            "description": "Social pressures that make workers reluctant to disclose AI tool usage"
        },
    },
    "6": {
        "Comparing and selecting between multiple AI tools": {
            "keywords": ["compar.*AI", "different.*AI.*tool", "switch.*between",
                         "which.*AI", "select.*tool", "prefer.*tool",
                         "ChatGPT.*vs", "multiple.*AI"],
            "description": "How workers evaluate, compare, and choose among the growing landscape of AI tools"
        },
        "Personal non-work AI use influencing professional adoption": {
            "keywords": ["personal.*use", "home.*AI", "outside.*work",
                         "personal.*life", "non-work", "personal.*project"],
            "description": "How experience with AI tools in personal life shapes professional adoption"
        },
    },
    "9": {
        "Prompt engineering as an emerging professional skill": {
            "keywords": ["prompt engineer", "prompt.*skill", "craft.*prompt",
                         "learn.*prompt", "prompt.*master", "prompt.*technique"],
            "description": "The emergence of prompt engineering as a valued professional competency"
        },
        "Impact of AI on career development and advancement": {
            "keywords": ["career.*growth", "promotion", "career.*path",
                         "advancement", "career.*develop", "career.*opportunit",
                         "competitive.*advantage"],
            "description": "How AI proficiency affects career trajectories and professional opportunities"
        },
    },
}

# ─── Analysis ────────────────────────────────────────────────────────────────

def load_all_memories():
    all_memories = []
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        if not mem_path.is_file():
            continue
        with open(mem_path) as f:
            data = json.load(f)
        for m in data.get("memories", []):
            all_memories.append({"pid": pid, **m})
    return all_memories


def load_existing_topics():
    with open(EXISTING_TOPICS_PATH) as f:
        return json.load(f)


def match_theme(memory, keywords):
    combined = " ".join([
        memory.get("title", ""),
        memory.get("text", ""),
        memory.get("source_interview_response", ""),
    ]).lower()
    return any(re.search(kw, combined, re.IGNORECASE) for kw in keywords)


def analyze_theme(all_memories, keywords):
    matches = defaultdict(list)
    for m in all_memories:
        if match_theme(m, keywords):
            matches[m["pid"]].append(m)
    return matches


def analyze_all_themes(all_memories):
    results = {"new_topics": {}, "new_subtopics": {}}

    for topic_name, topic_def in NEW_TOPIC_THEMES.items():
        topic_result = {
            "description": topic_def["description"],
            "subtopics": {},
            "total_memories": 0,
            "total_participants": set(),
        }
        for sub_name, sub_def in topic_def["theme_groups"].items():
            matches = analyze_theme(all_memories, sub_def["keywords"])
            n_mem = sum(len(v) for v in matches.values())
            topic_result["subtopics"][sub_name] = {
                "description": sub_def["description"],
                "n_memories": n_mem,
                "n_participants": len(matches),
                "example_memories": [
                    {"participant": pid[:8] + "...", "title": mems[0]["title"],
                     "text": mems[0]["text"][:200]}
                    for pid, mems in list(matches.items())[:3]
                ],
            }
            topic_result["total_memories"] += n_mem
            topic_result["total_participants"].update(matches.keys())

        topic_result["total_participants"] = list(topic_result["total_participants"])
        results["new_topics"][topic_name] = topic_result

    for topic_id, subtopics in NEW_SUBTOPICS_FOR_EXISTING.items():
        results["new_subtopics"][topic_id] = {}
        for sub_name, sub_def in subtopics.items():
            matches = analyze_theme(all_memories, sub_def["keywords"])
            results["new_subtopics"][topic_id][sub_name] = {
                "description": sub_def["description"],
                "n_memories": sum(len(v) for v in matches.values()),
                "n_participants": len(matches),
            }

    return results


# ─── Output ──────────────────────────────────────────────────────────────────

def generate_proposed_json(results, existing_topics):
    proposed = json.loads(json.dumps(existing_topics))
    existing_map = {str(i + 1): t for i, t in enumerate(proposed)}

    for topic_id, subtopics in results["new_subtopics"].items():
        if topic_id in existing_map:
            for name, info in subtopics.items():
                if info["n_participants"] >= MIN_PARTICIPANTS_FOR_SUBTOPIC:
                    existing_map[topic_id]["subtopics"].append(name)

    for topic_name, topic_data in results["new_topics"].items():
        if len(topic_data["total_participants"]) < MIN_PARTICIPANTS_FOR_TOPIC:
            continue
        subtopics = [
            name for name, info in topic_data["subtopics"].items()
            if info["n_participants"] >= MIN_PARTICIPANTS_FOR_SUBTOPIC
        ]
        if subtopics:
            proposed.append({"topic": topic_name, "subtopics": subtopics})

    return proposed


def generate_report(results, all_memories):
    n_part = len(set(m["pid"] for m in all_memories))
    existing_topics = load_existing_topics()
    topic_names = {str(i + 1): t["topic"] for i, t in enumerate(existing_topics)}

    lines = [
        "# Cross-Session Topic & Subtopic Proposal Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Participants:** {n_part} | **Memories:** {len(all_memories)}",
        "",
        "## Proposed New Topics", "",
    ]

    for name, data in results["new_topics"].items():
        n = len(data["total_participants"])
        lines.append(f"### {name}")
        lines.append(f"_{data['description']}_")
        lines.append(f"- **Evidence:** {data['total_memories']} memories, {n} participants")
        lines.append(f"- **Qualifies:** {'YES' if n >= MIN_PARTICIPANTS_FOR_TOPIC else 'NO'}")
        lines.append("")
        lines.append("| Subtopic | Memories | Participants | Qualifies |")
        lines.append("|----------|----------|-------------|-----------|")
        for sub, info in data["subtopics"].items():
            q = "Yes" if info["n_participants"] >= MIN_PARTICIPANTS_FOR_SUBTOPIC else "No"
            lines.append(f"| {sub} | {info['n_memories']} | {info['n_participants']} | {q} |")
        lines.append("")

    lines.extend(["## New Subtopics for Existing Topics", ""])
    lines.append("| Existing Topic | Proposed Subtopic | Participants | Qualifies |")
    lines.append("|---------------|-------------------|-------------|-----------|")
    for tid, subtopics in results["new_subtopics"].items():
        tname = topic_names.get(tid, f"Topic {tid}")
        for name, info in subtopics.items():
            q = "Yes" if info["n_participants"] >= MIN_PARTICIPANTS_FOR_SUBTOPIC else "No"
            lines.append(f"| {tname} | {name} | {info['n_participants']} | {q} |")

    return "\n".join(lines)


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("Loading memories...")
    all_memories = load_all_memories()
    existing_topics = load_existing_topics()
    print(f"  {len(all_memories)} memories from "
          f"{len(set(m['pid'] for m in all_memories))} participants")

    print("Analyzing themes...")
    results = analyze_all_themes(all_memories)

    print("\nResults:")
    for name, data in results["new_topics"].items():
        n = len(data["total_participants"])
        marker = "+" if n >= MIN_PARTICIPANTS_FOR_TOPIC else "-"
        print(f"  [{marker}] {name}: {data['total_memories']} mem, {n} part")

    proposed = generate_proposed_json(results, existing_topics)
    with open(PROPOSED_TOPICS_PATH, "w") as f:
        json.dump(proposed, f, indent=4)
    print(f"\nWritten: {PROPOSED_TOPICS_PATH}")

    report = generate_report(results, all_memories)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"Written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
