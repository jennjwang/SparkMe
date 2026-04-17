"""
Visualize O*NET comparison results organized by O*NET occupation,
with filters for coverage, novelty type, and category.

Usage:
    python analysis/onet/5_visualize_onet_comparison.py
"""

import json
import re
import argparse
from pathlib import Path
from html import escape
from collections import defaultdict


COVERAGE_COLOR = {
    "exact":   "#2ea44f",
    "partial": "#e36209",
    "novel":   "#cb2431",
    "unsure":  "#6f42c1",
}
NOVELTY_COLOR = {
    "ai_augmented": "#0366d6",
    "ai_new":       "#6f42c1",
    "new_non_ai":   "#586069",
}
ABSENT_COLOR = {
    "not_reported": "#e36209",
    "out_of_scope": "#959da5",
}


def build_html(study: list, canonical: list, comparison: list) -> str:
    # ── index comparison data by category ──────────────────────────────────
    comp_by_cat = {}   # category → {canonical_statement → coverage dict}
    absent_by_cat = {} # category → list of absent tasks
    for r in comparison:
        cat = r["category"]
        comp_by_cat[cat] = {
            re.sub(r'^\d+\.\s*', '', c["canonical_task"]): c
            for c in r.get("canonical_coverage", [])
        }
        absent_by_cat[cat] = r.get("onet_absent", [])

    # ── index study tasks: task_statement → verbatim quotes ────────────────
    # Index by both exact statement and lowercased stripped version for fuzzy lookup
    task_to_verbatim = {}
    task_to_verbatim_lower = {}  # stripped lowercase → verbatim
    for r in study:
        for t in r["tasks"]:
            stmt = t.get("task_statement", "")
            if stmt:
                # Deduplicate by normalized text to avoid near-identical quotes
                seen_norm = set()
                verbatim = []
                for v in t.get("sources", {}).values():
                    if v:
                        norm = " ".join(v.lower().split())
                        if norm not in seen_norm:
                            seen_norm.add(norm)
                            verbatim.append(v)
                task_to_verbatim[stmt] = verbatim
                task_to_verbatim_lower[stmt.lower().strip()] = verbatim

    def strip_occupation_prefix(src: str) -> str:
        """Remove '[Occupation Name] ' prefix added during aggregation."""
        if src.startswith("["):
            end = src.find("] ")
            if end != -1:
                return src[end + 2:]
        return src

    def lookup_verbatim(src: str) -> list:
        # Exact match first
        if src in task_to_verbatim:
            return task_to_verbatim[src]
        # Strip occupation prefix and retry
        stripped = strip_occupation_prefix(src)
        if stripped in task_to_verbatim:
            return task_to_verbatim[stripped]
        # Lowercase fuzzy match
        stripped_lower = stripped.lower().strip()
        if stripped_lower in task_to_verbatim_lower:
            return task_to_verbatim_lower[stripped_lower]
        # Partial match — find study stmt that starts with same words
        for study_stmt, vb in task_to_verbatim.items():
            if stripped_lower[:40] in study_stmt.lower():
                return vb
        # Even fuzzier: first 30 chars of each word-tokenized version
        src_words = stripped_lower.split()[:6]
        src_prefix = " ".join(src_words)
        for study_stmt, vb in task_to_verbatim.items():
            if src_prefix[:30] in study_stmt.lower() or study_stmt.lower()[:30] in stripped_lower:
                return vb
        return []

    # ── index canonical tasks by category ──────────────────────────────────
    can_by_cat = {}
    sources_by_stmt = {}
    source_count_by_stmt = {}
    for r in canonical:
        can_by_cat[r["category"]] = r["canonical_tasks"]
        for t in r["canonical_tasks"]:
            raw_quotes = []
            seen_norm = set()
            for src in t.get("source_statements", []):
                for q in lookup_verbatim(src):
                    norm = " ".join(q.lower().split())
                    if norm not in seen_norm:
                        seen_norm.add(norm)
                        raw_quotes.append(q)
            # Remove quotes that are substrings of a longer quote in the same set
            raw_norms = [" ".join(q.lower().split()) for q in raw_quotes]
            quotes = [
                q for i, q in enumerate(raw_quotes)
                if not any(
                    raw_norms[i] in raw_norms[j]
                    for j in range(len(raw_quotes)) if j != i
                )
            ]
            sources_by_stmt[t["canonical_statement"]] = quotes
            source_count_by_stmt[t["canonical_statement"]] = t.get("source_count", 0)

    # ── build occupation → participants mapping ─────────────────────────────
    # Canonical tasks are grouped by O*NET title, so each occupation's category = its title
    occ_map = defaultdict(lambda: {"participants": []})
    for r in study:
        code = r.get("onet_code", "")
        title = r.get("onet_title", "")
        if not code:
            continue
        key = (code, title)
        occ_map[key]["participants"].append({
            "occupation": r["occupation"],
            "category": r.get("category", ""),
            "industry": r.get("onet_industry", ""),
        })

    # ── global stats ───────────────────────────────────────────────────────
    all_cov = [c for r in comparison for c in r.get("canonical_coverage", [])]
    total   = len(all_cov)
    exact   = sum(1 for c in all_cov if c.get("coverage") == "exact")
    partial = sum(1 for c in all_cov if c.get("coverage") == "partial")
    novel   = sum(1 for c in all_cov if c.get("coverage") == "novel")
    ai_aug  = sum(1 for c in all_cov if c.get("novelty_type") == "ai_augmented")
    ai_new  = sum(1 for c in all_cov if c.get("novelty_type") == "ai_new")
    new_non = sum(1 for c in all_cov if c.get("novelty_type") == "new_non_ai")


    # ── build occupation blocks ─────────────────────────────────────────────
    occ_blocks = []
    for (code, title), info in sorted(occ_map.items(), key=lambda x: x[0][1]):
        participants = info["participants"]
        occ_id = code.replace(".", "-")
        cat = title  # canonical tasks are grouped by O*NET title

        # participant tags
        part_tags = " ".join(
            f'<span class="occ-tag">{escape(p["occupation"])}</span>'
            for p in sorted(participants, key=lambda x: x["occupation"])
        )

        tasks = can_by_cat.get(cat, [])
        cov_index = comp_by_cat.get(cat, {})
        absent = [a for a in absent_by_cat.get(cat, []) if a.get("onet_code") == code]

        task_rows = []
        for t in tasks:
            stmt = t["canonical_statement"]
            c = cov_index.get(stmt, {})
            cov = c.get("coverage", "novel")
            ntype = c.get("novelty_type") or ""
            cov_color = COVERAGE_COLOR.get(cov, "#888")
            ntype_color = NOVELTY_COLOR.get(ntype, "#888") if ntype else ""
            onet_match = escape(c.get("best_onet_match") or "—")
            notes = escape(c.get("notes") or "")

            novelty_badge = (
                f'<span class="badge" style="background:{ntype_color}">'
                f'{escape(ntype.replace("_"," "))}</span>'
            ) if ntype else ""

            sources = sources_by_stmt.get(stmt, [])
            sources_html = "".join(f'<li>{escape(s)}</li>' for s in sources)
            n_participants = source_count_by_stmt.get(stmt, 0)
            n_badge = f'<span class="n-badge" title="{n_participants} participant(s)">n={n_participants}</span>'

            if sources:
                stmt_cell = (
                    f'<details class="stmt-details">'
                    f'<summary class="stmt-summary">{escape(stmt)}{n_badge}</summary>'
                    f'<ul class="source-list">{sources_html}</ul>'
                    f'</details>'
                )
            else:
                stmt_cell = escape(stmt) + n_badge

            task_rows.append(
                f'<tr data-coverage="{cov}" data-novelty="{ntype}">'
                f'<td class="td-task">{stmt_cell}</td>'
                f'<td class="td-cov">'
                f'<span class="badge" style="background:{cov_color}">{escape(cov)}</span>'
                f'{novelty_badge}</td>'
                f'<td class="td-match"><span class="match-text">{onet_match}</span></td>'
                f'<td class="td-notes">{notes}</td>'
                f'</tr>'
            )

        absent_rows = []
        for a in absent:
            status = a.get("status", "not_reported")
            sc = ABSENT_COLOR.get(status, "#888")
            absent_rows.append(
                f'<tr data-coverage="absent" data-novelty="">'
                f'<td class="td-task">{escape(a.get("onet_task",""))}</td>'
                f'<td><span class="badge" style="background:{sc}">{escape(status.replace("_"," "))}</span></td>'
                f'<td class="td-notes" colspan="2">{escape(a.get("notes",""))}</td>'
                f'</tr>'
            )

        absent_block = ""
        if absent_rows:
            absent_block = (
                f'<h5 class="section-label">O*NET tasks absent from study</h5>'
                f'<table class="task-table"><thead><tr>'
                f'<th>O*NET Task</th><th>Status</th><th colspan="2">Notes</th>'
                f'</tr></thead><tbody>{"".join(absent_rows)}</tbody></table>'
            )

        occ_blocks.append(
            f'<div class="occ-section">'
            f'<div class="occ-header" onclick="toggleOcc(\'{occ_id}\')">'
            f'<span class="toggle-icon" id="icon-{occ_id}">▶</span>'
            f'<span class="occ-code">{escape(code)}</span>'
            f'<span class="occ-title">{escape(title)}</span>'
            f'<span class="occ-n">n={len(participants)}</span>'
            f'</div>'
            f'<div class="occ-body hidden" id="body-{occ_id}">'
            f'<div class="part-tags">{part_tags}</div>'
            f'<table class="task-table"><thead><tr>'
            f'<th>Canonical Task</th><th>Coverage</th>'
            f'<th>Best O*NET Match</th><th>Notes</th>'
            f'</tr></thead><tbody>{"".join(task_rows)}</tbody></table>'
            f'{absent_block}'
            f'</div>'
            f'</div>'
        )

    # Category filter buttons removed — tasks are now grouped by O*NET title directly

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>O*NET Comparison by Occupation</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         font-size: 14px; background: #f4f5f7; color: #24292e; padding: 24px; }}
  h1 {{ font-size: 22px; margin-bottom: 6px; }}
  .summary {{ color: #586069; font-size: 13px; margin-bottom: 16px; }}

  .summary-stats {{ display: flex; gap: 14px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat-box {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 6px;
               padding: 10px 18px; text-align: center; min-width: 100px; }}
  .stat-num {{ font-size: 26px; font-weight: 700; }}
  .stat-label {{ font-size: 11px; color: #586069; margin-top: 2px; }}

  .filters {{ background: #fff; border: 1px solid #e1e4e8; border-radius: 6px;
              padding: 12px 16px; margin-bottom: 16px; }}
  .filter-row {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }}
  .filter-row:last-child {{ margin-bottom: 0; }}
  .filter-label {{ font-size: 12px; font-weight: 600; color: #586069;
                   text-transform: uppercase; letter-spacing: .04em; min-width: 80px; }}
  .filter-btn {{ padding: 4px 10px; border-radius: 12px; border: 1px solid #ccc;
                 background: #fff; cursor: pointer; font-size: 12px; transition: all .15s; }}
  .filter-btn.active {{ color: #fff; border-color: transparent; }}
  .filter-btn[data-cov="exact"].active   {{ background: #2ea44f; }}
  .filter-btn[data-cov="partial"].active {{ background: #e36209; }}
  .filter-btn[data-cov="novel"].active   {{ background: #cb2431; }}
  .filter-btn[data-nov="ai_augmented"].active {{ background: #0366d6; }}
  .filter-btn[data-nov="ai_new"].active       {{ background: #6f42c1; }}
  .filter-btn[data-nov="new_non_ai"].active   {{ background: #586069; }}
  .controls {{ display: flex; gap: 8px; margin-bottom: 14px; }}
  .controls button {{ padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
                      background: #fff; cursor: pointer; font-size: 13px; }}
  .controls button:hover {{ background: #f0f0f0; }}

  .occ-section {{ margin-bottom: 10px; background: #fff;
                  border: 1px solid #e1e4e8; border-radius: 6px; overflow: hidden; }}
  .occ-header {{ padding: 11px 16px; cursor: pointer; display: flex; align-items: center;
                 gap: 10px; background: #24292e; color: #fff; user-select: none; }}
  .occ-header:hover {{ background: #333d47; }}
  .toggle-icon {{ font-size: 11px; width: 14px; flex-shrink: 0; }}
  .occ-code  {{ font-size: 12px; opacity: .7; flex-shrink: 0; font-family: monospace; }}
  .occ-title {{ font-weight: 700; font-size: 15px; flex: 1; }}
  .occ-n     {{ background: rgba(255,255,255,.2); border-radius: 10px;
                font-size: 12px; font-weight: 600; padding: 1px 8px; flex-shrink: 0; }}
  .occ-cats  {{ font-size: 11px; opacity: .65; }}

  .occ-body {{ padding: 12px 16px 14px; }}
  .part-tags {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 12px; }}
  .occ-tag {{ background: #e8f0fe; color: #1a1a2e; border-radius: 12px;
               padding: 2px 10px; font-size: 12px; }}

  .cat-block {{ margin-bottom: 14px; }}
  .cat-label {{ font-size: 12px; font-weight: 700; color: #0366d6;
                text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }}
  .section-label {{ font-size: 11px; font-weight: 600; color: #586069;
                    text-transform: uppercase; letter-spacing: .04em; margin: 10px 0 5px; }}

  .task-table {{ width: 100%; border-collapse: collapse; }}
  .task-table th {{ text-align: left; padding: 6px 10px; font-size: 11px; font-weight: 600;
                    color: #586069; border-bottom: 2px solid #e1e4e8; white-space: nowrap; }}
  .task-table td {{ padding: 7px 10px; vertical-align: top;
                    border-bottom: 1px solid #f0f0f0; font-size: 13px; }}
  .task-table tr:last-child td {{ border-bottom: none; }}
  .td-task  {{ width: 32%; }}
  .td-cov   {{ width: 16%; }}
  .td-match {{ width: 27%; }}
  .td-notes {{ width: 25%; color: #586069; font-style: italic; font-size: 12px; }}
  .match-text {{ font-size: 12px; }}

  .badge {{ color: #fff; padding: 2px 8px; border-radius: 10px;
             font-size: 11px; font-weight: 600; white-space: nowrap;
             display: inline-block; margin: 1px 2px 1px 0; }}
  .hidden {{ display: none !important; }}
  .dimmed {{ opacity: .25; }}
  .stmt-details {{ cursor: pointer; }}
  .stmt-summary {{ list-style: none; font-size: 13px; line-height: 1.5; }}
  .stmt-summary::-webkit-details-marker {{ display: none; }}
  .stmt-summary::before {{ content: "▶ "; font-size: 10px; color: #0366d6; }}
  details[open] .stmt-summary::before {{ content: "▼ "; }}
  .source-list {{ margin: 6px 0 2px 14px; }}
  .source-list li {{ font-size: 12px; color: #586069; font-style: italic;
                      margin-bottom: 3px; line-height: 1.4; }}
  .n-badge {{ display: inline-block; margin-left: 7px; padding: 1px 6px;
              border-radius: 9px; font-size: 11px; font-weight: 600;
              background: #e1e4e8; color: #586069; vertical-align: middle; }}
</style>
</head>
<body>
<h1>Canonical Tasks vs. O*NET — by Occupation</h1>
<p class="summary">{len(occ_map)} O*NET occupations &middot; {total} canonical tasks &middot; O*NET 30.2</p>

<div class="summary-stats">
  <div class="stat-box"><div class="stat-num">{total}</div><div class="stat-label">Tasks</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#2ea44f">{exact}</div><div class="stat-label">Exact ({100*exact//total}%)</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#e36209">{partial}</div><div class="stat-label">Partial ({100*partial//total}%)</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#cb2431">{novel}</div><div class="stat-label">Novel ({100*novel//total}%)</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#0366d6">{ai_aug}</div><div class="stat-label">AI-augmented</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#6f42c1">{ai_new}</div><div class="stat-label">AI-new</div></div>
  <div class="stat-box"><div class="stat-num" style="color:#586069">{new_non}</div><div class="stat-label">New non-AI</div></div>
</div>

<div class="filters">
  <div class="filter-row">
    <span class="filter-label">Coverage</span>
    <button class="filter-btn active" data-cov="exact"   onclick="toggleCov(this)">Exact</button>
    <button class="filter-btn active" data-cov="partial" onclick="toggleCov(this)">Partial</button>
    <button class="filter-btn active" data-cov="novel"   onclick="toggleCov(this)">Novel</button>
  </div>
  <div class="filter-row">
    <span class="filter-label">Novelty</span>
    <button class="filter-btn active" data-nov="ai_augmented" onclick="toggleNov(this)">AI-augmented</button>
    <button class="filter-btn active" data-nov="ai_new"       onclick="toggleNov(this)">AI-new</button>
    <button class="filter-btn active" data-nov="new_non_ai"   onclick="toggleNov(this)">New non-AI</button>
    <button class="filter-btn active" data-nov=""             onclick="toggleNov(this)">None</button>
  </div>
</div>

<div class="controls">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
</div>

{"".join(occ_blocks)}

<script>
const activeCov = new Set(['exact','partial','novel']);
const activeNov = new Set(['ai_augmented','ai_new','new_non_ai','']);

function applyFilters() {{
  document.querySelectorAll('.task-table tbody tr').forEach(row => {{
    const cov = row.dataset.coverage || '';
    const nov = row.dataset.novelty  || '';
    const covOk = activeCov.has(cov) || cov === 'absent';
    const novOk = activeNov.has(nov);
    row.classList.toggle('hidden', !(covOk && novOk));
  }});
}}

function toggleCov(btn) {{
  const v = btn.dataset.cov;
  btn.classList.toggle('active');
  if (activeCov.has(v)) activeCov.delete(v); else activeCov.add(v);
  applyFilters();
}}
function toggleNov(btn) {{
  const v = btn.dataset.nov;
  btn.classList.toggle('active');
  if (activeNov.has(v)) activeNov.delete(v); else activeNov.add(v);
  applyFilters();
}}
function toggleOcc(id) {{
  const body = document.getElementById('body-' + id);
  const icon = document.getElementById('icon-' + id);
  const hidden = body.classList.contains('hidden');
  body.classList.toggle('hidden', !hidden);
  icon.textContent = hidden ? '▼' : '▶';
}}
function expandAll() {{
  document.querySelectorAll('.occ-body').forEach(el => el.classList.remove('hidden'));
  document.querySelectorAll('.toggle-icon').forEach(el => el.textContent = '▼');
}}
function collapseAll() {{
  document.querySelectorAll('.occ-body').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.toggle-icon').forEach(el => el.textContent = '▶');
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study",      default="analysis/onet/data/study_tasks.json")
    parser.add_argument("--canonical",  default="analysis/onet/data/canonical_tasks.json")
    parser.add_argument("--comparison", default="analysis/onet/data/onet_comparison.json")
    parser.add_argument("--output",     default="analysis/onet/data/onet_comparison_viewer.html")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    with open(root / args.study)      as f: study      = json.load(f)
    with open(root / args.canonical)  as f: canonical  = json.load(f)
    with open(root / args.comparison) as f: comparison = json.load(f)

    html = build_html(study, canonical, comparison)
    out = root / args.output
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote → {out}")


if __name__ == "__main__":
    main()
