"""
LLooM concept viewer.

Builds an interactive HTML table showing:
  - Each concept with its name, prompt, summary, and % positive memories
  - Expandable rows: full list of matching memories (score >= threshold) with
    rationale, highlight, participant, subtopic
  - Sort by concept name or positive count
  - Threshold slider to filter by minimum score
  - Search bar across concept names and memory text

Usage:
    .venv-analysis/bin/python scripts/analysis/lloom_viewer.py
"""

import json
import os
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
TOPICS_PATH = BASE_DIR / "configs" / "topics.json"
IN_PATH = USER_STUDY_DIR / "lloom_results.json"
OUT_PATH = USER_STUDY_DIR / "lloom_viewer.html"


def load_topics():
    with open(TOPICS_PATH) as f:
        topics = json.load(f)
    subtopics = {}
    for i, t in enumerate(topics):
        for j, desc in enumerate(t["subtopics"]):
            sid = f"{i+1}.{j+1}"
            subtopics[sid] = {"topic": t["topic"], "desc": desc}
    return subtopics


def build_html(data, subtopics):
    concepts = data["concepts"]
    scores = data["scores"]

    # Group scores by concept
    by_concept = defaultdict(list)
    for s in scores:
        by_concept[s["concept_id"]].append(s)

    # Per concept stats
    concept_stats = []
    for c in concepts:
        cid = c["id"]
        rows = by_concept.get(cid, [])
        n_total = len(rows)
        pos_rows = [r for r in rows if r["score"] >= 0.5]
        n_pos = len(pos_rows)
        pct = round(n_pos / n_total * 100, 1) if n_total else 0
        n_part = len(set(r["pid"] for r in pos_rows))
        concept_stats.append({
            "id": cid,
            "name": c["name"],
            "prompt": c["prompt"],
            "summary": c.get("summary", ""),
            "n_total": n_total,
            "n_pos": n_pos,
            "pct": pct,
            "n_part": n_part,
            "pos_rows": sorted(pos_rows, key=lambda r: -r["score"]),
        })

    concept_stats.sort(key=lambda c: -c["n_pos"])

    # Build memory rows HTML per concept
    def mem_row_html(r):
        score_pct = int(r["score"] * 100)
        score_color = "#22c55e" if r["score"] >= 0.75 else "#f59e0b" if r["score"] >= 0.5 else "#94a3b8"
        sub = subtopics.get(r.get("top_subtopic", ""), {})
        sub_label = f"{r.get('top_subtopic','')} {sub.get('desc','')}" if sub else r.get("top_subtopic", "")
        highlight = (r.get("highlight") or "").replace("\n", " ").strip()
        rationale = (r.get("rationale") or "").replace("\n", " ").strip()
        title = (r.get("title") or "").replace('"', '&quot;')
        mem_text = (r.get("text") or "").replace("<", "&lt;").replace(">", "&gt;")
        return f"""
        <tr class="mem-row" data-score="{r['score']}" data-text="{title.lower()} {rationale.lower()}">
          <td style="padding:6px 8px; vertical-align:top; white-space:nowrap;">
            <span style="display:inline-block;background:{score_color};color:white;border-radius:4px;padding:1px 6px;font-size:11px;font-weight:600;">{score_pct}%</span>
          </td>
          <td style="padding:6px 8px; vertical-align:top; font-size:12px; font-weight:600; max-width:240px;">{title}</td>
          <td style="padding:6px 8px; vertical-align:top; font-size:11px; color:#64748b; max-width:120px;">{sub_label}</td>
          <td style="padding:6px 8px; vertical-align:top; font-size:11px; color:#334155; max-width:340px;">
            <span style="color:#0ea5e9; font-weight:500;">Highlight:</span> {highlight}<br>
            <span style="color:#94a3b8; font-style:italic; margin-top:3px; display:block;">{rationale}</span>
          </td>
        </tr>"""

    rows_html = []
    for i, c in enumerate(concept_stats):
        bar_w = int(c["pct"] * 2.5)  # max 250px at 100%
        bar_color = "#3b82f6" if c["pct"] >= 30 else "#94a3b8"

        mem_rows = "".join(mem_row_html(r) for r in c["pos_rows"])

        rows_html.append(f"""
        <tr class="concept-row" data-name="{c['name'].lower()}" data-summary="{c['summary'].lower()}" data-n="{c['n_pos']}" data-pct="{c['pct']}">
          <td style="padding:10px 12px; vertical-align:middle; cursor:pointer;" onclick="toggle({i})">
            <span id="arrow-{i}" style="font-size:12px; color:#94a3b8; margin-right:6px;">▶</span>
            <strong>{c['name']}</strong>
          </td>
          <td style="padding:10px 12px; vertical-align:middle; font-size:12px; color:#475569; max-width:380px;">{c['prompt']}</td>
          <td style="padding:10px 12px; vertical-align:middle; text-align:center;">
            <span style="font-weight:600; font-size:14px;">{c['n_pos']}</span>
            <span style="color:#94a3b8; font-size:11px;"> / {c['n_total']}</span>
          </td>
          <td style="padding:10px 12px; vertical-align:middle;">
            <div style="display:flex; align-items:center; gap:8px;">
              <div style="background:#e2e8f0; border-radius:3px; height:8px; width:250px; flex-shrink:0;">
                <div style="background:{bar_color}; height:8px; width:{bar_w}px; border-radius:3px;"></div>
              </div>
              <span style="font-size:12px; color:#475569; white-space:nowrap;">{c['pct']}%</span>
            </div>
          </td>
          <td style="padding:10px 12px; vertical-align:middle; text-align:center; font-size:12px; color:#64748b;">{c['n_part']}</td>
        </tr>
        <tr id="expand-{i}" style="display:none;">
          <td colspan="5" style="padding:0; background:#f8fafc; border-bottom:2px solid #e2e8f0;">
            <div style="padding:8px 16px 4px; font-size:11px; color:#64748b; font-style:italic;">{c['summary']}</div>
            <div style="padding:4px 16px 8px;">
              <label style="font-size:11px; color:#64748b;">Min score: </label>
              <input type="range" min="0" max="100" value="50" step="25"
                     oninput="filterMems({i}, this.value/100, document.getElementById('thresh-{i}'))"
                     style="width:120px; vertical-align:middle;">
              <span id="thresh-{i}" style="font-size:11px; color:#475569; margin-left:4px;">50%</span>
              &nbsp;&nbsp;
              <input type="text" placeholder="Search memories..."
                     oninput="searchMems({i}, this.value)"
                     style="padding:2px 8px; font-size:11px; border:1px solid #cbd5e1; border-radius:4px; width:200px;">
            </div>
            <table style="width:100%; border-collapse:collapse;" id="mem-table-{i}">
              <thead>
                <tr style="background:#f1f5f9; font-size:11px; color:#64748b; text-align:left;">
                  <th style="padding:6px 8px; width:55px;">Score</th>
                  <th style="padding:6px 8px;">Memory Title</th>
                  <th style="padding:6px 8px; width:130px;">Subtopic</th>
                  <th style="padding:6px 8px;">Highlight / Rationale</th>
                </tr>
              </thead>
              <tbody id="mem-body-{i}">
                {mem_rows}
              </tbody>
            </table>
          </td>
        </tr>""")

    rows_joined = "\n".join(rows_html)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LLooM Concept Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; color: #1e293b; }}
  h1 {{ font-size: 20px; font-weight: 700; color: #1e293b; }}
  .toolbar {{ display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  table {{ width: 100%; border-collapse: collapse; }}
  .concept-row {{ background: white; border-bottom: 1px solid #e2e8f0; transition: background 0.1s; }}
  .concept-row:hover {{ background: #f0f9ff; }}
  th {{ padding: 10px 12px; font-size: 12px; font-weight: 600; color: #64748b; text-align: left; background: #f1f5f9; border-bottom: 2px solid #e2e8f0; cursor: pointer; user-select: none; }}
  th:hover {{ background: #e2e8f0; }}
  .mem-row:nth-child(odd) {{ background: #ffffff; }}
  .mem-row:nth-child(even) {{ background: #f8fafc; }}
  input[type=text] {{ outline: none; }}
  input[type=text]:focus {{ border-color: #3b82f6; box-shadow: 0 0 0 2px #bfdbfe; }}
</style>
</head>
<body>
<div style="max-width:1400px; margin:0 auto; padding:24px 20px;">

  <div style="margin-bottom:20px;">
    <h1>LLooM Concept Viewer</h1>
    <p style="font-size:13px; color:#64748b; margin-top:4px;">{len(concepts)} concepts discovered across {len(set(s['pid'] for s in scores))} participants · {len(scores)} scored memory-concept pairs</p>
  </div>

  <div class="toolbar" style="margin-bottom:16px; padding:12px 16px; background:white; border:1px solid #e2e8f0; border-radius:8px;">
    <input type="text" id="search" placeholder="Search concepts..."
           oninput="filterConcepts()"
           style="padding:6px 12px; font-size:13px; border:1px solid #cbd5e1; border-radius:6px; width:220px;">
    <label style="font-size:13px; color:#64748b;">Min % positive:
      <input type="range" id="pct-filter" min="0" max="50" value="0" step="5"
             oninput="document.getElementById('pct-val').textContent=this.value+'%'; filterConcepts()"
             style="width:120px; vertical-align:middle; margin: 0 6px;">
      <span id="pct-val" style="font-size:13px; color:#475569;">0%</span>
    </label>
    <button onclick="sortTable('n')" style="padding:5px 12px; font-size:12px; border:1px solid #cbd5e1; border-radius:6px; background:white; cursor:pointer;">Sort by count</button>
    <button onclick="sortTable('pct')" style="padding:5px 12px; font-size:12px; border:1px solid #cbd5e1; border-radius:6px; background:white; cursor:pointer;">Sort by %</button>
    <button onclick="sortTable('name')" style="padding:5px 12px; font-size:12px; border:1px solid #cbd5e1; border-radius:6px; background:white; cursor:pointer;">Sort by name</button>
    <button onclick="collapseAll()" style="padding:5px 12px; font-size:12px; border:1px solid #cbd5e1; border-radius:6px; background:white; cursor:pointer;">Collapse all</button>
  </div>

  <div style="background:white; border:1px solid #e2e8f0; border-radius:8px; overflow:hidden;">
    <table id="main-table">
      <thead>
        <tr>
          <th style="width:180px;">Concept</th>
          <th>Prompt</th>
          <th style="width:110px; text-align:center;">Positive memories</th>
          <th style="width:320px;">Coverage</th>
          <th style="width:90px; text-align:center;">Participants</th>
        </tr>
      </thead>
      <tbody id="table-body">
        {rows_joined}
      </tbody>
    </table>
  </div>

</div>

<script>
function toggle(i) {{
  const row = document.getElementById('expand-' + i);
  const arrow = document.getElementById('arrow-' + i);
  if (row.style.display === 'none') {{
    row.style.display = '';
    arrow.textContent = '▼';
  }} else {{
    row.style.display = 'none';
    arrow.textContent = '▶';
  }}
}}

function collapseAll() {{
  document.querySelectorAll('[id^="expand-"]').forEach(r => r.style.display = 'none');
  document.querySelectorAll('[id^="arrow-"]').forEach(a => a.textContent = '▶');
}}

function filterConcepts() {{
  const q = document.getElementById('search').value.toLowerCase();
  const minPct = parseFloat(document.getElementById('pct-filter').value);
  document.querySelectorAll('.concept-row').forEach(row => {{
    const name = row.dataset.name || '';
    const summary = row.dataset.summary || '';
    const pct = parseFloat(row.dataset.pct || 0);
    const match = (name.includes(q) || summary.includes(q)) && pct >= minPct;
    const idx = row.querySelector('[id^="arrow-"]').id.replace('arrow-', '');
    row.style.display = match ? '' : 'none';
    const exp = document.getElementById('expand-' + idx);
    if (!match && exp) exp.style.display = 'none';
  }});
}}

function filterMems(i, threshold, labelEl) {{
  labelEl.textContent = Math.round(threshold * 100) + '%';
  document.querySelectorAll('#mem-body-' + i + ' .mem-row').forEach(r => {{
    const score = parseFloat(r.dataset.score);
    r.style.display = score >= threshold ? '' : 'none';
  }});
}}

function searchMems(i, q) {{
  const ql = q.toLowerCase();
  document.querySelectorAll('#mem-body-' + i + ' .mem-row').forEach(r => {{
    const text = r.dataset.text || '';
    r.style.display = text.includes(ql) ? '' : 'none';
  }});
}}

function sortTable(by) {{
  const tbody = document.getElementById('table-body');
  const pairs = [];
  // Group concept + expand rows
  const rows = Array.from(tbody.children);
  let i = 0;
  while (i < rows.length) {{
    const conceptRow = rows[i];
    const expandRow = rows[i+1];
    if (conceptRow && conceptRow.classList.contains('concept-row')) {{
      pairs.push([conceptRow, expandRow]);
      i += 2;
    }} else {{
      i++;
    }}
  }}
  pairs.sort((a, b) => {{
    const ra = a[0], rb = b[0];
    if (by === 'n') return parseFloat(rb.dataset.n) - parseFloat(ra.dataset.n);
    if (by === 'pct') return parseFloat(rb.dataset.pct) - parseFloat(ra.dataset.pct);
    if (by === 'name') return ra.dataset.name.localeCompare(rb.dataset.name);
    return 0;
  }});
  pairs.forEach(([cr, er]) => {{
    tbody.appendChild(cr);
    if (er) tbody.appendChild(er);
  }});
}}
</script>
</body>
</html>"""
    return html


def main():
    print("Loading LLooM results...")
    with open(IN_PATH) as f:
        data = json.load(f)

    subtopics = load_topics()
    print(f"  {data['n_concepts']} concepts, {len(data['scores'])} score rows")

    print("Building HTML...")
    html = build_html(data, subtopics)

    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
