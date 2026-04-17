"""
Visualize structured tasks generated from user-study memory banks.

Reads data/study_tasks.json and produces an interactive HTML viewer
that shows each participant's tasks with all schema fields.

Usage:
    python analysis/onet/visualize_study_tasks.py
    python analysis/onet/visualize_study_tasks.py --input data/study_tasks.json --output data/study_tasks_viewer.html
"""

import json
import argparse
from pathlib import Path
from html import escape


# Fields shown in the detail table (task_statement shown separately as banner)
TASK_FIELDS = [
    ("action",             "Action",             "What the person actually does"),
    ("object",             "Object",             "What the action is performed on"),
    ("purpose",            "Purpose",            "Why the task is performed"),
    ("tools",              "Tools",              "Software / equipment used"),
    ("information_sources","Information Sources","Where guidance or inputs come from"),
    ("method",             "Method",             "How the task is carried out"),
    ("judgment",           "Judgment",           "Where discretion is exercised"),
    ("quality_criteria",   "Quality Criteria",   "What counts as done well"),
    ("work_context",       "Work Context",       "Conditions that shape execution"),
    ("frequency",          "Frequency",          "How often the task is performed"),
    ("duration",           "Duration",           "How long it takes"),
    ("skills",             "Skills",             "Capabilities drawn on"),
    ("experience",         "Experience",         "Prior exposure needed"),
    ("training",           "Training",           "How the task is learned"),
]

PRIORITY_FIELDS = {"action", "object", "purpose", "judgment", "quality_criteria",
                   "frequency", "duration", "skills"}


def build_html(results: list) -> str:
    total_tasks = sum(len(r["tasks"]) for r in results)

    # Group participants by category
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in results:
        cat = r.get("category", "Other")
        by_category[cat].append(r)

    participants_html = []
    for cat in sorted(by_category.keys()):
        members = by_category[cat]
        cat_id = escape(cat.replace(" ", "-").replace("&", "and"))

        member_blocks = []
        for r in members:
            uid = escape(r["user_id"])
            occ = escape(r.get("occupation", "Unknown"))
            tasks = r["tasks"]

            task_cards = []
            for i, task in enumerate(tasks, 1):
                statement = escape(task.get("task_statement", ""))
                sources = task.get("sources", {})
                rows = []
                for field_id, label, tooltip in TASK_FIELDS:
                    value = task.get(field_id, "")
                    is_priority = field_id in PRIORITY_FIELDS
                    row_class = "priority" if is_priority else "secondary"
                    quote = sources.get(field_id, "")
                    if value and quote:
                        value_html = (
                            f'<span class="citable" '
                            f'data-quote="{escape(quote)}" '
                            f'onclick="showQuote(this)">'
                            f'{escape(str(value))}</span>'
                        )
                    else:
                        value_html = escape(str(value))
                    rows.append(f"""
                        <tr class="{row_class}" title="{escape(tooltip)}">
                            <td class="field-label">{escape(label)}</td>
                            <td class="field-value">{value_html}</td>
                        </tr>""")

                task_id = f"{uid}-task-{i}"
                task_cards.append(f"""
                <div class="task-card">
                    <div class="task-header" onclick="toggleTask('{task_id}')">
                        <span class="toggle-icon" id="icon-{task_id}">▶</span>
                        {f'<span class="task-statement-inline">{statement}</span>' if statement else "<span class='task-statement-inline'>(no statement)</span>"}
                    </div>
                    <div class="task-detail" id="{task_id}" style="display:none">
                        <table class="task-table"><tbody>{"".join(rows)}</tbody></table>
                    </div>
                </div>""")

            tasks_html = "\n".join(task_cards)
            member_blocks.append(f"""
            <div class="participant" id="{uid}">
                <div class="participant-header" onclick="toggleParticipant('{uid}')">
                    <span class="toggle-icon" id="icon-{uid}">▶</span>
                    <span class="participant-title">{occ}</span>
                    <span class="participant-meta">{uid} &middot; {len(tasks)} task(s)</span>
                </div>
                <div class="participant-body" id="body-{uid}" style="display:none">
                    <div class="task-grid">{tasks_html}</div>
                </div>
            </div>""")

        total_cat_tasks = sum(len(r["tasks"]) for r in members)
        participants_html.append(f"""
        <div class="category-section" id="cat-{cat_id}">
            <div class="category-header" onclick="toggleCategory('{cat_id}')">
                <span class="toggle-icon" id="icon-cat-{cat_id}">▼</span>
                <span class="category-title">{escape(cat)}</span>
                <span class="category-meta">{len(members)} participant(s) &middot; {total_cat_tasks} task(s)</span>
            </div>
            <div class="category-body" id="body-cat-{cat_id}">
                {"".join(member_blocks)}
            </div>
        </div>""")

        task_cards = []
        for i, task in enumerate(tasks, 1):
            statement = escape(task.get("task_statement", ""))
            statement_html = (
                f'<div class="task-statement">{statement}</div>'
                if statement else ""
            )

            sources = task.get("sources", {})
            rows = []
            for field_id, label, tooltip in TASK_FIELDS:
                value = task.get(field_id, "")
                is_priority = field_id in PRIORITY_FIELDS
                is_inferred = "(inferred)" in str(value).lower()
                row_class = "priority" if is_priority else "secondary"
                if is_inferred:
                    row_class += " inferred"
                quote = sources.get(field_id, "")
                if value and quote:
                    value_html = (
                        f'<span class="citable" '
                        f'data-quote="{escape(quote)}" '
                        f'onclick="showQuote(this)">'
                        f'{escape(str(value))}</span>'
                    )
                else:
                    value_html = escape(str(value))
                rows.append(f"""
                    <tr class="{row_class}" title="{escape(tooltip)}">
                        <td class="field-label">{escape(label)}</td>
                        <td class="field-value">{value_html}</td>
                    </tr>""")

            task_id = f"{uid}-task-{i}"
            task_cards.append(f"""
            <div class="task-card">
                <div class="task-header" onclick="toggleTask('{task_id}')">
                    <span class="toggle-icon" id="icon-{task_id}">▶</span>
                    {f'<span class="task-statement-inline">{statement}</span>' if statement else "<span class='task-statement-inline'>(no statement)</span>"}
                </div>
                <div class="task-detail" id="{task_id}" style="display:none">
                    <table class="task-table">
                        <tbody>{"".join(rows)}
                        </tbody>
                    </table>
                </div>
            </div>""")


    participants_section = "\n".join(participants_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Study Tasks Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px;
    background: #f4f5f7;
    color: #24292e;
    padding: 24px;
  }}
  h1 {{ font-size: 22px; margin-bottom: 6px; }}
  .summary {{ color: #586069; margin-bottom: 24px; font-size: 13px; }}
  .controls {{ margin-bottom: 16px; display: flex; gap: 8px; align-items: center; }}
  .controls button {{
    padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 13px;
  }}
  .controls button:hover {{ background: #f0f0f0; }}
  .search-box {{
    padding: 6px 12px; border: 1px solid #ccc; border-radius: 4px;
    font-size: 13px; width: 300px;
  }}
  .category-section {{
    margin-bottom: 20px;
  }}
  .category-header {{
    padding: 10px 16px; cursor: pointer; display: flex;
    align-items: center; gap: 10px; user-select: none;
    background: #24292e; color: #fff; border-radius: 6px;
    margin-bottom: 6px;
  }}
  .category-header:hover {{ background: #333d47; }}
  .category-title {{ font-weight: 700; font-size: 15px; flex: 1; }}
  .category-meta {{ font-size: 12px; opacity: 0.7; }}
  .category-body {{ padding-left: 8px; }}
  .participant {{
    background: #fff; border: 1px solid #e1e4e8;
    border-radius: 6px; margin-bottom: 8px;
    overflow: hidden;
  }}
  .participant-header {{
    padding: 10px 16px; cursor: pointer; display: flex;
    align-items: center; gap: 10px; user-select: none;
    background: #fafbfc;
    border-bottom: 1px solid #e1e4e8;
  }}
  .participant-header:hover {{ background: #f0f4f8; }}
  .toggle-icon {{ font-size: 11px; color: #888; width: 14px; }}
  .participant-title {{ font-weight: 600; font-size: 14px; flex: 1; }}
  .participant-meta {{ font-size: 12px; color: #888; font-family: monospace; }}
  .participant-body {{ padding: 16px; }}
  .task-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(460px, 1fr));
    gap: 14px;
  }}
  .task-card {{
    border: 1px solid #e1e4e8; border-radius: 5px; overflow: hidden;
  }}
  .task-header {{
    background: #0366d6; color: #fff;
    padding: 8px 12px; font-weight: 600; font-size: 13px;
    cursor: pointer; display: flex; align-items: baseline; gap: 8px;
  }}
  .task-header:hover {{ background: #0256b3; }}
  .task-header .toggle-icon {{ font-size: 10px; flex-shrink: 0; }}
  .task-statement-inline {{
    font-weight: 400; font-size: 12px; opacity: 0.9;
    font-style: italic; line-height: 1.4;
  }}
  .task-statement {{
    background: #e8f0fe; color: #1a1a2e;
    padding: 8px 12px; font-size: 13px; font-style: italic;
    border-bottom: 1px solid #c8d8f8; line-height: 1.5;
  }}
  .task-table {{ width: 100%; border-collapse: collapse; }}
  .task-table tr {{ border-bottom: 1px solid #f0f0f0; }}
  .task-table tr:last-child {{ border-bottom: none; }}
  .task-table tr.priority {{ background: #fff; }}
  .task-table tr.secondary {{ background: #fafbfc; }}
  .task-table tr.inferred {{ opacity: 0.72; }}
  .task-table td {{ padding: 5px 10px; vertical-align: top; }}
  .field-label {{
    font-weight: 600; font-size: 12px; color: #444;
    white-space: nowrap; width: 140px; min-width: 140px;
  }}
  .field-value {{ font-size: 13px; color: #24292e; }}
  .hidden {{ display: none !important; }}
  .legend {{ font-size: 12px; color: #586069; margin-bottom: 12px; }}
  .citable {{
    cursor: pointer; border-bottom: 1px dashed #0366d6; color: inherit;
  }}
  .citable:hover {{ background: #e8f0fe; border-radius: 2px; }}
  #quote-popover {{
    display: none; position: fixed; z-index: 1000;
    background: #fff; border: 1px solid #ccc; border-radius: 6px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    max-width: 420px; padding: 12px 16px;
    font-size: 13px; line-height: 1.6;
  }}
  #quote-popover .quote-label {{
    font-size: 11px; font-weight: 600; color: #888;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px;
  }}
  #quote-popover .quote-text {{
    color: #24292e; font-style: italic;
  }}
  #quote-popover .quote-close {{
    position: absolute; top: 6px; right: 10px;
    cursor: pointer; font-size: 16px; color: #888; line-height: 1;
  }}
  #quote-popover .quote-close:hover {{ color: #333; }}
</style>
</head>
<body>
<h1>Study Tasks Viewer</h1>
<p class="summary">{len(results)} participants &middot; {total_tasks} total tasks generated from worker responsibility memories</p>

<div class="controls">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
  <input class="search-box" type="text" placeholder="Search occupation or user ID..." oninput="filterParticipants(this.value)">
</div>
<p class="legend">
  White rows = priority fields &nbsp;|&nbsp; Grey rows = secondary fields &nbsp;|&nbsp;
  Faded rows = inferred (not directly stated in memories) &nbsp;|&nbsp;
  Blue banner = task statement
</p>

{participants_section}

<div id="quote-popover">
  <span class="quote-close" onclick="closeQuote()">✕</span>
  <div class="quote-label">Verbatim quote</div>
  <div class="quote-text" id="quote-text"></div>
</div>

<script>
function toggleParticipant(uid) {{
  const body = document.getElementById('body-' + uid);
  const icon = document.getElementById('icon-' + uid);
  if (body.style.display === 'none') {{
    body.style.display = '';
    icon.textContent = '▼';
  }} else {{
    body.style.display = 'none';
    icon.textContent = '▶';
  }}
}}

function toggleTask(tid) {{
  const detail = document.getElementById(tid);
  const icon = document.getElementById('icon-' + tid);
  if (detail.style.display === 'none') {{
    detail.style.display = '';
    icon.textContent = '▼';
  }} else {{
    detail.style.display = 'none';
    icon.textContent = '▶';
  }}
}}

function toggleCategory(cid) {{
  const body = document.getElementById('body-cat-' + cid);
  const icon = document.getElementById('icon-cat-' + cid);
  if (body.style.display === 'none') {{
    body.style.display = '';
    icon.textContent = '▼';
  }} else {{
    body.style.display = 'none';
    icon.textContent = '▶';
  }}
}}

function expandAll() {{
  document.querySelectorAll('.category-body, .participant-body').forEach(el => el.style.display = '');
  document.querySelectorAll('.task-detail').forEach(el => el.style.display = '');
  document.querySelectorAll('.toggle-icon').forEach(el => el.textContent = '▼');
}}

function collapseAll() {{
  document.querySelectorAll('.category-body, .participant-body').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.task-detail').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.toggle-icon').forEach(el => el.textContent = '▶');
}}

function showQuote(el) {{
  const pop = document.getElementById('quote-popover');
  const text = document.getElementById('quote-text');
  text.textContent = '"' + el.dataset.quote + '"';
  pop.style.display = 'block';
  // Position near the clicked element
  const rect = el.getBoundingClientRect();
  const top = Math.min(rect.bottom + 6, window.innerHeight - 180);
  const left = Math.min(rect.left, window.innerWidth - 440);
  pop.style.top = top + 'px';
  pop.style.left = Math.max(8, left) + 'px';
}}

function closeQuote() {{
  document.getElementById('quote-popover').style.display = 'none';
}}

document.addEventListener('click', function(e) {{
  const pop = document.getElementById('quote-popover');
  if (!pop.contains(e.target) && !e.target.classList.contains('citable')) {{
    pop.style.display = 'none';
  }}
}});

function filterParticipants(query) {{
  query = query.toLowerCase();
  document.querySelectorAll('.participant').forEach(el => {{
    const text = el.querySelector('.participant-title').textContent.toLowerCase()
                 + el.querySelector('.participant-meta').textContent.toLowerCase();
    el.classList.toggle('hidden', query.length > 0 && !text.includes(query));
  }});
  // Show categories that have at least one visible participant
  document.querySelectorAll('.category-section').forEach(sec => {{
    const hasVisible = [...sec.querySelectorAll('.participant')].some(p => !p.classList.contains('hidden'));
    sec.classList.toggle('hidden', query.length > 0 && !hasVisible);
  }});
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Visualize study tasks as an HTML viewer")
    parser.add_argument("--input", default="analysis/onet/data/study_tasks.json")
    parser.add_argument("--output", default="analysis/onet/data/study_tasks_viewer.html")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    input_path = root / args.input
    output_path = root / args.output

    with open(input_path) as f:
        results = json.load(f)

    html = build_html(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    total = sum(len(r["tasks"]) for r in results)
    print(f"Wrote {len(results)} participants, {total} tasks → {output_path}")


if __name__ == "__main__":
    main()
