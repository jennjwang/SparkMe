"""
Visualize canonical tasks aggregated from user-study data.

Usage:
    python analysis/onet/visualize_canonical_tasks.py
    python analysis/onet/visualize_canonical_tasks.py --input data/canonical_tasks.json --output data/canonical_tasks_viewer.html
"""

import json
import argparse
from pathlib import Path
from html import escape


VALIDITY_COLOR = {
    "valid": "#2ea44f",
    "too_broad": "#e36209",
    "not_a_task": "#cb2431",
}


def build_html(results: list) -> str:
    total_canonical = sum(len(r["canonical_tasks"]) for r in results)
    total_raw = sum(r["n_raw_tasks"] for r in results)

    category_blocks = []
    for r in results:
        cat = escape(r["category"])
        cat_id = cat.replace(" ", "-").replace("&", "and").replace("/", "-")
        occupations = r.get("occupations", [])
        canonical_tasks = r["canonical_tasks"]
        valid_count = sum(1 for t in canonical_tasks if t.get("validity") == "valid")

        occ_tags = " ".join(
            f'<span class="occ-tag">{escape(o)}</span>' for o in occupations
        )

        task_rows = []
        for t in canonical_tasks:
            stmt = escape(t.get("canonical_statement", ""))
            abbrev = escape(t.get("abbreviated_phrase", ""))
            count = t.get("source_count", "?")
            validity = t.get("validity", "valid")
            notes = escape(t.get("notes", ""))
            sources = t.get("source_statements", [])
            validity_color = VALIDITY_COLOR.get(validity, "#888")

            sources_html = "".join(
                f'<li>{escape(s)}</li>' for s in sources
            )
            sources_block = f'<ul class="source-list">{sources_html}</ul>' if sources else ""

            task_rows.append(f"""
            <tr class="task-row {'dimmed' if validity != 'valid' else ''}">
                <td class="td-statement">
                    <div class="canonical-stmt">{stmt}</div>
                    {f'<div class="notes-text">{notes}</div>' if notes else ""}
                    <details class="sources-details">
                        <summary>{len(sources)} source statement(s)</summary>
                        {sources_block}
                    </details>
                </td>
                <td class="td-abbrev"><code>{abbrev}</code></td>
                <td class="td-count">{count}</td>
                <td class="td-validity">
                    <span class="validity-badge" style="background:{validity_color}">
                        {escape(validity.replace('_', ' '))}
                    </span>
                </td>
            </tr>""")

        category_blocks.append(f"""
        <div class="category-section">
            <div class="category-header" onclick="toggleSection('{cat_id}')">
                <span class="toggle-icon" id="icon-{cat_id}">▼</span>
                <span class="cat-title">{cat}</span>
                <span class="cat-meta">
                    {valid_count} valid &middot; {len(canonical_tasks)} canonical &middot; {r['n_raw_tasks']} raw &middot; {len(occupations)} occupation(s)
                </span>
            </div>
            <div class="category-body" id="body-{cat_id}">
                <div class="occ-tags">{occ_tags}</div>
                <table class="task-table">
                    <thead>
                        <tr>
                            <th>Canonical Task Statement</th>
                            <th>Abbreviated Phrase</th>
                            <th>N</th>
                            <th>Validity</th>
                        </tr>
                    </thead>
                    <tbody>{"".join(task_rows)}</tbody>
                </table>
            </div>
        </div>""")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Canonical Tasks Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 14px; background: #f4f5f7; color: #24292e; padding: 24px;
  }}
  h1 {{ font-size: 22px; margin-bottom: 6px; }}
  .summary {{ color: #586069; margin-bottom: 20px; font-size: 13px; }}
  .controls {{ margin-bottom: 16px; display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
  .controls button {{
    padding: 6px 14px; border: 1px solid #ccc; border-radius: 4px;
    background: #fff; cursor: pointer; font-size: 13px;
  }}
  .controls button:hover {{ background: #f0f0f0; }}
  .controls label {{ font-size: 13px; display: flex; align-items: center; gap: 5px; }}

  .category-section {{ margin-bottom: 18px; background: #fff; border: 1px solid #e1e4e8; border-radius: 6px; overflow: hidden; }}
  .category-header {{
    padding: 12px 16px; cursor: pointer; display: flex; align-items: center;
    gap: 10px; background: #24292e; color: #fff; user-select: none;
  }}
  .category-header:hover {{ background: #333d47; }}
  .toggle-icon {{ font-size: 11px; width: 14px; }}
  .cat-title {{ font-weight: 700; font-size: 15px; flex: 1; }}
  .cat-meta {{ font-size: 12px; opacity: 0.75; }}
  .category-body {{ padding: 12px 16px 16px; }}

  .occ-tags {{ margin-bottom: 10px; display: flex; flex-wrap: wrap; gap: 6px; }}
  .occ-tag {{
    background: #e8f0fe; color: #1a1a2e; border-radius: 12px;
    padding: 2px 10px; font-size: 12px;
  }}

  .task-table {{ width: 100%; border-collapse: collapse; }}
  .task-table th {{
    text-align: left; padding: 7px 10px; font-size: 12px; font-weight: 600;
    color: #586069; border-bottom: 2px solid #e1e4e8; white-space: nowrap;
  }}
  .task-table td {{ padding: 8px 10px; vertical-align: top; border-bottom: 1px solid #f0f0f0; }}
  .task-row.dimmed {{ opacity: 0.55; }}
  .task-row:last-child td {{ border-bottom: none; }}

  .td-statement {{ width: 60%; }}
  .td-abbrev {{ width: 18%; }}
  .td-count {{ width: 5%; text-align: center; font-weight: 600; }}
  .td-validity {{ width: 10%; }}

  .canonical-stmt {{ font-size: 13px; line-height: 1.5; margin-bottom: 4px; }}
  .notes-text {{ font-size: 12px; color: #e36209; font-style: italic; margin-bottom: 4px; }}
  code {{ background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-size: 12px; }}

  .validity-badge {{
    color: #fff; padding: 2px 8px; border-radius: 10px;
    font-size: 11px; font-weight: 600; white-space: nowrap;
  }}

  .sources-details {{ margin-top: 4px; }}
  .sources-details summary {{
    font-size: 11px; color: #0366d6; cursor: pointer; user-select: none;
  }}
  .source-list {{ margin: 4px 0 0 16px; }}
  .source-list li {{ font-size: 12px; color: #586069; font-style: italic; margin-bottom: 2px; }}

  .hidden {{ display: none !important; }}
</style>
</head>
<body>
<h1>Canonical Tasks Viewer</h1>
<p class="summary">
  {len(results)} categories &middot; {total_canonical} canonical tasks (from {total_raw} raw task statements)
  &middot; applying O*NET &amp; DOT heuristics
</p>

<div class="controls">
  <button onclick="expandAll()">Expand All</button>
  <button onclick="collapseAll()">Collapse All</button>
  <label>
    <input type="checkbox" id="hide-invalid" onchange="toggleInvalid(this.checked)">
    Hide non-valid tasks
  </label>
</div>

{"".join(category_blocks)}

<script>
function toggleSection(id) {{
  const body = document.getElementById('body-' + id);
  const icon = document.getElementById('icon-' + id);
  const hidden = body.style.display === 'none';
  body.style.display = hidden ? '' : 'none';
  icon.textContent = hidden ? '▼' : '▶';
}}
function expandAll() {{
  document.querySelectorAll('.category-body').forEach(el => el.style.display = '');
  document.querySelectorAll('.toggle-icon').forEach(el => el.textContent = '▼');
}}
function collapseAll() {{
  document.querySelectorAll('.category-body').forEach(el => el.style.display = 'none');
  document.querySelectorAll('.toggle-icon').forEach(el => el.textContent = '▶');
}}
function toggleInvalid(hide) {{
  document.querySelectorAll('.task-row.dimmed').forEach(el => {{
    el.classList.toggle('hidden', hide);
  }});
}}
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="analysis/onet/data/canonical_tasks.json")
    parser.add_argument("--output", default="analysis/onet/data/canonical_tasks_viewer.html")
    args = parser.parse_args()

    root = Path(__file__).parent.parent.parent
    with open(root / args.input) as f:
        results = json.load(f)

    html = build_html(results)
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)

    total = sum(len(r["canonical_tasks"]) for r in results)
    print(f"Wrote {len(results)} categories, {total} canonical tasks → {output_path}")


if __name__ == "__main__":
    main()
