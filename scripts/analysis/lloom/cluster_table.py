"""
Interactive cluster table for exploring LLM or agglomerative cluster contents.

Each row = one cluster. Expandable to show all memory titles + full text.
Sortable by: n, purity, cos, dominant topic, emergent flag.
Filterable by topic and emergent status.

Usage:
    .venv-analysis/bin/python scripts/analysis/cluster_table.py         # agglomerative
    .venv-analysis/bin/python scripts/analysis/cluster_table.py --llm   # LLM clusters
"""

import sys
import json
import os
import html as htmllib
from pathlib import Path

USE_LLM = "--llm" in sys.argv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"

if USE_LLM:
    CLUSTERS_PATH = USER_STUDY_DIR / "llm_clustering.json"
    LABELS_CACHE = USER_STUDY_DIR / "llm_cluster_labels_cache.json"
    OUT_PATH = USER_STUDY_DIR / "cluster_table_llm.html"
    TITLE = "LLM (GPT-4o) Cluster Explorer"
else:
    CLUSTERS_PATH = USER_STUDY_DIR / "hierarchical_clustering.json"
    LABELS_CACHE = USER_STUDY_DIR / "cluster_labels_cache.json"
    OUT_PATH = USER_STUDY_DIR / "cluster_table.html"
    TITLE = "Agglomerative Cluster Explorer"


def load_topics():
    with open(TOPICS_PATH) as f:
        topics = json.load(f)
    subtopics = {}
    for i, t in enumerate(topics):
        for j, desc in enumerate(t["subtopics"]):
            subtopics[f"{i+1}.{j+1}"] = {"topic_name": t["topic"], "desc": desc, "topic_idx": i}
    return topics, subtopics


def load_memories():
    memories = {}
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        if not mem_path.is_file():
            continue
        with open(mem_path) as f:
            mem_data = json.load(f)
        for m in mem_data.get("memories", []):
            memories[m["id"]] = {"pid": pid, **m}
    return memories


def dominant_topic(memory_ids, memories, subtopics, topics):
    from collections import Counter
    counts = Counter()
    for mid in memory_ids:
        m = memories.get(mid, {})
        links = m.get("subtopic_links", [])
        if links:
            best = max(links, key=lambda l: l.get("importance", 0))
            sid = best["subtopic_id"]
            if sid in subtopics:
                counts[subtopics[sid]["topic_idx"]] += 1
    if not counts:
        return 0, "Unknown"
    tidx = counts.most_common(1)[0][0]
    return tidx, topics[tidx]["topic"]


def build_rows(cluster_meta, memories, subtopics, topics):
    from collections import Counter
    rows = []
    for cid_str, meta in cluster_meta.items():
        memory_ids = meta.get("memory_ids", [])
        label = meta.get("label", f"Cluster {cid_str}")
        n = meta.get("n", len(memory_ids))
        n_part = meta.get("n_part", 0)
        purity = meta.get("purity", 0)
        cos = meta.get("cos", 0)
        is_emergent = meta.get("is_emergent", False)

        tidx, tname = dominant_topic(memory_ids, memories, subtopics, topics)

        # Count memories per subtopic
        sub_counts = Counter()
        for mid in memory_ids:
            m = memories.get(mid, {})
            links = m.get("subtopic_links", [])
            if links:
                best = max(links, key=lambda l: l.get("importance", 0))
                sid = best["subtopic_id"]
                if sid in subtopics:
                    sub_counts[sid] += 1
        subtopic_breakdown = [
            {"sid": sid, "desc": subtopics[sid]["desc"], "topic_name": subtopics[sid]["topic_name"], "count": cnt}
            for sid, cnt in sub_counts.most_common()
        ]

        mem_details = []
        for mid in memory_ids:
            m = memories.get(mid)
            if not m:
                continue
            links = m.get("subtopic_links", [])
            top_link = max(links, key=lambda l: l.get("importance", 0)) if links else None
            sub_str = f"{top_link['subtopic_id']}: {subtopics.get(top_link['subtopic_id'], {}).get('desc', '')[:50]}" if top_link else ""
            mem_details.append({
                "title": m.get("title", ""),
                "text": m.get("text", ""),
                "subtopic": sub_str,
                "pid": m.get("pid", ""),
            })

        rows.append({
            "cid": int(cid_str),
            "label": label,
            "n": n,
            "n_part": n_part,
            "purity": purity,
            "cos": cos,
            "is_emergent": is_emergent,
            "topic_idx": tidx,
            "topic_name": tname,
            "subtopic_breakdown": subtopic_breakdown,
            "memories": mem_details,
        })

    rows.sort(key=lambda r: -r["n"])
    return rows


def topic_color(tidx):
    import colorsys
    h = tidx / 10
    r, g, b = colorsys.hls_to_rgb(h, 0.88, 0.60)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"


def render_html(rows, topics, title):
    topic_names = [t["topic"] for t in topics]
    topic_colors = {i: topic_color(i) for i in range(len(topics))}

    # Build filter options
    topic_options = "\n".join(
        f'<option value="{i}">{i+1}: {t["topic"][:35]}</option>'
        for i, t in enumerate(topics)
    )

    # Build table rows HTML
    table_rows_html = []
    for r in rows:
        if r["n"] < 10:
            continue
        cid = r["cid"]
        bg = topic_colors.get(r["topic_idx"], "#eee")
        emergent_badge = '<span class="badge emergent">⭐ emergent</span>' if r["is_emergent"] else ""
        purity_class = "good" if r["purity"] >= 0.6 else ("mid" if r["purity"] >= 0.4 else "low")
        cos_class = "good" if r["cos"] >= 0.55 else ("mid" if r["cos"] >= 0.45 else "low")

        # Subtopic breakdown bars
        max_count = r["subtopic_breakdown"][0]["count"] if r["subtopic_breakdown"] else 1
        sub_bars = []
        for sb in r["subtopic_breakdown"]:
            pct = sb["count"] / r["n"] * 100
            bar_w = sb["count"] / max_count * 180
            sub_bars.append(f"""
              <div class="sub-bar-row">
                <div class="sub-bar-label">{htmllib.escape(sb['sid'])}: {htmllib.escape(sb['desc'][:45])}</div>
                <div class="sub-bar-wrap">
                  <div class="sub-bar" style="width:{bar_w:.0f}px"></div>
                  <span class="sub-bar-count">{sb['count']} ({pct:.0f}%)</span>
                </div>
              </div>""")
        sub_breakdown_html = "\n".join(sub_bars)

        # Purity visual bar
        purity_pct = r["purity"] * 100
        purity_color = "#1a7a3a" if r["purity"] >= 0.6 else ("#b07800" if r["purity"] >= 0.4 else "#c0392b")

        # Memory rows
        mem_rows = []
        for m in r["memories"]:
            t = htmllib.escape(m["title"])
            txt = htmllib.escape(m["text"])
            sub = htmllib.escape(m["subtopic"])
            pid = htmllib.escape(m["pid"])
            mem_rows.append(f"""
              <div class="mem-row">
                <div class="mem-title">{t}</div>
                <div class="mem-sub">{sub} &nbsp;·&nbsp; {pid}</div>
                <div class="mem-text">{txt}</div>
              </div>""")

        mem_html = "\n".join(mem_rows)
        label_esc = htmllib.escape(r["label"])
        tname_esc = htmllib.escape(r["topic_name"])

        table_rows_html.append(f"""
        <tr class="cluster-row"
            data-cid="{cid}"
            data-n="{r['n']}"
            data-purity="{r['purity']}"
            data-cos="{r['cos']}"
            data-topic="{r['topic_idx']}"
            data-emergent="{1 if r['is_emergent'] else 0}"
            onclick="toggleExpand({cid})">
          <td class="td-cid">C{cid}</td>
          <td class="td-label">{label_esc} {emergent_badge}</td>
          <td class="td-topic"><span class="topic-pill" style="background:{bg}">{tname_esc[:30]}</span></td>
          <td class="td-num">{r['n']}</td>
          <td class="td-num">{r['n_part']}</td>
          <td class="td-num metric {purity_class}">{r['purity']:.2f}</td>
          <td class="td-num metric {cos_class}">{r['cos']:.3f}</td>
          <td class="td-toggle">▶</td>
        </tr>
        <tr class="expand-row" id="expand-{cid}" style="display:none">
          <td colspan="8">
            <div class="expand-container">
              <div class="expand-left">
                <div class="section-title">Subtopic Breakdown
                  <span class="purity-inline">
                    Purity: <span style="color:{purity_color};font-weight:600">{r['purity']:.2f}</span>
                    <div class="purity-bar-wrap"><div class="purity-bar" style="width:{purity_pct:.0f}%;background:{purity_color}"></div></div>
                  </span>
                </div>
                {sub_breakdown_html}
              </div>
              <div class="expand-right">
                <div class="section-title">Memories ({r['n']})</div>
                {mem_html}
              </div>
            </div>
          </td>
        </tr>""")

    table_body = "\n".join(table_rows_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{htmllib.escape(title)}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          font-size: 13px; background: #f8f8f8; color: #222; }}
  #header {{ background: #fff; border-bottom: 1px solid #ddd;
             padding: 14px 20px; position: sticky; top: 0; z-index: 100;
             display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  #header h1 {{ font-size: 16px; font-weight: 600; }}
  .filter-group {{ display: flex; align-items: center; gap: 6px; }}
  .filter-group label {{ font-size: 12px; color: #555; }}
  select, input[type=text] {{
    font-size: 12px; padding: 4px 7px; border: 1px solid #ccc;
    border-radius: 4px; background: white;
  }}
  #search {{ min-width: 200px; }}
  #reset-btn {{
    font-size: 12px; padding: 4px 12px; border: 1px solid #bbb;
    border-radius: 4px; cursor: pointer; background: #f0f0f0;
  }}
  #reset-btn:hover {{ background: #e0e0e0; }}
  #stats {{ font-size: 11px; color: #888; margin-left: auto; }}

  table {{ width: 100%; border-collapse: collapse; background: white; }}
  thead th {{
    position: sticky; top: 57px; background: #f0f0f0;
    padding: 8px 10px; text-align: left; font-size: 12px;
    border-bottom: 2px solid #ddd; cursor: pointer; user-select: none;
    white-space: nowrap;
  }}
  thead th:hover {{ background: #e4e4e4; }}
  thead th.sorted-asc::after {{ content: " ▲"; font-size: 10px; }}
  thead th.sorted-desc::after {{ content: " ▼"; font-size: 10px; }}

  .cluster-row {{ cursor: pointer; border-bottom: 1px solid #eee; }}
  .cluster-row:hover {{ background: #fafafa; }}
  .cluster-row td {{ padding: 7px 10px; vertical-align: middle; }}
  .td-cid {{ font-size: 11px; color: #888; width: 40px; }}
  .td-label {{ font-weight: 500; max-width: 320px; }}
  .td-topic {{ max-width: 180px; }}
  .td-num {{ text-align: right; width: 60px; }}
  .td-toggle {{ text-align: center; color: #aaa; width: 30px; font-size: 11px; }}
  .cluster-row.open .td-toggle {{ transform: rotate(90deg); display: inline-block; }}

  .topic-pill {{
    display: inline-block; padding: 2px 7px; border-radius: 10px;
    font-size: 11px; font-weight: 500;
  }}
  .badge {{ font-size: 10px; margin-left: 4px; }}
  .metric {{ font-family: monospace; }}
  .good {{ color: #1a7a3a; font-weight: 600; }}
  .mid  {{ color: #b07800; }}
  .low  {{ color: #c0392b; }}

  .expand-row td {{ padding: 0; background: #fafafa; }}
  .expand-container {{
    display: flex; gap: 0;
    border-top: 1px solid #eee; border-bottom: 2px solid #ddd;
  }}
  .expand-left {{
    width: 320px; min-width: 320px; padding: 12px 16px;
    border-right: 1px solid #e0e0e0; background: #f5f5f5;
  }}
  .expand-right {{
    flex: 1; padding: 10px 16px 14px;
    max-height: 500px; overflow-y: auto;
  }}
  .section-title {{
    font-weight: 600; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.05em; color: #666; margin-bottom: 8px;
    display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  }}
  .purity-inline {{ display: flex; align-items: center; gap: 5px; font-weight: normal; text-transform: none; letter-spacing: 0; }}
  .purity-bar-wrap {{ width: 60px; height: 6px; background: #ddd; border-radius: 3px; overflow: hidden; }}
  .purity-bar {{ height: 100%; border-radius: 3px; }}
  .sub-bar-row {{ margin-bottom: 5px; }}
  .sub-bar-label {{ font-size: 11px; color: #444; margin-bottom: 2px; }}
  .sub-bar-wrap {{ display: flex; align-items: center; gap: 6px; }}
  .sub-bar {{ height: 8px; background: #7aaddc; border-radius: 3px; min-width: 2px; }}
  .sub-bar-count {{ font-size: 11px; color: #666; white-space: nowrap; }}
  .mem-row {{
    padding: 8px 0; border-bottom: 1px solid #eee;
  }}
  .mem-row:last-child {{ border-bottom: none; }}
  .mem-title {{ font-weight: 600; margin-bottom: 3px; }}
  .mem-sub {{ font-size: 11px; color: #888; margin-bottom: 4px; }}
  .mem-text {{ color: #444; line-height: 1.5; }}
</style>
</head>
<body>

<div id="header">
  <h1>{htmllib.escape(title)}</h1>
  <div class="filter-group">
    <label>Search</label>
    <input type="text" id="search" placeholder="label or memory…" oninput="applyFilters()">
  </div>
  <div class="filter-group">
    <label>Topic</label>
    <select id="filter-topic" onchange="applyFilters()">
      <option value="">All topics</option>
      {topic_options}
    </select>
  </div>
  <div class="filter-group">
    <label>Emergent only</label>
    <input type="checkbox" id="filter-emergent" onchange="applyFilters()">
  </div>
  <button id="reset-btn" onclick="resetFilters()">Reset</button>
  <span id="stats"></span>
</div>

<table id="cluster-table">
  <thead>
    <tr>
      <th onclick="sortBy('cid')">ID</th>
      <th onclick="sortBy('label')">Cluster Label</th>
      <th onclick="sortBy('topic')">Dominant Topic</th>
      <th onclick="sortBy('n')" class="sorted-desc">n</th>
      <th onclick="sortBy('n_part')">Part.</th>
      <th onclick="sortBy('purity')">Purity</th>
      <th onclick="sortBy('cos')">Intra-cos</th>
      <th></th>
    </tr>
  </thead>
  <tbody id="table-body">
    {table_body}
  </tbody>
</table>

<script>
var sortCol = 'n', sortAsc = false;

function toggleExpand(cid) {{
  var row = document.getElementById('expand-' + cid);
  var tr = row.previousElementSibling;
  if (row.style.display === 'none') {{
    row.style.display = '';
    tr.classList.add('open');
  }} else {{
    row.style.display = 'none';
    tr.classList.remove('open');
  }}
}}

function sortBy(col) {{
  if (sortCol === col) {{ sortAsc = !sortAsc; }}
  else {{ sortCol = col; sortAsc = col === 'label'; }}
  updateSortHeaders();
  applyFilters();
}}

function updateSortHeaders() {{
  document.querySelectorAll('thead th').forEach(function(th) {{
    th.classList.remove('sorted-asc', 'sorted-desc');
  }});
  var cols = ['cid','label','topic','n','n_part','purity','cos'];
  var idx = cols.indexOf(sortCol);
  if (idx >= 0) {{
    var ths = document.querySelectorAll('thead th');
    ths[idx].classList.add(sortAsc ? 'sorted-asc' : 'sorted-desc');
  }}
}}

function applyFilters() {{
  var search = document.getElementById('search').value.toLowerCase();
  var topicFilter = document.getElementById('filter-topic').value;
  var emergentOnly = document.getElementById('filter-emergent').checked;
  var tbody = document.getElementById('table-body');
  var rows = Array.from(tbody.querySelectorAll('tr.cluster-row'));

  // Filter
  var visible = [];
  rows.forEach(function(tr) {{
    var expandRow = document.getElementById('expand-' + tr.dataset.cid);
    var label = tr.querySelector('.td-label').textContent.toLowerCase();
    var memTexts = expandRow ? expandRow.textContent.toLowerCase() : '';
    var matchSearch = !search || label.includes(search) || memTexts.includes(search);
    var matchTopic = !topicFilter || tr.dataset.topic === topicFilter;
    var matchEmergent = !emergentOnly || tr.dataset.emergent === '1';
    var show = matchSearch && matchTopic && matchEmergent;
    tr.style.display = show ? '' : 'none';
    if (expandRow) expandRow.style.display = (show && tr.classList.contains('open')) ? '' : 'none';
    if (show) visible.push(tr);
  }});

  // Sort
  visible.sort(function(a, b) {{
    var va, vb;
    if (sortCol === 'label') {{ va = a.querySelector('.td-label').textContent.trim(); vb = b.querySelector('.td-label').textContent.trim(); return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va); }}
    if (sortCol === 'topic') {{ va = parseInt(a.dataset.topic); vb = parseInt(b.dataset.topic); }}
    else if (sortCol === 'cid') {{ va = parseInt(a.dataset.cid); vb = parseInt(b.dataset.cid); }}
    else if (sortCol === 'n') {{ va = parseInt(a.dataset.n); vb = parseInt(b.dataset.n); }}
    else if (sortCol === 'n_part') {{ va = parseInt(a.querySelector('td:nth-child(5)').textContent); vb = parseInt(b.querySelector('td:nth-child(5)').textContent); }}
    else if (sortCol === 'purity') {{ va = parseFloat(a.dataset.purity); vb = parseFloat(b.dataset.purity); }}
    else if (sortCol === 'cos') {{ va = parseFloat(a.dataset.cos); vb = parseFloat(b.dataset.cos); }}
    return sortAsc ? va - vb : vb - va;
  }});

  visible.forEach(function(tr) {{
    var expandRow = document.getElementById('expand-' + tr.dataset.cid);
    tbody.appendChild(tr);
    if (expandRow) tbody.appendChild(expandRow);
  }});

  document.getElementById('stats').textContent = visible.length + ' clusters shown';
}}

function resetFilters() {{
  document.getElementById('search').value = '';
  document.getElementById('filter-topic').value = '';
  document.getElementById('filter-emergent').checked = false;
  applyFilters();
}}

// Init
applyFilters();
</script>
</body>
</html>"""


def main():
    print(f"Building cluster table ({'LLM' if USE_LLM else 'agglomerative'})...")
    topics, subtopics = load_topics()
    memories = load_memories()

    with open(CLUSTERS_PATH) as f:
        raw = json.load(f)
    cluster_meta = raw.get("fine") or raw.get("clusters") or {}

    # Attach memory_ids from labels cache if not already in meta
    if LABELS_CACHE.exists():
        with open(LABELS_CACHE) as f:
            id_to_label = json.load(f)
        from collections import defaultdict
        cid_to_ids = defaultdict(list)
        for mid, cid in id_to_label.items():
            cid_to_ids[str(cid)].append(mid)
        for cid_str, meta in cluster_meta.items():
            if not meta.get("memory_ids"):
                meta["memory_ids"] = cid_to_ids.get(cid_str, [])

    rows = build_rows(cluster_meta, memories, subtopics, topics)
    print(f"  {len(rows)} clusters, {sum(r['n'] for r in rows)} memories total")

    html = render_html(rows, topics, TITLE)
    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"  Written: {OUT_PATH}")


if __name__ == "__main__":
    main()
