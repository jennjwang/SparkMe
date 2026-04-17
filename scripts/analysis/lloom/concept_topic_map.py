"""
Concept ↔ Topic/Subtopic breakdown viewer.

Generates an interactive HTML showing:
  1. Heatmap: topics × concepts (% of topic memories matching each concept)
  2. Heatmap: subtopics × concepts (same, finer-grained)
  3. Reverse view: concepts × topics (% of concept's positive memories from each topic)

Usage:
    .venv-analysis/bin/python scripts/analysis/concept_topic_map.py
"""

import json
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent.parent.parent
OUT_PATH = BASE / "user_study" / "concept_topic_map.html"

def load_data():
    import os
    with open(BASE / 'user_study/lloom_results.json') as f:
        data = json.load(f)
    with open(BASE / 'configs/topics.json') as f:
        topics_raw = json.load(f)

    subtopics = {}
    topic_subtopics = {}
    for i, t in enumerate(topics_raw, 1):
        topic_subtopics[i] = []
        for j, desc in enumerate(t['subtopics'], 1):
            sid = f"{i}.{j}"
            subtopics[sid] = {'num': i, 'topic': t['topic'], 'desc': desc}
            topic_subtopics[i].append(sid)

    concepts = data['concepts']

    # Build mem_id -> {subtopic_id: weight} from all subtopic_links, weighted by importance.
    # A memory with links T6(imp=8), T7(imp=4) contributes weight 0.67 to T6, 0.33 to T7.
    USER_STUDY = BASE / 'user_study'
    mem_weights = {}   # mem_id -> {sid: weight}
    for pid in sorted(os.listdir(USER_STUDY)):
        p = USER_STUDY / pid / 'memory_bank_content.json'
        if not p.is_file():
            continue
        with open(p) as f:
            mem_data = json.load(f)
        for m in mem_data.get('memories', []):
            links = [l for l in m.get('subtopic_links', [])
                     if l.get('subtopic_id', '') in subtopics]
            if not links:
                continue
            total_imp = sum(max(l.get('importance', 1), 1) for l in links)
            mem_weights[m['id']] = {
                l['subtopic_id']: max(l.get('importance', 1), 1) / total_imp
                for l in links
            }

    # Weighted counts: each memory contributes fractionally to each subtopic/topic
    # sub_concept_hits[sid][cid]  = sum of weights where score >= 0.5
    # sub_total_n[sid]            = sum of weights across all score rows for that sid
    sub_concept_hits  = defaultdict(lambda: defaultdict(float))
    topic_concept_hits = defaultdict(lambda: defaultdict(float))
    sub_total_n   = defaultdict(float)
    topic_total_n = defaultdict(float)

    # De-duplicate: each (mem_id, concept_id) should only be counted once per score row
    seen = set()
    for s in data['scores']:
        mem_id = s.get('doc_id', '')
        cid    = s.get('concept_id', '')
        key    = (mem_id, cid)
        if key in seen:
            continue
        seen.add(key)

        weights = mem_weights.get(mem_id, {})
        if not weights:
            continue
        score = s.get('score', 0)

        for sid, w in weights.items():
            tnum_str = sid.split('.')[0]
            if not tnum_str.isdigit():
                continue
            tnum = int(tnum_str)
            if tnum < 1 or tnum > 10:
                continue
            sub_total_n[sid]   += w
            topic_total_n[tnum] += w
            if score >= 0.5:
                sub_concept_hits[sid][cid]   += w
                topic_concept_hits[tnum][cid] += w

    return (concepts, topics_raw, subtopics, topic_subtopics,
            sub_concept_hits, topic_concept_hits,
            dict(sub_total_n), dict(topic_total_n))


def pct_color(pct):
    """Blue heatmap: 0=white, 100=dark blue."""
    if pct <= 0:
        return "#ffffff"
    # clamp to 0-100
    p = min(pct, 100)
    # white -> #1d4ed8 (blue-700)
    r = int(255 - (255 - 29)  * p / 100)
    g = int(255 - (255 - 78)  * p / 100)
    b = int(255 - (255 - 216) * p / 100)
    return f"rgb({r},{g},{b})"


def text_color(pct):
    return "#fff" if pct > 55 else "#1e293b"


def build_html(concepts, topics_raw, subtopics, topic_subtopics,
               sub_concept_hits, topic_concept_hits, sub_total_n, topic_total_n):

    c_names = [c['name'] for c in concepts]
    c_ids   = [c['id']   for c in concepts]
    n_c = len(concepts)

    # ── Helper: build a heatmap table ─────────────────────────────────────────
    def heatmap_table(row_labels, row_ids, hit_fn, total_fn, row_headers, table_id):
        """
        row_labels: list of display strings
        row_ids:    list of keys for hit_fn / total_fn
        hit_fn(row_id, cid) -> int
        total_fn(row_id) -> int
        row_headers: list of (label, colspan) for grouped header rows
        """
        # Column headers
        header_top = ""
        for label, cs in row_headers:
            header_top += f'<th colspan="{cs}" style="background:#1e3a5f;color:#fff;padding:6px 4px;font-size:11px;white-space:nowrap;border:1px solid #cbd5e1;">{label}</th>'

        col_heads = "".join(
            f'<th style="background:#1e40af;color:#fff;padding:4px 3px;font-size:10px;'
            f'writing-mode:vertical-rl;transform:rotate(180deg);white-space:nowrap;'
            f'min-width:22px;max-width:22px;border:1px solid #93c5fd;" '
            f'title="{c_names[j]}">{c_names[j][:28]}</th>'
            for j in range(n_c)
        )

        rows_html = ""
        for i, (label, rid) in enumerate(zip(row_labels, row_ids)):
            total = total_fn(rid)
            cells = ""
            for cid in c_ids:
                hits = hit_fn(rid, cid)
                pct  = round(hits / total * 100) if total else 0
                bg   = pct_color(pct)
                fg   = text_color(pct)
                tip  = f"{hits}/{total} ({pct}%)"
                cells += (
                    f'<td style="background:{bg};color:{fg};text-align:center;'
                    f'font-size:9px;padding:2px 1px;border:1px solid #e2e8f0;" '
                    f'title="{tip}">'
                    f'{"" if pct == 0 else str(pct)}</td>'
                )
            bg_row = "#f8fafc" if i % 2 == 0 else "#ffffff"
            rows_html += (
                f'<tr style="background:{bg_row};">'
                f'<td style="padding:4px 8px;font-size:11px;white-space:nowrap;'
                f'border:1px solid #e2e8f0;font-weight:500;">{label}</td>'
                f'<td style="padding:4px 8px;font-size:11px;text-align:center;'
                f'border:1px solid #e2e8f0;color:#64748b;">{total}</td>'
                f'{cells}</tr>'
            )

        return f"""
        <div style="overflow-x:auto; margin-bottom:32px;">
          <table id="{table_id}" style="border-collapse:collapse; font-family:-apple-system,sans-serif;">
            <thead>
              <tr>
                <th rowspan="2" style="background:#334155;color:#fff;padding:6px 8px;font-size:11px;border:1px solid #cbd5e1;">Topic / Subtopic</th>
                <th rowspan="2" style="background:#334155;color:#fff;padding:6px 8px;font-size:11px;border:1px solid #cbd5e1;">Mems</th>
                {header_top}
              </tr>
              <tr>{col_heads}</tr>
            </thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>"""

    # ── Topic-level heatmap ───────────────────────────────────────────────────
    topic_row_labels = [f"T{i}: {t['topic']}" for i, t in enumerate(topics_raw, 1)]
    topic_row_ids    = list(range(1, 11))

    def topic_hit(tnum, cid):
        return topic_concept_hits.get(tnum, {}).get(cid, 0)
    def topic_total_fn(tnum):
        return topic_total_n.get(tnum, 0)

    # Group concepts by their dominant topic for the column header grouping
    # (just label each concept column with its dominant topic number)
    concept_dom = []
    for c in concepts:
        cid = c['id']
        best_t, best_n = 0, 0
        for tnum in range(1, 11):
            n = topic_concept_hits.get(tnum, {}).get(cid, 0)
            if n > best_n:
                best_n, best_t = n, tnum
        concept_dom.append(best_t)

    # Build grouped header: group consecutive same-dominant-topic concepts
    col_groups = []
    for dom in concept_dom:
        if col_groups and col_groups[-1][0] == dom:
            col_groups[-1] = (dom, col_groups[-1][1] + 1)
        else:
            col_groups.append((dom, 1))
    col_group_headers = [(f"T{dom}" if dom else "?", cs) for dom, cs in col_groups]

    topic_heatmap = heatmap_table(
        topic_row_labels, topic_row_ids,
        topic_hit, topic_total_fn,
        col_group_headers, "topic-table"
    )

    # ── Subtopic-level heatmap ────────────────────────────────────────────────
    sub_rows = []
    for i, t in enumerate(topics_raw, 1):
        for sid in topic_subtopics[i]:
            if sid in subtopics:
                label = f"<span style='color:#94a3b8;'>{sid}</span> {subtopics[sid]['desc'][:55]}"
                sub_rows.append((label, sid, i))

    sub_row_labels = [r[0] for r in sub_rows]
    sub_row_ids    = [r[1] for r in sub_rows]

    def sub_hit(sid, cid):
        return sub_concept_hits.get(sid, {}).get(cid, 0)
    def sub_total_fn(sid):
        return sub_total_n.get(sid, 0)

    # Group header: topic blocks
    sub_col_headers = []
    prev_t = None
    t_count = 0
    for _, _, tnum in sub_rows:
        if tnum != prev_t:
            if prev_t is not None:
                sub_col_headers.append((f"T{prev_t}", t_count))
            prev_t, t_count = tnum, 1
        else:
            t_count += 1
    if prev_t is not None:
        sub_col_headers.append((f"T{prev_t}", t_count))

    # Reuse concept col grouping from topic heatmap
    subtopic_heatmap = heatmap_table(
        sub_row_labels, sub_row_ids,
        sub_hit, sub_total_fn,
        col_group_headers, "subtopic-table"
    )

    # ── Reverse: concept → topic breakdown (bar chart style table) ───────────
    reverse_rows = ""
    concept_topic_pcts = []
    for c in concepts:
        cid = c['id']
        total_pos = sum(topic_concept_hits.get(t, {}).get(cid, 0) for t in range(1, 11))
        tpcts = []
        for tnum in range(1, 11):
            hits = topic_concept_hits.get(tnum, {}).get(cid, 0)
            pct = round(hits / total_pos * 100) if total_pos else 0
            tpcts.append(pct)
        concept_topic_pcts.append(tpcts)

    for j, c in enumerate(concepts):
        tpcts = concept_topic_pcts[j]
        bars = ""
        for t_i, pct in enumerate(tpcts, 1):
            bg = pct_color(pct * 1.2)  # slightly amplified for visibility
            fg = text_color(pct * 1.2)
            bars += (
                f'<td style="background:{bg};color:{fg};text-align:center;'
                f'font-size:10px;padding:3px 4px;border:1px solid #e2e8f0;min-width:36px;">'
                f'{"" if pct < 2 else str(pct)+"%"}</td>'
            )
        bg_row = "#f8fafc" if j % 2 == 0 else "#ffffff"
        reverse_rows += (
            f'<tr style="background:{bg_row};">'
            f'<td style="padding:4px 8px;font-size:11px;font-weight:500;border:1px solid #e2e8f0;">{c["name"]}</td>'
            f'{bars}</tr>'
        )

    topic_headers = "".join(
        f'<th style="background:#1e40af;color:#fff;padding:5px 4px;font-size:11px;'
        f'white-space:nowrap;border:1px solid #93c5fd;min-width:36px;">T{i}<br>'
        f'<span style="font-weight:400;font-size:9px;">{t["topic"][:12]}</span></th>'
        for i, t in enumerate(topics_raw, 1)
    )

    reverse_table = f"""
    <div style="overflow-x:auto; margin-bottom:32px;">
      <table style="border-collapse:collapse; font-family:-apple-system,sans-serif;">
        <thead>
          <tr>
            <th style="background:#334155;color:#fff;padding:6px 8px;font-size:11px;border:1px solid #cbd5e1;">Concept</th>
            {topic_headers}
          </tr>
        </thead>
        <tbody>{reverse_rows}</tbody>
      </table>
    </div>"""

    # ── Full HTML ─────────────────────────────────────────────────────────────
    legend = "".join(
        f'<span style="display:inline-block;width:28px;height:14px;background:{pct_color(p)};'
        f'border:1px solid #cbd5e1;vertical-align:middle;margin-right:2px;"></span>{p}% '
        for p in [0, 20, 40, 60, 80, 100]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Concept ↔ Topic Map</title>
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif; background:#f8fafc; color:#1e293b; padding:24px 20px; }}
h1 {{ font-size:20px; font-weight:700; margin-bottom:6px; }}
h2 {{ font-size:15px; font-weight:600; margin:24px 0 10px; color:#1e40af; border-left:4px solid #3b82f6; padding-left:10px; }}
p.note {{ font-size:12px; color:#64748b; margin-bottom:12px; }}
.tab-bar {{ display:flex; gap:8px; margin-bottom:20px; }}
.tab {{ padding:7px 16px; font-size:13px; border:1px solid #cbd5e1; border-radius:6px; background:white; cursor:pointer; }}
.tab.active {{ background:#1e40af; color:white; border-color:#1e40af; }}
.panel {{ display:none; }}
.panel.active {{ display:block; }}
</style>
</head>
<body>
<h1>Concept ↔ Topic / Subtopic Map</h1>
<p class="note">Cell values = % of that row's memories scoring ≥ 0.5 for that concept (topic→concept views) or % of concept's positive memories from that topic (concept→topic view). Hover for counts.</p>
<div style="margin-bottom:16px;">Legend: {legend}</div>

<div class="tab-bar">
  <button class="tab active" onclick="show('topics')">Topics → Concepts</button>
  <button class="tab" onclick="show('subtopics')">Subtopics → Concepts</button>
  <button class="tab" onclick="show('reverse')">Concepts → Topics</button>
</div>

<div id="topics" class="panel active">
  <h2>How much does each concept appear in each interview topic?</h2>
  <p class="note">Row = interview topic. Column = induced concept (sorted by dominant topic). Cell = % of that topic's memories matching the concept.</p>
  {topic_heatmap}
</div>

<div id="subtopics" class="panel">
  <h2>How much does each concept appear in each subtopic?</h2>
  <p class="note">Row = interview subtopic. Column = induced concept. Cell = % of that subtopic's memories matching the concept.</p>
  {subtopic_heatmap}
</div>

<div id="reverse" class="panel">
  <h2>Where does each concept's signal come from?</h2>
  <p class="note">Row = induced concept. Column = interview topic. Cell = % of concept's positive memories that came from that topic.</p>
  {reverse_table}
</div>

<script>
function show(id) {{
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""
    return html


def main():
    print("Loading data...")
    (concepts, topics_raw, subtopics, topic_subtopics,
     sub_concept_hits, topic_concept_hits, sub_total_n, topic_total_n) = load_data()

    print(f"  {len(concepts)} concepts, {len(topics_raw)} topics, {len(subtopics)} defined subtopics")

    print("Building HTML...")
    html = build_html(concepts, topics_raw, subtopics, topic_subtopics,
                      sub_concept_hits, topic_concept_hits, sub_total_n, topic_total_n)

    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"  Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()