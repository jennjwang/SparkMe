"""
Generate an interactive HTML visualizer from a saved pipeline state.

Shows the 2-level cluster tree. Toggle participants on/off to see how
cluster memberships change live — which clusters empty out, which shrink,
and which tasks each participant contributes.

Usage:
    python analysis/task_clustering/visualize.py \
        --state analysis/task_clustering/output/cirs_clusters.json \
        --output analysis/task_clustering/output/viz.html
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path

PARTICIPANT_COLORS = [
    "#4C6EF5", "#FA5252", "#40C057", "#FAB005",
    "#BE4BDB", "#15AABF", "#FD7E14", "#20C997",
    "#E64980", "#74C0FC",
]


def build_data(state: dict) -> dict:
    """Extract and reshape state into what the HTML needs."""
    clusters = state["clusters"]
    items = state["items"]

    # item_id → cluster_id
    item_to_cluster: dict[str, str] = {}
    for cid, c in clusters.items():
        for mid in c["members"]:
            item_to_cluster[mid] = cid

    participants = sorted(set(
        item["source"] for item in items.values() if item.get("source")
    ))

    js_items = [
        {
            "id": item["id"],
            "text": item["text"],
            "source": item["source"] or "",
            "cluster_id": item_to_cluster.get(item["id"], ""),
        }
        for item in items.values()
    ]

    roots = [c for c in clusters.values() if c["parent_id"] is None]
    roots.sort(key=lambda c: -c["weight"])

    js_clusters = []
    for c in clusters.values():
        js_clusters.append({
            "id": c["id"],
            "leader": c["leader"],
            "level": c["level"],
            "parent_id": c["parent_id"],
            "children": c["children"],
            "anchored": c.get("anchored", False),
            "weight": round(c["weight"], 2),
        })

    orphaned = [it for it in js_items if not it["cluster_id"]]

    return {
        "participants": participants,
        "items": js_items,
        "clusters": js_clusters,
        "root_ids": [r["id"] for r in roots],
        "orphaned": orphaned,
    }


def generate_html(data: dict) -> str:
    colors_js = json.dumps(PARTICIPANT_COLORS)
    data_js = json.dumps(data, ensure_ascii=False, indent=2)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Task Cluster Visualizer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f0f2f5; color: #1a1a2e; min-height: 100vh; }}

  /* ── Header ── */
  .header {{ background: #fff; border-bottom: 1px solid #e0e0e0;
             padding: 16px 24px; position: sticky; top: 0; z-index: 100;
             box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
  .header h1 {{ font-size: 18px; font-weight: 700; margin-bottom: 12px; }}
  .stats {{ font-size: 13px; color: #666; margin-bottom: 12px; }}

  /* ── Participant chips ── */
  .chips {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 5px 12px;
           border-radius: 20px; font-size: 13px; font-weight: 500; cursor: pointer;
           border: 2px solid transparent; transition: all .15s; user-select: none; }}
  .chip.active   {{ color: #fff; }}
  .chip.inactive {{ background: #f5f5f5 !important; color: #999; border-color: #ddd; }}
  .chip .dot {{ width: 8px; height: 8px; border-radius: 50%; background: currentColor; }}

  /* ── Tree ── */
  .tree {{ padding: 24px; display: flex; flex-wrap: wrap; gap: 20px;
           align-items: flex-start; }}

  /* ── Root card ── */
  .root-card {{ background: #fff; border-radius: 12px; width: 340px;
                box-shadow: 0 2px 12px rgba(0,0,0,.08);
                border: 2px solid transparent; transition: all .2s; overflow: hidden; }}
  .root-card.new-cluster {{ border-color: #40C057; }}
  .new-badge {{ display: inline-block; font-size: 10px; font-weight: 700; color: #fff;
                background: #40C057; border-radius: 4px; padding: 1px 5px;
                letter-spacing: .3px; margin-left: 6px; vertical-align: middle; }}
  .root-card.empty {{ opacity: .45; border-color: #e0e0e0; }}
  .root-card.newly-empty {{ animation: flash-red .4s ease; }}
  .root-card.newly-filled {{ animation: flash-green .4s ease; }}

  .root-header {{ padding: 14px 16px 10px; cursor: pointer; }}
  .root-header:hover {{ background: #fafafa; }}
  .root-label {{ font-size: 13px; font-weight: 600; text-transform: uppercase;
                  letter-spacing: .5px; color: #888; margin-bottom: 4px; }}
  .root-leader {{ font-size: 15px; font-weight: 600; line-height: 1.35;
                  margin-bottom: 8px; color: #1a1a2e; }}
  .root-meta {{ display: flex; align-items: center; gap: 10px; font-size: 12px; color: #888; }}
  .count-badge {{ background: #f0f0f0; border-radius: 10px; padding: 2px 8px;
                  font-weight: 600; color: #444; transition: all .2s; }}
  .count-badge.changed {{ background: #ffe8b0; color: #b07800; }}

  /* participant bar */
  .pbar {{ height: 6px; border-radius: 3px; overflow: hidden;
           display: flex; margin-top: 8px; background: #f0f0f0; }}
  .pbar-seg {{ height: 100%; transition: width .25s; }}

  /* ── Direct items (items assigned to root, not a child) ── */
  .direct-section {{ padding: 0 16px 4px; }}
  .direct-section .section-label {{ font-size: 11px; font-weight: 600; color: #aaa;
                                     text-transform: uppercase; letter-spacing: .4px;
                                     padding: 6px 0 4px; }}

  /* ── Children ── */
  .children {{ padding: 8px 12px 12px; display: flex; flex-direction: column; gap: 8px; }}
  .child-card {{ border-radius: 8px; border: 1.5px solid #eee;
                 overflow: hidden; transition: all .2s; }}
  .child-card.empty {{ opacity: .4; }}
  .child-header {{ padding: 10px 12px 6px; cursor: pointer; }}
  .child-header:hover {{ background: #fafafa; }}
  .child-leader {{ font-size: 13px; font-weight: 600; line-height: 1.3; margin-bottom: 4px; }}
  .child-meta {{ font-size: 12px; color: #888; display: flex; align-items: center; gap: 8px; }}

  /* ── Task items list ── */
  .items-list {{ border-top: 1px solid #f0f0f0; max-height: 0; overflow: hidden;
                 transition: max-height .25s ease; }}
  .items-list.open {{ max-height: 600px; overflow-y: auto; }}
  .task-item {{ padding: 8px 14px; border-bottom: 1px solid #f8f8f8;
                font-size: 12px; line-height: 1.45; display: flex; gap: 8px;
                align-items: flex-start; transition: opacity .2s; }}
  .task-item.hidden {{ display: none; }}
  .task-item .src-dot {{ width: 8px; height: 8px; border-radius: 50%;
                          flex-shrink: 0; margin-top: 4px; }}
  .task-body {{ display: flex; flex-direction: column; gap: 2px; }}
  .task-text {{ color: #333; }}
  .task-src  {{ font-size: 11px; color: #aaa; }}

  /* ── Empty state ── */
  .empty-msg {{ font-size: 12px; color: #bbb; padding: 8px 14px; font-style: italic; }}

  @keyframes flash-red   {{ 0%,100%{{border-color:transparent}} 50%{{border-color:#fa5252}} }}
  @keyframes flash-green {{ 0%,100%{{border-color:transparent}} 50%{{border-color:#40c057}} }}
</style>
</head>
<body>

<div class="header">
  <h1>Task Cluster Visualizer</h1>
  <div class="stats" id="stats"></div>
  <div class="chips" id="chips"></div>
</div>
<div class="tree" id="tree"></div>

<script>
const COLORS = {colors_js};
const DATA   = {data_js};

/* ── Build lookup tables ── */
const clusterMap = {{}};
DATA.clusters.forEach(c => clusterMap[c.id] = c);

const itemsByCluster = {{}};
DATA.items.forEach(item => {{
  if (!itemsByCluster[item.cluster_id]) itemsByCluster[item.cluster_id] = [];
  itemsByCluster[item.cluster_id].push(item);
}});

const orphaned = DATA.orphaned || [];

const participantColor = {{}};
DATA.participants.forEach((p, i) => participantColor[p] = COLORS[i % COLORS.length]);

/* ── State ── */
const active = new Set(DATA.participants);

/* ── Render chips ── */
function renderChips() {{
  const el = document.getElementById('chips');
  el.innerHTML = '';
  DATA.participants.forEach(p => {{
    const chip = document.createElement('label');
    chip.className = 'chip ' + (active.has(p) ? 'active' : 'inactive');
    chip.style.background = active.has(p) ? participantColor[p] : '';
    chip.innerHTML = `<span class="dot" style="background:${{active.has(p)?'#fff':participantColor[p]}}"></span>${{p}}`;
    chip.onclick = () => toggleParticipant(p);
    el.appendChild(chip);
  }});
}}

/* ── Toggle participant ── */
function toggleParticipant(p) {{
  const wasBefore = snapshotCounts();
  if (active.has(p)) active.delete(p); else active.add(p);
  renderChips();
  updateTree(wasBefore);
  updateStats();
}}

function snapshotCounts() {{
  const snap = {{}};
  DATA.clusters.forEach(c => {{
    snap[c.id] = countActive(c.id);
  }});
  return snap;
}}

function countActive(clusterId) {{
  return (itemsByCluster[clusterId] || []).filter(i => active.has(i.source)).length;
}}

/* ── Build tree DOM ── */
function buildTree() {{
  const tree = document.getElementById('tree');
  tree.innerHTML = '';
  const hasTaxonomy = DATA.clusters.some(c => c.anchored);
  DATA.root_ids.forEach(rid => {{
    tree.appendChild(buildRoot(rid, hasTaxonomy));
  }});
  if (orphaned.length > 0) {{
    tree.appendChild(buildOrphanedCard());
  }}
}}

function buildOrphanedCard() {{
  const card = document.createElement('div');
  card.className = 'root-card';
  card.id = 'card-orphaned';
  card.style.borderColor = '#e0e0e0';

  const header = document.createElement('div');
  header.className = 'root-header';
  header.innerHTML = `
    <div class="root-label" style="color:#c0392b">Pruned / Uncategorized</div>
    <div class="root-leader" style="color:#888;font-style:italic">Tasks whose clusters were removed by fading</div>
    <div class="root-meta">
      <span class="count-badge" id="badge-orphaned">${{orphaned.length}} tasks</span>
    </div>
  `;
  const list = document.createElement('div');
  list.className = 'items-list';
  orphaned.forEach(item => list.appendChild(buildTaskItem(item)));
  header.onclick = () => list.classList.toggle('open');
  card.appendChild(header);
  card.appendChild(list);
  return card;
}}

function buildRoot(rootId, hasTaxonomy) {{
  const c = clusterMap[rootId];
  const children = (c.children || []).filter(cid => clusterMap[cid]);
  const isNew = hasTaxonomy && !c.anchored;

  const card = document.createElement('div');
  card.className = 'root-card' + (isNew ? ' new-cluster' : '');
  card.id = `card-${{rootId}}`;

  /* header */
  const header = document.createElement('div');
  header.className = 'root-header';
  const newBadge = isNew ? '<span class="new-badge">NEW</span>' : '';
  header.innerHTML = `
    <div class="root-label">Category${{newBadge}}</div>
    <div class="root-leader">${{c.leader}}</div>
    <div class="root-meta">
      <span class="count-badge" id="badge-${{rootId}}">0 tasks</span>
      <span id="pmeta-${{rootId}}"></span>
    </div>
    <div class="pbar" id="pbar-${{rootId}}"></div>
  `;
  header.onclick = () => {{
    const dl = card.querySelector('.direct-items-list');
    if (dl) dl.classList.toggle('open');
  }};
  card.appendChild(header);

  /* direct items (assigned to root, not a child) */
  const directItems = (itemsByCluster[rootId] || []);
  if (directItems.length > 0) {{
    const section = document.createElement('div');
    section.className = 'direct-section';
    section.innerHTML = `<div class="section-label">General (unassigned to sub-type)</div>`;
    const list = document.createElement('div');
    list.className = 'items-list direct-items-list';
    directItems.forEach(item => {{
      list.appendChild(buildTaskItem(item));
    }});
    section.appendChild(list);
    card.appendChild(section);
  }}

  /* children */
  if (children.length > 0) {{
    const childContainer = document.createElement('div');
    childContainer.className = 'children';
    children.forEach(cid => childContainer.appendChild(buildChild(cid)));
    card.appendChild(childContainer);
  }}

  return card;
}}

function buildChild(childId) {{
  const c = clusterMap[childId];
  const card = document.createElement('div');
  card.className = 'child-card';
  card.id = `card-${{childId}}`;

  const header = document.createElement('div');
  header.className = 'child-header';
  header.innerHTML = `
    <div class="child-leader">${{c.leader}}</div>
    <div class="child-meta">
      <span class="count-badge" id="badge-${{childId}}">0 tasks</span>
      <div class="pbar" id="pbar-${{childId}}" style="width:100%;height:5px;border-radius:3px;overflow:hidden;display:flex;background:#f0f0f0;"></div>
    </div>
  `;
  const list = document.createElement('div');
  list.className = 'items-list';
  (itemsByCluster[childId] || []).forEach(item => list.appendChild(buildTaskItem(item)));
  if (!itemsByCluster[childId]?.length) {{
    list.innerHTML = '<div class="empty-msg">No tasks assigned</div>';
  }}

  header.onclick = () => list.classList.toggle('open');
  card.appendChild(header);
  card.appendChild(list);
  return card;
}}

function buildTaskItem(item) {{
  const el = document.createElement('div');
  el.className = 'task-item';
  el.id = `item-${{item.id}}`;
  el.innerHTML = `
    <span class="src-dot" style="background:${{participantColor[item.source] || '#ccc'}}"></span>
    <span class="task-body">
      <span class="task-text">${{item.text}}</span>
      <span class="task-src">${{item.source}}</span>
    </span>
  `;
  return el;
}}

/* ── Update tree after toggle ── */
function updateTree(prevCounts) {{
  DATA.items.forEach(item => {{
    const el = document.getElementById(`item-${{item.id}}`);
    if (!el) return;
    el.classList.toggle('hidden', !active.has(item.source));
  }});

  DATA.clusters.forEach(c => {{
    const count = countActive(c.id);
    const prev  = prevCounts ? (prevCounts[c.id] || 0) : -1;

    /* badge */
    const badge = document.getElementById(`badge-${{c.id}}`);
    if (badge) {{
      badge.textContent = count === 1 ? '1 task' : `${{count}} tasks`;
      badge.classList.toggle('changed', prev !== -1 && count !== prev);
    }}

    /* participant bar */
    updatePbar(c.id);

    /* card fade */
    const card = document.getElementById(`card-${{c.id}}`);
    if (!card) return;
    const wasEmpty = prev === 0;
    const nowEmpty = count === 0;
    card.classList.toggle('empty', nowEmpty);
    if (!wasEmpty && nowEmpty && prev !== -1) card.classList.add('newly-empty');
    if (wasEmpty  && !nowEmpty && prev !== -1) card.classList.add('newly-filled');
    setTimeout(() => card.classList.remove('newly-empty','newly-filled'), 500);
  }});
}}

function updatePbar(clusterId) {{
  const pbar = document.getElementById(`pbar-${{clusterId}}`);
  if (!pbar) return;
  const items = itemsByCluster[clusterId] || [];
  const total = items.filter(i => active.has(i.source)).length;
  if (total === 0) {{ pbar.innerHTML = ''; return; }}

  const counts = {{}};
  items.forEach(i => {{ if (active.has(i.source)) counts[i.source] = (counts[i.source]||0)+1; }});
  pbar.innerHTML = Object.entries(counts).map(([src, n]) =>
    `<div class="pbar-seg" style="width:${{(n/total*100).toFixed(1)}}%;background:${{participantColor[src]||'#ccc'}}"></div>`
  ).join('');
}}

function updateStats() {{
  const totalActive = DATA.items.filter(i => active.has(i.source)).length;
  const activePeople = active.size;
  document.getElementById('stats').textContent =
    `${{activePeople}} of ${{DATA.participants.length}} participants · ${{totalActive}} of ${{DATA.items.length}} tasks shown`;
}}

/* ── Init ── */
renderChips();
buildTree();
updateTree(null);
updateStats();
</script>
</body>
</html>"""


def generate_trace_html(snapshots: list[dict]) -> str:
    participants = sorted(set(s["task_source"] for s in snapshots if s.get("task_source")))
    colors_js = json.dumps(PARTICIPANT_COLORS)
    trace_js = json.dumps(snapshots, ensure_ascii=False)
    participants_js = json.dumps(participants)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Task Cluster Trace</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          background: #f0f2f5; color: #1a1a2e; min-height: 100vh; }}

  /* ── Header ── */
  .header {{ background: #fff; border-bottom: 1px solid #e0e0e0;
             padding: 16px 24px; position: sticky; top: 0; z-index: 100;
             box-shadow: 0 2px 8px rgba(0,0,0,.06); }}
  .header h1 {{ font-size: 18px; font-weight: 700; margin-bottom: 14px; }}

  /* ── Timeline controls ── */
  .timeline {{ display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }}
  .nav-btn {{ background: #f0f0f0; border: none; border-radius: 6px; padding: 5px 12px;
              font-size: 14px; cursor: pointer; font-weight: 600; }}
  .nav-btn:hover {{ background: #e0e0e0; }}
  .nav-btn:disabled {{ opacity: .35; cursor: default; }}
  #slider {{ flex: 1; accent-color: #4C6EF5; }}
  .step-label {{ font-size: 13px; color: #666; white-space: nowrap; }}

  /* ── Current task box ── */
  .task-box {{ border-radius: 8px; padding: 10px 14px; font-size: 13px;
               line-height: 1.5; border-left: 4px solid #4C6EF5;
               background: #f5f7ff; display: flex; gap: 10px; align-items: flex-start; }}
  .task-box .dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 3px; }}
  .task-box .task-info {{ flex: 1; }}
  .task-box .task-text {{ color: #1a1a2e; font-weight: 500; }}
  .task-box .task-meta {{ font-size: 11px; color: #888; margin-top: 3px; }}
  .task-box .assigned {{ font-size: 11px; color: #4C6EF5; margin-top: 3px; font-weight: 500; }}
  .task-box .reasoning {{ font-size: 11px; color: #666; margin-top: 4px; font-style: italic; border-top: 1px solid #e0e8ff; padding-top: 4px; }}

  /* ── Tree ── */
  .tree {{ padding: 24px; display: flex; flex-wrap: wrap; gap: 20px;
           align-items: flex-start; }}

  /* ── Root card ── */
  .root-card {{ background: #fff; border-radius: 12px; width: 340px;
                box-shadow: 0 2px 12px rgba(0,0,0,.08);
                border: 2px solid transparent; transition: border-color .2s; overflow: hidden; }}
  .root-card.highlight {{ border-color: #4C6EF5; }}
  .root-card.new-cluster {{ border-color: #40C057; }}
  .root-card.new-cluster.highlight {{ border-color: #4C6EF5; }}
  .new-badge {{ display: inline-block; font-size: 10px; font-weight: 700; color: #fff;
                background: #40C057; border-radius: 4px; padding: 1px 5px;
                letter-spacing: .3px; margin-left: 6px; vertical-align: middle; }}

  .root-header {{ padding: 14px 16px 10px; cursor: pointer; user-select: none; }}
  .root-header:hover {{ background: #fafafa; }}
  .chevron {{ float: right; font-size: 12px; color: #aaa; transition: transform .2s; margin-top: 2px; }}
  .open > .chevron, .expanded .chevron {{ transform: rotate(180deg); }}
  .children.collapsed {{ display: none; }}
  .task-list-empty {{ padding: 8px 14px; font-size: 11px; color: #bbb; font-style: italic;
                      border-top: 1px solid #f0f0f0; }}
  .root-label {{ font-size: 11px; font-weight: 600; text-transform: uppercase;
                  letter-spacing: .5px; color: #888; margin-bottom: 4px; }}
  .root-leader {{ font-size: 14px; font-weight: 600; line-height: 1.35;
                  margin-bottom: 8px; color: #1a1a2e; }}
  .root-meta {{ display: flex; align-items: center; gap: 10px; font-size: 12px; color: #888; }}
  .count-badge {{ background: #f0f0f0; border-radius: 10px; padding: 2px 8px;
                  font-weight: 600; color: #444; }}
  .pbar {{ height: 6px; border-radius: 3px; overflow: hidden;
           display: flex; margin-top: 8px; background: #f0f0f0; }}
  .pbar-seg {{ height: 100%; }}

  /* ── Children ── */
  .children {{ padding: 8px 12px 12px; display: flex; flex-direction: column; gap: 8px; }}
  .child-card {{ border-radius: 8px; border: 1.5px solid #eee; overflow: hidden;
                 transition: border-color .2s; }}
  .child-card.highlight {{ border-color: #4C6EF5; }}
  .child-header {{ padding: 10px 12px 8px; cursor: pointer; user-select: none; }}
  .child-header:hover {{ background: #fafafa; }}
  .child-leader {{ font-size: 13px; font-weight: 600; line-height: 1.3; margin-bottom: 4px; }}
  .child-meta {{ font-size: 12px; color: #888; display: flex; align-items: center; gap: 8px; }}

  /* ── Expandable task list ── */
  .task-list {{ max-height: 0; overflow: hidden; transition: max-height .25s ease;
                border-top: 1px solid #f0f0f0; }}
  .task-list.open {{ max-height: 600px; overflow-y: auto; }}
  .task-row {{ padding: 7px 14px; border-bottom: 1px solid #f8f8f8; font-size: 12px;
               display: flex; gap: 8px; align-items: flex-start; }}
  .task-row:last-child {{ border-bottom: none; }}
  .task-row .tdot {{ width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }}
  .task-row .tbody {{ display: flex; flex-direction: column; gap: 1px; }}
  .task-row .ttext {{ color: #333; line-height: 1.4; }}
  .task-row .tsrc {{ font-size: 11px; color: #aaa; }}
</style>
</head>
<body>

<div class="header">
  <h1>Task Cluster Trace</h1>
  <div class="timeline">
    <button class="nav-btn" id="btn-prev">&#8592;</button>
    <input type="range" id="slider" min="1" value="1">
    <button class="nav-btn" id="btn-next">&#8594;</button>
    <span class="step-label" id="step-label"></span>
    <span class="step-label" id="cluster-count" style="color:#4C6EF5"></span>
  </div>
  <div class="task-box" id="task-box">
    <span class="dot" id="task-dot"></span>
    <span class="task-info">
      <div class="task-text" id="task-text"></div>
      <div class="task-meta" id="task-meta"></div>
      <div class="assigned" id="task-assigned"></div>
      <div class="reasoning" id="task-reasoning" style="display:none"></div>
    </span>
  </div>
</div>
<div class="tree" id="tree"></div>

<script>
const COLORS = {colors_js};
const TRACE  = {trace_js};
const PARTICIPANTS = {participants_js};

const participantColor = {{}};
PARTICIPANTS.forEach((p, i) => participantColor[p] = COLORS[i % COLORS.length]);

const slider  = document.getElementById('slider');
const btnPrev = document.getElementById('btn-prev');
const btnNext = document.getElementById('btn-next');
slider.max = TRACE.length;

function render(stepIdx) {{
  const snap = TRACE[stepIdx];
  const step = snap.step;

  /* ── Controls ── */
  slider.value = step;
  document.getElementById('step-label').textContent = `Interview ${{step}} of ${{TRACE.length}}`;
  document.getElementById('cluster-count').textContent = `· ${{snap.clusters.length}} clusters`;
  btnPrev.disabled = step <= 1;
  btnNext.disabled = step >= TRACE.length;

  /* ── Task box ── */
  const color = participantColor[snap.task_source] || '#ccc';
  document.getElementById('task-dot').style.background = color;
  document.getElementById('task-text').textContent = snap.task_text;
  document.getElementById('task-meta').textContent = snap.task_source;
  document.getElementById('task-assigned').textContent =
    `→ L${{snap.assigned_level}}: ${{snap.assigned_label}}`;
  const reasoningEl = document.getElementById('task-reasoning');
  const reasonings = snap.reasonings || [];
  if (reasonings.length) {{
    reasoningEl.textContent = reasonings.join(' · ');
    reasoningEl.style.display = '';
  }} else {{
    reasoningEl.style.display = 'none';
  }}

  /* ── Build cluster map ── */
  const clusterMap = {{}};
  snap.clusters.forEach(c => clusterMap[c.id] = c);
  const hasTaxonomy = snap.clusters.some(c => c.anchored);

  const roots = snap.clusters
    .filter(c => c.parent_id === null)
    .sort((a, b) => b.weight - a.weight);

  /* ── Tree ── */
  const tree = document.getElementById('tree');
  tree.innerHTML = '';
  roots.forEach(root => tree.appendChild(buildRoot(root, clusterMap, snap.assigned_id, hasTaxonomy)));
}}

function buildRoot(root, clusterMap, assignedId, hasTaxonomy) {{
  const isHighlight = root.id === assignedId;
  const isNew = hasTaxonomy && !root.anchored;
  const card = document.createElement('div');
  card.className = 'root-card' + (isHighlight ? ' highlight' : '') + (isNew ? ' new-cluster' : '');

  const totalDirect = root.member_count;
  const childTotal = (root.children || []).reduce((s, cid) =>
    s + (clusterMap[cid]?.member_count || 0), 0);

  const header = document.createElement('div');
  header.className = 'root-header';
  header.style.cursor = 'pointer';
  header.style.userSelect = 'none';
  const newBadge = isNew ? '<span class="new-badge">NEW</span>' : '';
  header.innerHTML = `
    <div class="root-label">Category${{newBadge}} <span class="chevron">▾</span></div>
    <div class="root-leader">${{root.leader}}</div>
    <div class="root-meta">
      <span class="count-badge">${{totalDirect + childTotal}} tasks</span>
    </div>
    ${{pbarHTML(root.sources)}}
  `;
  card.appendChild(header);

  if (root.children?.length) {{
    const childContainer = document.createElement('div');
    childContainer.className = 'children';
    root.children.forEach(cid => {{
      const child = clusterMap[cid];
      if (!child) return;
      childContainer.appendChild(buildChild(child, assignedId));
    }});
    card.appendChild(childContainer);
    header.onclick = () => {{
      childContainer.classList.toggle('collapsed');
      header.classList.toggle('expanded');
    }};
  }} else {{
    const list = buildTaskList(root.members || []);
    card.appendChild(list);
    header.onclick = () => {{
      list.classList.toggle('open');
      header.querySelector('.chevron').style.transform =
        list.classList.contains('open') ? 'rotate(180deg)' : '';
    }};
  }}

  return card;
}}

function buildChild(child, assignedId) {{
  const isHighlight = child.id === assignedId;
  const cc = document.createElement('div');
  cc.className = 'child-card' + (isHighlight ? ' highlight' : '');

  const header = document.createElement('div');
  header.className = 'child-header';
  header.innerHTML = `
    <div class="child-leader">${{child.leader}} <span class="chevron" style="float:right;font-size:11px;color:#aaa">▾</span></div>
    <div class="child-meta">
      <span class="count-badge">${{child.member_count}} tasks</span>
      ${{pbarHTML(child.sources, '100%', '5px')}}
    </div>
  `;
  cc.appendChild(header);

  const list = buildTaskList(child.members || []);
  header.onclick = () => {{
    list.classList.toggle('open');
    header.querySelector('.chevron').style.transform =
      list.classList.contains('open') ? 'rotate(180deg)' : '';
  }};
  cc.appendChild(list);
  return cc;
}}

function buildTaskList(members) {{
  const list = document.createElement('div');
  list.className = 'task-list';
  if (!members || !members.length) {{
    const empty = document.createElement('div');
    empty.className = 'task-list-empty';
    empty.textContent = 'Re-run trace.py to load task details';
    list.appendChild(empty);
    return list;
  }}
  members.forEach(m => {{
    const row = document.createElement('div');
    row.className = 'task-row';
    const color = participantColor[m.source] || '#ccc';
    row.innerHTML = `
      <span class="tdot" style="background:${{color}}"></span>
      <span class="tbody">
        <span class="ttext">${{m.text}}</span>
        <span class="tsrc">${{m.source}}</span>
      </span>`;
    list.appendChild(row);
  }});
  return list;
}}

function pbarHTML(sources, width, height) {{
  if (!sources || !Object.keys(sources).length) return '';
  const total = Object.values(sources).reduce((s, n) => s + n, 0);
  const segs = Object.entries(sources).map(([src, n]) =>
    `<div class="pbar-seg" style="width:${{(n/total*100).toFixed(1)}}%;background:${{participantColor[src]||'#ccc'}}"></div>`
  ).join('');
  const w = width ? `width:${{width}};` : '';
  const h = height ? `height:${{height}};` : '';
  return `<div class="pbar" style="${{w}}${{h}}">${{segs}}</div>`;
}}

slider.oninput  = () => render(+slider.value - 1);
btnPrev.onclick = () => {{ if (+slider.value > 1) {{ slider.value--; render(+slider.value - 1); }} }};
btnNext.onclick = () => {{ if (+slider.value < TRACE.length) {{ slider.value++; render(+slider.value - 1); }} }};

/* keyboard arrow keys */
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft'  && !btnPrev.disabled) btnPrev.click();
  if (e.key === 'ArrowRight' && !btnNext.disabled) btnNext.click();
}});

render(0);
</script>
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser(description="Generate interactive cluster visualizer")
    parser.add_argument("--state", required=True, help="Path to saved state JSON")
    parser.add_argument("--output", required=True, help="Path for output HTML file")
    parser.add_argument("--trace", default=None,
                        help="Path to trace JSON (from trace.py) — adds timeline slider")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.trace:
        with open(args.trace) as f:
            snapshots = json.load(f)
        html = generate_trace_html(snapshots)
        out.write_text(html, encoding="utf-8")
        print(f"Trace visualizer written to {out}  ({len(snapshots)} steps)")
    else:
        with open(args.state) as f:
            state = json.load(f)
        data = build_data(state)
        html = generate_html(data)
        out.write_text(html, encoding="utf-8")
        print(f"Visualizer written to {out}")
        print(f"  {len(data['participants'])} participants, "
              f"{len(data['items'])} items, "
              f"{len(data['clusters'])} clusters")


if __name__ == "__main__":
    main()
