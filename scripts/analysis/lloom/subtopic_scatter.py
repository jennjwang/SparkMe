"""
Scatter plot of memories colored by subtopic (finest granularity),
with cluster overlay (centroids + convex hulls).
Hover on cluster centroid shows: label, purity, intra-cos, n_memories, n_participants.

Run:
    .venv-analysis/bin/python scripts/analysis/subtopic_scatter.py         # agglomerative clusters
    .venv-analysis/bin/python scripts/analysis/subtopic_scatter.py --llm   # LLM clusters
"""

import sys
import json
import os
import colorsys
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.spatial import ConvexHull
import plotly.graph_objects as go
from umap import UMAP

USE_LLM = "--llm" in sys.argv

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"

if USE_LLM:
    CLUSTERS_PATH = USER_STUDY_DIR / "llm_clustering.json"
    CLUSTER_LABELS_CACHE = USER_STUDY_DIR / "llm_cluster_labels_cache.json"
    OUT_PATH = USER_STUDY_DIR / "subtopic_scatter_llm.html"
else:
    CLUSTERS_PATH = USER_STUDY_DIR / "hierarchical_clustering.json"
    CLUSTER_LABELS_CACHE = USER_STUDY_DIR / "cluster_labels_cache.json"
    OUT_PATH = USER_STUDY_DIR / "subtopic_scatter.html"

# ── Helpers ───────────────────────────────────────────────────────────────────

def wrap_text(text, width=80):
    """Wrap text at word boundaries, joining with <br> for Plotly hover."""
    import textwrap
    return "<br>".join(textwrap.wrap(text, width=width))


# ── Load ──────────────────────────────────────────────────────────────────────

def load_topics():
    with open(TOPICS_PATH) as f:
        topics = json.load(f)
    subtopics = {}
    for i, t in enumerate(topics):
        tid = str(i + 1)
        for j, desc in enumerate(t["subtopics"]):
            subtopics[f"{tid}.{j+1}"] = {
                "topic_name": t["topic"], "desc": desc,
                "topic_idx": i, "sub_idx": j
            }
    return topics, subtopics

def load_memories_and_embeddings():
    memories, embeddings = [], []
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        emb_path = USER_STUDY_DIR / pid / "memory_bank_embeddings.json"
        if not mem_path.is_file() or not emb_path.is_file():
            continue
        with open(mem_path) as f:
            mem_data = json.load(f)
        with open(emb_path) as f:
            emb_data = json.load(f)
        emb_lookup = {e["id"]: e["embedding"] for e in emb_data.get("embeddings", [])}
        for m in mem_data.get("memories", []):
            if m.get("id") in emb_lookup:
                memories.append({"pid": pid, **m})
                embeddings.append(emb_lookup[m["id"]])
    return memories, np.array(embeddings, dtype=np.float32)

def get_cluster_assignments(memories, embeddings):
    """Return (labels array, cluster_meta dict).
    Handles both hierarchical_clustering.json (key: 'fine') and
    llm_clustering.json (key: 'clusters'). Uses cached id->label if available.
    """
    from sklearn.cluster import AgglomerativeClustering

    with open(CLUSTERS_PATH) as f:
        raw = json.load(f)
    # Normalise: both formats expose per-cluster dicts keyed by string int
    cluster_meta = raw.get("fine") or raw.get("clusters") or {}

    if CLUSTER_LABELS_CACHE.exists():
        print("  Loading cluster labels from cache...")
        with open(CLUSTER_LABELS_CACHE) as f:
            id_to_label = json.load(f)
        labels = np.array([id_to_label.get(m["id"], 0) for m in memories])
        print(f"  {len(set(labels))} clusters")
        return labels, cluster_meta

    # Fallback: recompute agglomerative (only for non-LLM mode)
    print("  Computing agglomerative clusters (10D UMAP + Ward, n_clusters=56)...")
    reducer = UMAP(n_components=10, n_neighbors=15, min_dist=0.0,
                   metric="cosine", random_state=42)
    reduced = reducer.fit_transform(embeddings)
    labels = AgglomerativeClustering(n_clusters=56, linkage="ward").fit_predict(reduced)

    id_to_label = {m["id"]: int(labels[i]) for i, m in enumerate(memories)}
    with open(CLUSTER_LABELS_CACHE, "w") as f:
        json.dump(id_to_label, f)
    print(f"  Cached {len(set(labels))} cluster labels")
    return labels, cluster_meta

# ── UMAP 2D ───────────────────────────────────────────────────────────────────

def reduce_for_viz(embeddings):
    print("  Running UMAP 2D...")
    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                   metric="cosine", random_state=42)
    return reducer.fit_transform(embeddings)

# ── Colors ────────────────────────────────────────────────────────────────────

def subtopic_color(topic_idx, sub_idx, n_subs):
    h = topic_idx / 10
    l = 0.30 + (sub_idx / max(n_subs - 1, 1)) * 0.35
    r, g, b = colorsys.hls_to_rgb(h, l, 0.70)
    return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

# ── Convex hull ───────────────────────────────────────────────────────────────

def hull_coords(pts):
    """Return (x_closed, y_closed) for convex hull of pts, or None if < 3 points."""
    if len(pts) < 3:
        return None, None
    try:
        hull = ConvexHull(pts)
        verts = np.append(hull.vertices, hull.vertices[0])
        return pts[verts, 0].tolist(), pts[verts, 1].tolist()
    except Exception:
        return None, None

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading topics...")
    topics, subtopics = load_topics()

    print("Loading memories and embeddings...")
    memories, embeddings = load_memories_and_embeddings()
    print(f"  {len(memories)} memories from {len(set(m['pid'] for m in memories))} participants")

    coords = reduce_for_viz(embeddings)

    cluster_labels, fine_meta = get_cluster_assignments(memories, embeddings)

    # ── Subtopic dots ─────────────────────────────────────────────────────────

    def dominant_subtopic(m):
        links = m.get("subtopic_links", [])
        if not links:
            return None
        return max(links, key=lambda l: l.get("importance", 0))["subtopic_id"]

    dom = [dominant_subtopic(m) for m in memories]
    topic_n_subs = [len(t["subtopics"]) for t in topics]

    def get_color(sid):
        if sid is None or sid not in subtopics:
            return "lightgray"
        info = subtopics[sid]
        return subtopic_color(info["topic_idx"], info["sub_idx"], topic_n_subs[info["topic_idx"]])

    groups = defaultdict(list)
    for i, sid in enumerate(dom):
        groups[sid].append(i)

    fig = go.Figure()

    # Unlinked gray
    unlinked = groups.get(None, [])
    if unlinked:
        fig.add_trace(go.Scatter(
            x=coords[unlinked, 0], y=coords[unlinked, 1],
            mode="markers",
            marker=dict(size=4, color="lightgray", opacity=0.3),
            name="unlinked",
            text=[f"<b>{memories[i]['title']}</b><br>No subtopic link" for i in unlinked],
            hovertemplate="%{text}<extra></extra>",
            legendgroup="unlinked",
            customdata=[[int(cluster_labels[i])] for i in unlinked],
        ))

    all_sids = sorted(
        [sid for sid in groups if sid is not None and sid in subtopics],
        key=lambda s: (subtopics[s]["topic_idx"], subtopics[s]["sub_idx"])
    )

    for sid in all_sids:
        idxs = groups[sid]
        info = subtopics[sid]
        color = get_color(sid)
        topic_name = info["topic_name"]
        desc = info["desc"]
        hover = [
            f"<b>{memories[i]['title']}</b><br>"
            f"Subtopic: {sid} — {desc}<br>"
            f"Topic: {topic_name}<br>"
            f"Cluster: C{cluster_labels[i]}<br>"
            "<i>" + wrap_text(memories[i].get('text',''), 80) + "</i>"
            for i in idxs
        ]
        fig.add_trace(go.Scatter(
            x=coords[idxs, 0], y=coords[idxs, 1],
            mode="markers",
            marker=dict(size=6, color=color, opacity=0.80,
                        line=dict(width=0.4, color="white")),
            name=f"{sid}: {desc[:35]}",
            legendgroup=f"T{info['topic_idx']+1}",
            legendgrouptitle=dict(
                text=f"T{info['topic_idx']+1}: {topic_name[:28]}"
            ) if info["sub_idx"] == 0 else dict(),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
            customdata=[[int(cluster_labels[i])] for i in idxs],
        ))

    # ── Compute per-cluster stats on the fly ──────────────────────────────────
    from sklearn.metrics.pairwise import cosine_similarity as cos_sim

    def compute_cluster_stats(idxs, memories, embeddings):
        """Compute purity, intra-cos, n_participants for a set of memory indices."""
        n = len(idxs)
        n_part = len(set(memories[i]["pid"] for i in idxs))

        # Purity: fraction sharing dominant topic
        topic_counts = defaultdict(int)
        for i in idxs:
            links = memories[i].get("subtopic_links", [])
            if links:
                best = max(links, key=lambda l: l.get("importance", 0))
                tid = best["subtopic_id"].split(".")[0]
                topic_counts[tid] += 1
        purity = max(topic_counts.values()) / n if topic_counts else 0.0

        # Intra-cluster cosine (sample up to 100 for speed)
        sample = idxs if len(idxs) <= 100 else np.random.choice(idxs, 100, replace=False)
        emb_slice = embeddings[sample]
        sims = cos_sim(emb_slice)
        mask = np.triu(np.ones_like(sims, dtype=bool), k=1)
        intra_cos = float(sims[mask].mean()) if mask.sum() > 0 else 0.0

        # Emergent: purity < 0.60 and intra_cos >= 0.55
        is_emergent = purity < 0.60 and intra_cos >= 0.55

        return purity, intra_cos, n, n_part, is_emergent

    # ── Cluster overlay ───────────────────────────────────────────────────────
    unique_clusters = sorted(set(cluster_labels))

    for cid in unique_clusters:
        idxs = np.where(cluster_labels == cid)[0]
        pts = coords[idxs]

        purity, cos, n, n_part, is_emergent = compute_cluster_stats(idxs, memories, embeddings)

        sample_titles = [memories[i]["title"] for i in idxs[:3]]
        samples_str = "<br>".join(f"  · {t}" for t in sample_titles)
        label = fine_meta.get(str(cid), {}).get("label", f"C{cid}")

        hover_text = (
            f"<b>C{cid}: {label}</b>{'  ⭐ emergent' if is_emergent else ''}<br>"
            f"Purity: {purity:.2f}  |  Intra-cos: {cos:.3f}<br>"
            f"Memories: {n}  |  Participants: {n_part}<br>"
            f"<b>Samples:</b><br>{samples_str}"
        )

        # Convex hull outline
        hx, hy = hull_coords(pts)
        if hx is not None:
            hull_color = "rgba(255,80,80,0.55)" if is_emergent else "rgba(80,80,200,0.30)"
            hull_fill = "rgba(255,80,80,0.04)" if is_emergent else "rgba(80,80,200,0.03)"
            fig.add_trace(go.Scatter(
                x=hx, y=hy,
                mode="lines",
                line=dict(color=hull_color, width=1.2, dash="dot" if not is_emergent else "solid"),
                fill="toself",
                fillcolor=hull_fill,
                showlegend=False,
                hoverinfo="skip",
                legendgroup="clusters",
            ))

        # Centroid marker
        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
        marker_color = "rgba(220,40,40,0.9)" if is_emergent else "rgba(60,60,180,0.75)"
        fig.add_trace(go.Scatter(
            x=[cx], y=[cy],
            mode="markers+text",
            marker=dict(
                symbol="diamond" if is_emergent else "circle",
                size=10 if is_emergent else 8,
                color=marker_color,
                line=dict(width=1.5, color="white"),
            ),
            text=[f"C{cid}: {label[:25]}"],
            textposition="top center",
            textfont=dict(size=7, color=marker_color),
            showlegend=False,
            name=f"C{cid}",
            customdata=[[purity, cos, n, n_part, is_emergent]],
            hovertext=[hover_text],
            hovertemplate="%{hovertext}<extra></extra>",
            legendgroup="clusters",
        ))

    # ── Legend entries for cluster types ──────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="circle", size=8, color="rgba(60,60,180,0.75)"),
        name="Cluster centroid", legendgroup="clusters",
        legendgrouptitle=dict(text="Clusters"),
    ))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers",
        marker=dict(symbol="diamond", size=10, color="rgba(220,40,40,0.9)"),
        name="Emergent cluster ⭐", legendgroup="clusters",
    ))

    # ── Topic filter buttons ───────────────────────────────────────────────────
    # Traces: [unlinked, *subtopic_traces, *hull_traces, *centroid_traces, legend_entries]
    # For buttons, we only toggle the subtopic dot traces (indices 0..n_subtopic_traces-1)
    # Cluster traces are always visible

    trace_topic_idx = [None]  # unlinked
    for sid in all_sids:
        trace_topic_idx.append(subtopics[sid]["topic_idx"])

    n_dot_traces = len(trace_topic_idx)
    n_total_traces = len(fig.data)
    # Cluster traces start at n_dot_traces and run to end
    cluster_always_true = [True] * (n_total_traces - n_dot_traces)

    def vis_for_topic(selected):
        dot_vis = [
            True if ti is None or ti in selected else False
            for ti in trace_topic_idx
        ]
        return dot_vis + cluster_always_true

    buttons = [
        dict(label="All", method="restyle",
             args=[{"visible": [True] * n_total_traces}]),
        dict(label="None (clusters only)", method="restyle",
             args=[{"visible": [False] * n_dot_traces + cluster_always_true}]),
    ]
    for ti, t in enumerate(topics):
        buttons.append(dict(
            label=f"T{ti+1}: {t['topic'][:20]}",
            method="restyle",
            args=[{"visible": vis_for_topic({ti})}],
        ))

    fig.update_layout(
        title=dict(
            text=f"Memories by Subtopic + {'LLM (GPT-4o)' if USE_LLM else 'Agglomerative'} Cluster Overlay<br>"
                 "<sup>Hue = topic · Shade = subtopic · "
                 "Blue outline = cluster · Red diamond = emergent cluster · "
                 "Hover centroid for purity & cosine</sup>",
            font=dict(size=14),
        ),
        height=900, width=1350,
        template="plotly_white",
        hovermode="closest",
        showlegend=False,
        margin=dict(l=10, r=10, t=100, b=10),
        updatemenus=[dict(
            type="buttons",
            direction="right",
            x=0.0, y=1.12,
            xanchor="left",
            yanchor="top",
            pad=dict(r=4, t=4),
            buttons=buttons,
            bgcolor="white",
            bordercolor="#ccc",
            font=dict(size=10),
        )],
        xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
    )

    # ── Build data for dropdown and sidebar legend ────────────────────────────
    cluster_options = []
    for cid in unique_clusters:
        meta = fine_meta.get(str(cid), {})
        label = meta.get("label", f"C{cid}")
        emergent = meta.get("is_emergent", False)
        n = meta.get("n", 0)
        star = " ⭐" if emergent else ""
        cluster_options.append({"cid": int(cid), "label": f"C{cid}: {label} (n={n}){star}"})
    cluster_options_json = json.dumps(cluster_options)

    # Subtopic sidebar entries: [{sid, color, topic_name, desc, topic_idx}]
    sidebar_entries = []
    for sid in all_sids:
        info = subtopics[sid]
        color = get_color(sid)
        sidebar_entries.append({
            "sid": sid,
            "color": color,
            "topic_name": info["topic_name"],
            "topic_idx": info["topic_idx"],
            "desc": info["desc"],
        })
    # Group by topic for section headers
    sidebar_json = json.dumps(sidebar_entries)

    # ── Write HTML with dropdown + click-to-filter JS ─────────────────────────
    html = fig.to_html(include_plotlyjs="cdn", full_html=True)

    inject = f"""
<style>
  body {{ margin: 0; display: flex; font-family: sans-serif; }}
  #main-plot {{ flex: 1; min-width: 0; }}
  #sidebar {{
    width: 240px;
    min-width: 240px;
    height: 100vh;
    overflow-y: auto;
    border-left: 1px solid #ddd;
    background: #fafafa;
    padding: 10px 8px;
    box-sizing: border-box;
    font-size: 11px;
  }}
  #sidebar h3 {{ margin: 0 0 6px; font-size: 12px; color: #333; }}
  .topic-section {{ margin-bottom: 8px; }}
  .topic-header {{
    font-weight: bold; font-size: 11px; color: #444;
    margin: 6px 0 2px; border-bottom: 1px solid #ddd; padding-bottom: 2px;
  }}
  .legend-row {{
    display: flex; align-items: flex-start; gap: 5px;
    margin: 2px 0; cursor: pointer; padding: 1px 3px; border-radius: 3px;
  }}
  .legend-row:hover {{ background: #eee; }}
  .legend-swatch {{
    width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; margin-top: 2px;
  }}
  .legend-label {{ color: #333; line-height: 1.3; white-space: normal; word-break: break-word; }}
  #cluster-controls {{
    position: fixed;
    top: 10px;
    right: 258px;
    z-index: 9999;
    background: white;
    border: 1px solid #ccc;
    border-radius: 6px;
    padding: 6px 10px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    font-size: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
  }}
  #cluster-select {{
    font-size: 12px; padding: 3px 5px;
    border: 1px solid #bbb; border-radius: 4px; min-width: 240px;
  }}
  #cluster-reset {{
    font-size: 11px; padding: 3px 8px;
    border: 1px solid #bbb; border-radius: 4px;
    cursor: pointer; background: #f5f5f5;
  }}
  #cluster-reset:hover {{ background: #e0e0e0; }}
</style>

<div id="cluster-controls">
  <label for="cluster-select"><b>Cluster:</b></label>
  <select id="cluster-select">
    <option value="">— show all —</option>
  </select>
  <button id="cluster-reset">Reset</button>
</div>

<div id="sidebar">
  <h3>Subtopics</h3>
  <div id="legend-content"></div>
</div>

<script>
(function() {{
  // ── Move plot into #main-plot wrapper ──
  var plotDiv = document.querySelector('.plotly-graph-div');
  var wrapper = document.getElementById('main-plot');
  if (!wrapper) {{
    wrapper = document.createElement('div');
    wrapper.id = 'main-plot';
    plotDiv.parentNode.insertBefore(wrapper, plotDiv);
    wrapper.appendChild(plotDiv);
  }}

  // ── Build sidebar legend ──
  var sidebarEntries = {sidebar_json};
  var legendContent = document.getElementById('legend-content');
  var currentTopic = null;
  sidebarEntries.forEach(function(e) {{
    if (e.topic_idx !== currentTopic) {{
      currentTopic = e.topic_idx;
      var hdr = document.createElement('div');
      hdr.className = 'topic-header';
      hdr.textContent = 'T' + (e.topic_idx + 1) + ': ' + e.topic_name;
      legendContent.appendChild(hdr);
    }}
    var row = document.createElement('div');
    row.className = 'legend-row';
    row.dataset.sid = e.sid;
    row.innerHTML =
      '<div class="legend-swatch" style="background:' + e.color + '"></div>' +
      '<span class="legend-label">' + e.sid + ': ' + e.desc + '</span>';
    row.addEventListener('click', function() {{
      toggleSubtopic(e.sid, row);
    }});
    legendContent.appendChild(row);
  }});

  // Track hidden subtopic trace names
  var hiddenSids = new Set();
  function toggleSubtopic(sid, row) {{
    var plot = document.querySelector('.plotly-graph-div');
    var traceIdx = plot.data.findIndex(function(t) {{ return t.name && t.name.startsWith(sid + ':'); }});
    if (traceIdx === -1) return;
    if (hiddenSids.has(sid)) {{
      hiddenSids.delete(sid);
      row.style.opacity = '1';
      Plotly.restyle(plot, {{'visible': true}}, [traceIdx]);
    }} else {{
      hiddenSids.add(sid);
      row.style.opacity = '0.3';
      Plotly.restyle(plot, {{'visible': false}}, [traceIdx]);
    }}
  }}

  // ── Cluster dropdown ──
  var clusterOptions = {cluster_options_json};
  var select = document.getElementById('cluster-select');
  var resetBtn = document.getElementById('cluster-reset');

  clusterOptions.forEach(function(opt) {{
    var el = document.createElement('option');
    el.value = opt.cid;
    el.textContent = opt.label;
    select.appendChild(el);
  }});

  var plot = document.querySelector('.plotly-graph-div');
  var selectedCluster = null;

  // Dropdown change
  select.addEventListener('change', function() {{
    var val = select.value;
    if (val === '') {{
      selectedCluster = null;
      resetOpacity();
    }} else {{
      selectedCluster = parseInt(val);
      filterToCluster(selectedCluster);
    }}
  }});

  // Reset button
  resetBtn.addEventListener('click', function() {{
    select.value = '';
    selectedCluster = null;
    resetOpacity();
  }});

  // Click on centroid
  plot.on('plotly_click', function(data) {{
    var pt = data.points[0];
    var traceName = pt.data.name || '';
    if (!/^C\d+$/.test(traceName)) return;
    var clickedCid = parseInt(traceName.slice(1));
    if (selectedCluster === clickedCid) {{
      selectedCluster = null;
      select.value = '';
      resetOpacity();
    }} else {{
      selectedCluster = clickedCid;
      select.value = clickedCid;
      filterToCluster(clickedCid);
    }}
  }});

  // Double-click canvas: reset
  plot.on('plotly_doubleclick', function() {{
    selectedCluster = null;
    select.value = '';
    resetOpacity();
  }});

  function filterToCluster(cid) {{
    var update = {{ 'marker.opacity': [], 'marker.size': [] }};
    plot.data.forEach(function(trace) {{
      if (!trace.customdata || trace.customdata.length === 0) {{
        update['marker.opacity'].push(trace.marker ? trace.marker.opacity : undefined);
        update['marker.size'].push(trace.marker ? trace.marker.size : undefined);
        return;
      }}
      var opacities = [], sizes = [];
      trace.customdata.forEach(function(cd) {{
        if (cd[0] === cid) {{ opacities.push(0.92); sizes.push(9); }}
        else               {{ opacities.push(0.04); sizes.push(4); }}
      }});
      update['marker.opacity'].push(opacities);
      update['marker.size'].push(sizes);
    }});
    Plotly.restyle(plot, update);
  }}

  function resetOpacity() {{
    var update = {{ 'marker.opacity': [], 'marker.size': [] }};
    plot.data.forEach(function(trace) {{
      if (!trace.customdata || trace.customdata.length === 0) {{
        update['marker.opacity'].push(trace.marker ? trace.marker.opacity : undefined);
        update['marker.size'].push(trace.marker ? trace.marker.size : undefined);
        return;
      }}
      var isUnlinked = trace.name === 'unlinked';
      update['marker.opacity'].push(isUnlinked ? 0.3 : 0.80);
      update['marker.size'].push(isUnlinked ? 4 : 6);
    }});
    Plotly.restyle(plot, update);
  }}
}})();
</script>
"""

    html = html.replace("</body>", inject + "\n</body>")

    with open(OUT_PATH, "w") as f:
        f.write(html)
    print(f"\nWritten: {OUT_PATH}")
    print(f"Open with: open {OUT_PATH}")

if __name__ == "__main__":
    main()
