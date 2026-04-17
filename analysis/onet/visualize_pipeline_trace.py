"""
Build an interactive HTML viewer showing what was filtered at each pipeline stage.

Usage:
    python analysis/onet/visualize_pipeline_trace.py
"""

import json
import html
from pathlib import Path


STAGE_COLORS = {
    "raw":      "#6c757d",
    "valid":    "#28a745",
    "rewritten_3a": "#fd7e14",
    "rejected": "#dc3545",
    "grouped":  "#007bff",
    "rewritten":"#17a2b8",
    "deduped":  "#6610f2",
}

CHECK_LABELS = {
    "specific_action":       "Specific action",
    "concrete_object":       "Concrete object",
    "bounded_activity":      "Bounded activity",
    "single_task":           "Single task",
    "common_to_occupation":  "Common to occupation",
}


def build_source_quote_lookup(study_tasks: list) -> dict:
    """Build lookup: task_statement text → {occupation, sources dict with interview quotes}."""
    lookup = {}
    for participant in study_tasks:
        occ = participant.get("occupation", "")
        for task in participant.get("tasks", []):
            stmt = task.get("task_statement", "")
            if stmt:
                key = stmt.strip().lower()
                lookup[key] = {
                    "occupation": occ,
                    "sources": task.get("sources", {}),
                    "task_statement": stmt,
                }
    return lookup


def _find_source_quotes(source_stmt: str, quote_lookup: dict) -> dict:
    """Try to find interview quotes for a source statement."""
    # Strip [Occupation] prefix and normalization artifacts
    clean = source_stmt.strip()
    if clean.startswith("["):
        bracket_end = clean.find("]")
        if bracket_end > 0:
            clean = clean[bracket_end + 1:].strip()
    # May have double [Occupation] prefix from pipeline bugs
    if clean.startswith("["):
        bracket_end = clean.find("]")
        if bracket_end > 0:
            clean = clean[bracket_end + 1:].strip()
    # Strip appended metadata after |
    if "|" in clean:
        clean = clean.split("|")[0].strip()

    key = clean.lower()
    if key in quote_lookup:
        return quote_lookup[key]
    # Fuzzy: try prefix match
    for k, v in quote_lookup.items():
        if k.startswith(key[:60]) or key.startswith(k[:60]):
            return v
    return {}


def build_html(trace: list, quote_lookup: dict = None) -> str:
    if quote_lookup is None:
        quote_lookup = {}
    total_raw = sum(len(c["raw_tasks"]) for c in trace)
    total_valid = sum(len(c["stage_3a_valid"]) for c in trace)
    total_rewritten_3a = sum(len(c.get("stage_3a_rewritten", [])) for c in trace)
    total_rejected = sum(len(c.get("stage_3a_rejected", c.get("stage_3a_invalid", []))) for c in trace)
    total_pass_screening = total_valid + total_rewritten_3a
    total_grouped = sum(len(c["stage_3b_grouped"]) for c in trace)
    total_deduped = sum(len(c["stage_3d_deduped"]) for c in trace)

    # Count check failures across rejected
    check_fail_counts = {k: 0 for k in CHECK_LABELS}
    for cat in trace:
        for inv in cat.get("stage_3a_rejected", cat.get("stage_3a_invalid", [])):
            checks = inv.get("checks", {})
            for k in CHECK_LABELS:
                if not checks.get(k, True):
                    check_fail_counts[k] += 1

    cat_sections = []
    for cat in sorted(trace, key=lambda c: c["category"]):
        cat_sections.append(_build_category(cat, quote_lookup))

    funnel = f"""
    <div class="funnel">
      <div class="funnel-step" style="background:{STAGE_COLORS['raw']}">
        <div class="funnel-num">{total_raw}</div>
        <div class="funnel-label">Raw statements</div>
      </div>
      <div class="funnel-arrow">&rarr;</div>
      <div class="funnel-step" style="background:{STAGE_COLORS['valid']}">
        <div class="funnel-num">{total_valid}</div>
        <div class="funnel-label">3a: Passed as-is</div>
      </div>
      <div class="funnel-plus">+</div>
      <div class="funnel-step" style="background:{STAGE_COLORS['rewritten_3a']}">
        <div class="funnel-num">{total_rewritten_3a}</div>
        <div class="funnel-label">3a: Rewritten</div>
      </div>
      <div class="funnel-eq">=</div>
      <div class="funnel-step" style="background:#218838">
        <div class="funnel-num">{total_pass_screening}</div>
        <div class="funnel-label">To grouping</div>
      </div>
      <div class="funnel-arrow">&rarr;</div>
      <div class="funnel-step" style="background:{STAGE_COLORS['grouped']}">
        <div class="funnel-num">{total_grouped}</div>
        <div class="funnel-label">3b: Grouped</div>
      </div>
      <div class="funnel-arrow">&rarr;</div>
      <div class="funnel-step" style="background:{STAGE_COLORS['deduped']}">
        <div class="funnel-num">{total_deduped}</div>
        <div class="funnel-label">3d: Final</div>
      </div>
    </div>
    <div class="rejected-badge">
      <span style="color:{STAGE_COLORS['rejected']}; font-weight:600;">{total_rejected} rejected</span>
      <span style="color:#666;">({total_rejected}/{total_raw} = {total_rejected/total_raw*100:.0f}% of raw statements unsalvageable)</span>
    </div>
    """

    # Screening failure breakdown (for rejected only)
    fail_bars = ""
    for k, label in CHECK_LABELS.items():
        count = check_fail_counts[k]
        pct = (count / total_rejected * 100) if total_rejected else 0
        fail_bars += f"""
        <div class="fail-row">
          <span class="fail-label">{label}</span>
          <div class="fail-bar-bg">
            <div class="fail-bar" style="width:{pct:.0f}%">{count}</div>
          </div>
          <span class="fail-pct">{pct:.0f}%</span>
        </div>"""

    screening_summary = f"""
    <div class="screening-summary">
      <h3>Stage 3a — Rejected statements ({total_rejected} unsalvageable)</h3>
      <p class="subtitle">Which checks caused final rejections? (after rewrite attempts failed)</p>
      {fail_bars}
    </div>
    """

    return f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Pipeline Trace — Aggregation Stages</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #212529; }}
  h1 {{ margin-bottom: 8px; }}
  .meta {{ color: #666; margin-bottom: 20px; }}

  .funnel {{ display: flex; align-items: center; gap: 8px; margin: 24px 0; flex-wrap: wrap; }}
  .funnel-step {{ color: white; border-radius: 8px; padding: 16px 24px; text-align: center; min-width: 110px; }}
  .funnel-num {{ font-size: 28px; font-weight: 700; }}
  .funnel-label {{ font-size: 13px; opacity: 0.9; }}
  .funnel-arrow, .funnel-plus, .funnel-eq {{ font-size: 24px; color: #999; }}

  .rejected-badge {{ margin: -12px 0 16px 0; }}

  .screening-summary {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                        border: 1px solid #dee2e6; }}
  .screening-summary h3 {{ margin-bottom: 4px; }}
  .subtitle {{ color: #666; font-size: 13px; margin-bottom: 12px; }}
  .fail-row {{ display: flex; align-items: center; gap: 8px; margin: 6px 0; }}
  .fail-label {{ width: 160px; font-size: 13px; text-align: right; }}
  .fail-bar-bg {{ flex: 1; background: #e9ecef; border-radius: 4px; height: 22px; }}
  .fail-bar {{ background: #dc3545; border-radius: 4px; height: 22px; color: white;
               font-size: 12px; line-height: 22px; padding: 0 8px; min-width: fit-content; }}
  .fail-pct {{ width: 40px; font-size: 13px; color: #666; }}

  .category {{ background: white; border-radius: 8px; padding: 20px; margin: 16px 0;
               border: 1px solid #dee2e6; }}
  .cat-header {{ display: flex; align-items: center; gap: 12px; cursor: pointer; }}
  .cat-header h2 {{ flex: 1; font-size: 18px; }}
  .cat-funnel {{ display: flex; gap: 4px; align-items: center; font-size: 13px; }}
  .cat-num {{ display: inline-block; padding: 2px 10px; border-radius: 12px; color: white;
              font-weight: 600; }}

  .cat-body {{ display: none; margin-top: 16px; }}
  .category.open .cat-body {{ display: block; }}

  .stage {{ margin: 16px 0; }}
  .stage-title {{ font-size: 14px; font-weight: 600; margin-bottom: 8px; padding: 4px 8px;
                  border-radius: 4px; color: white; display: inline-block; }}

  .task-card {{ border: 1px solid #e9ecef; border-radius: 6px; padding: 10px 14px; margin: 6px 0;
                font-size: 13px; }}
  .task-card.rejected {{ border-left: 3px solid #dc3545; background: #fff5f5; }}
  .task-card.valid {{ border-left: 3px solid #28a745; }}
  .task-card.rewritten-3a {{ border-left: 3px solid #fd7e14; background: #fff8f0; }}
  .task-card.grouped {{ border-left: 3px solid #007bff; }}
  .task-card.deduped {{ border-left: 3px solid #6610f2; }}

  .task-statement {{ font-weight: 500; margin-bottom: 4px; }}
  .task-original {{ color: #888; font-size: 12px; text-decoration: line-through; margin-bottom: 4px; }}
  .task-rewritten {{ color: #e65100; font-weight: 600; margin-bottom: 4px; }}
  .task-meta {{ color: #666; font-size: 12px; }}
  .task-meta .occupation {{ background: #e9ecef; padding: 1px 6px; border-radius: 3px; }}
  .task-meta .reason {{ color: #dc3545; }}
  .task-meta .fix-reason {{ color: #e65100; }}

  .checks {{ display: flex; gap: 6px; flex-wrap: wrap; margin-top: 4px; }}
  .check {{ font-size: 11px; padding: 1px 6px; border-radius: 3px; }}
  .check.pass {{ background: #d4edda; color: #155724; }}
  .check.fail {{ background: #f8d7da; color: #721c24; font-weight: 600; }}

  .sources {{ font-size: 12px; color: #666; margin-top: 4px; }}
  .sources summary {{ cursor: pointer; color: #007bff; }}
  .source-item {{ margin: 6px 0; padding-left: 12px; border-left: 2px solid #dee2e6; }}
  .source-stmt {{ font-size: 12px; color: #333; }}
  .source-quote {{ font-size: 11px; color: #888; font-style: italic; margin-top: 2px; }}

  .merge-note {{ font-size: 12px; color: #6610f2; font-style: italic; margin-top: 4px; }}
  .arrow-label {{ text-align: center; color: #999; font-size: 13px; margin: 8px 0; }}
</style>
</head><body>
<h1>Pipeline Trace — Aggregation Stages</h1>
<p class="meta">{len(trace)} categories, {total_raw} raw statements &rarr; {total_deduped} final canonical tasks</p>

{funnel}
{screening_summary}

<div id="categories">
{"".join(cat_sections)}
</div>

<script>
document.querySelectorAll('.cat-header').forEach(h => {{
  h.addEventListener('click', () => h.parentElement.classList.toggle('open'));
}});
</script>
</body></html>"""


def _build_category(cat: dict, quote_lookup: dict = None) -> str:
    if quote_lookup is None:
        quote_lookup = {}
    name = cat["category"]
    n_raw = len(cat["raw_tasks"])
    n_valid = len(cat["stage_3a_valid"])
    n_rewritten_3a = len(cat.get("stage_3a_rewritten", []))
    n_rejected = len(cat.get("stage_3a_rejected", cat.get("stage_3a_invalid", [])))
    n_grouped = len(cat["stage_3b_grouped"])
    n_deduped = len(cat["stage_3d_deduped"])

    funnel = f"""
    <div class="cat-funnel">
      <span class="cat-num" style="background:{STAGE_COLORS['raw']}">{n_raw}</span> &rarr;
      <span class="cat-num" style="background:{STAGE_COLORS['valid']}">{n_valid}</span> +
      <span class="cat-num" style="background:{STAGE_COLORS['rewritten_3a']}">{n_rewritten_3a}</span>
      (<span style="color:#dc3545">&minus;{n_rejected}</span>) &rarr;
      <span class="cat-num" style="background:{STAGE_COLORS['grouped']}">{n_grouped}</span> &rarr;
      <span class="cat-num" style="background:{STAGE_COLORS['deduped']}">{n_deduped}</span>
    </div>"""

    # Stage 3a — Rejected statements
    rejected_cards = ""
    for inv in cat.get("stage_3a_rejected", cat.get("stage_3a_invalid", [])):
        stmt = html.escape(_extract_statement(inv))
        occ = html.escape(inv.get("occupation", ""))
        reason = html.escape(inv.get("reason", ""))
        checks = inv.get("checks", {})
        check_badges = _check_badges(checks)

        rejected_cards += f"""
        <div class="task-card rejected">
          <div class="task-statement">{stmt}</div>
          <div class="task-meta"><span class="occupation">{occ}</span>
            {f'<span class="reason">&mdash; {reason}</span>' if reason else ''}
          </div>
          <div class="checks">{check_badges}</div>
        </div>"""

    # Stage 3a — Rewritten statements
    rewritten_3a_cards = ""
    for rw in cat.get("stage_3a_rewritten", []):
        original = html.escape(_extract_statement_from_original(rw))
        rewritten = html.escape(rw.get("rewritten", rw.get("statement", "")))
        occ = html.escape(rw.get("occupation", ""))
        reason = html.escape(rw.get("reason", ""))
        checks = rw.get("checks", {})
        check_badges = _check_badges(checks)

        rewritten_3a_cards += f"""
        <div class="task-card rewritten-3a">
          <div class="task-original">{original}</div>
          <div class="task-rewritten">&rarr; {rewritten}</div>
          <div class="task-meta"><span class="occupation">{occ}</span>
            {f'<span class="fix-reason">&mdash; {reason}</span>' if reason else ''}
          </div>
          <div class="checks">{check_badges}</div>
        </div>"""

    # Stage 3a — Valid statements (passed as-is)
    valid_cards = ""
    for v in cat["stage_3a_valid"]:
        stmt = html.escape(_extract_statement(v))
        occ = html.escape(v.get("occupation", ""))
        valid_cards += f"""
        <div class="task-card valid">
          <div class="task-statement">{stmt}</div>
          <div class="task-meta"><span class="occupation">{occ}</span></div>
        </div>"""

    # Stage 3b — Grouped
    grouped_cards = ""
    for g in cat["stage_3b_grouped"]:
        stmt = html.escape(g.get("canonical_statement", ""))
        phrase = html.escape(g.get("abbreviated_phrase", ""))
        n_src = g.get("source_count", 0)
        sources = g.get("source_statements", [])
        source_html = _source_details_with_quotes(sources, quote_lookup)

        grouped_cards += f"""
        <div class="task-card grouped">
          <div class="task-statement">{stmt}</div>
          <div class="task-meta">{phrase} &middot; n={n_src}</div>
          {source_html}
        </div>"""

    # Stage 3d — Final deduped
    deduped_cards = ""
    for d in cat["stage_3d_deduped"]:
        stmt = html.escape(d.get("canonical_statement", ""))
        notes = d.get("notes", "")
        n_src = d.get("source_count", 0)
        sources = d.get("source_statements", [])
        source_html = _source_details_with_quotes(sources, quote_lookup)
        merge_html = f'<div class="merge-note">{html.escape(notes)}</div>' if "merged" in notes.lower() else ""

        deduped_cards += f"""
        <div class="task-card deduped">
          <div class="task-statement">{stmt}</div>
          <div class="task-meta">n={n_src}</div>
          {merge_html}
          {source_html}
        </div>"""

    return f"""
    <div class="category">
      <div class="cat-header">
        <h2>{html.escape(name)}</h2>
        {funnel}
      </div>
      <div class="cat-body">
        <div class="stage">
          <span class="stage-title" style="background:{STAGE_COLORS['rejected']}">3a &mdash; Rejected ({n_rejected})</span>
          {rejected_cards if rejected_cards else '<p style="color:#999;font-size:13px;margin-top:4px;">None rejected</p>'}
        </div>

        <div class="stage">
          <span class="stage-title" style="background:{STAGE_COLORS['rewritten_3a']}">3a &mdash; Rewritten ({n_rewritten_3a})</span>
          {rewritten_3a_cards if rewritten_3a_cards else '<p style="color:#999;font-size:13px;margin-top:4px;">None needed rewriting</p>'}
        </div>

        <div class="stage">
          <span class="stage-title" style="background:{STAGE_COLORS['valid']}">3a &mdash; Passed As-Is ({n_valid})</span>
          {valid_cards if valid_cards else '<p style="color:#999;font-size:13px;margin-top:4px;">None passed as-is</p>'}
        </div>

        <div class="arrow-label">&darr; Group &amp; merge</div>

        <div class="stage">
          <span class="stage-title" style="background:{STAGE_COLORS['grouped']}">3b &mdash; Grouped ({n_grouped})</span>
          {grouped_cards if grouped_cards else '<p style="color:#999;font-size:13px;margin-top:4px;">No tasks</p>'}
        </div>

        <div class="arrow-label">&darr; Rewrite &amp; dedup</div>

        <div class="stage">
          <span class="stage-title" style="background:{STAGE_COLORS['deduped']}">3d &mdash; Final ({n_deduped})</span>
          {deduped_cards if deduped_cards else '<p style="color:#999;font-size:13px;margin-top:4px;">No tasks</p>'}
        </div>
      </div>
    </div>"""


def _check_badges(checks: dict) -> str:
    badges = ""
    for k, label in CHECK_LABELS.items():
        passed = checks.get(k, True)
        cls = "pass" if passed else "fail"
        icon = "&#10003;" if passed else "&#10007;"
        badges += f'<span class="check {cls}">{icon} {label}</span>'
    return badges


def _source_details(sources: list) -> str:
    if not sources:
        return ""
    items = "".join(f'<div class="source-item">{html.escape(s)}</div>' for s in sources)
    return f'<details class="sources"><summary>{len(sources)} source statements</summary>{items}</details>'


def _source_details_with_quotes(sources: list, quote_lookup: dict) -> str:
    """Show source statements with their interview quotes from study_tasks.json."""
    if not sources:
        return ""
    items = ""
    for s in sources:
        match = _find_source_quotes(s, quote_lookup)
        interview_quotes = match.get("sources", {}) if match else {}
        # Use the task_statement quote as the primary interview source
        quote = interview_quotes.get("task_statement", "")
        stmt_display = html.escape(_extract_statement({"statement": s}))
        if quote:
            items += (
                f'<div class="source-item">'
                f'<div class="source-stmt">{stmt_display}</div>'
                f'<div class="source-quote">&ldquo;{html.escape(quote)}&rdquo;</div>'
                f'</div>'
            )
        else:
            items += f'<div class="source-item"><div class="source-stmt">{stmt_display}</div></div>'
    return f'<details class="sources"><summary>{len(sources)} source statements</summary>{items}</details>'


def _extract_statement(entry: dict) -> str:
    """Extract the task statement text from a screening result."""
    stmt = entry.get("statement", "")
    if "|" in stmt:
        stmt = stmt.split("|")[0].strip()
    if stmt.startswith("["):
        bracket_end = stmt.find("]")
        if bracket_end > 0:
            stmt = stmt[bracket_end + 1:].strip()
    return stmt


def _extract_statement_from_original(entry: dict) -> str:
    """Extract original statement text from a rewritten entry."""
    stmt = entry.get("original_statement", entry.get("statement", ""))
    if "|" in stmt:
        stmt = stmt.split("|")[0].strip()
    if stmt.startswith("["):
        bracket_end = stmt.find("]")
        if bracket_end > 0:
            stmt = stmt[bracket_end + 1:].strip()
    return stmt


def main():
    root = Path(__file__).parent.parent.parent
    trace_path = root / "analysis/onet/data/pipeline_trace.json"
    study_path = root / "analysis/onet/data/study_tasks.json"
    output_path = root / "analysis/onet/data/pipeline_trace_viewer.html"

    with open(trace_path) as f:
        trace = json.load(f)

    quote_lookup = {}
    if study_path.exists():
        with open(study_path) as f:
            quote_lookup = build_source_quote_lookup(json.load(f))

    html_content = build_html(trace, quote_lookup)
    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Wrote → {output_path}")


if __name__ == "__main__":
    main()
