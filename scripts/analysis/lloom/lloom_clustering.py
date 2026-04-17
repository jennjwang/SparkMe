"""
LLooM concept induction on memory bank content.

Runs LLooM's full pipeline:
  1. gen()   — distill + cluster + synthesize concepts from memories
  2. score() — score all memories against discovered concepts
  3. Save results to user_study/lloom_results.json

Usage:
    .venv-analysis/bin/python scripts/analysis/lloom_clustering.py

LLooM docs: https://stanfordhci.github.io/lloom/
"""

import asyncio
import json
import os
from pathlib import Path

import builtins
import pandas as pd
from dotenv import load_dotenv

# Auto-confirm LLooM's interactive prompts
_real_input = builtins.input
def _auto_input(prompt=""):
    print(prompt + "y")
    return "y"
builtins.input = _auto_input

# Stub out Jupyter display() used by LLooM outside notebook context
import builtins as _builtins
try:
    display
except NameError:
    import IPython.display as _ipd
    import builtins
    builtins.display = _ipd.display
except Exception:
    builtins.display = lambda *a, **kw: None

load_dotenv(dotenv_path=".env")

# Patch LLooM's cluster function to use a smaller min_cluster_size so we get
# more than ~3 clusters when the bullet corpus is large.
# Default: max(3, n_items/10) → with 1500 bullets → min_cluster_size=150 → 3 clusters.
# We cap it at 20 so HDBSCAN produces ~50–80 clusters.
def _patch_lloom_cluster():
    import text_lloom.concept_induction as ci
    _orig_cluster = ci.cluster

    async def _patched_cluster(text_df, doc_col, doc_id_col, embed_model,
                               cluster_id_col="cluster_id", min_cluster_size=None,
                               batch_size=20, randomize=False, sess=None):
        import math
        n_items = len(text_df)
        if min_cluster_size is None:
            min_cluster_size = min(10, max(3, int(n_items / 10)))
        print(f"  [patch] cluster: n_items={n_items}, min_cluster_size={min_cluster_size}")
        return await _orig_cluster(
            text_df=text_df,
            doc_col=doc_col,
            doc_id_col=doc_id_col,
            embed_model=embed_model,
            cluster_id_col=cluster_id_col,
            min_cluster_size=min_cluster_size,
            batch_size=batch_size,
            randomize=randomize,
            sess=sess,
        )

    ci.cluster = _patched_cluster
    # Workbench does `from .concept_induction import *` so cluster lives in its namespace too
    try:
        import text_lloom.workbench as wb_mod
        wb_mod.cluster = _patched_cluster
    except Exception:
        pass

_patch_lloom_cluster()

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
USER_STUDY_DIR = BASE_DIR / "user_study"
TOPICS_PATH = BASE_DIR / "data" / "configs" / "topics.json"
OUT_PATH = USER_STUDY_DIR / "lloom_results.json"
BULLETS_CACHE = USER_STUDY_DIR / "lloom_bullets_cache.parquet"


def load_memories():
    memories = []
    for pid in sorted(os.listdir(USER_STUDY_DIR)):
        mem_path = USER_STUDY_DIR / pid / "memory_bank_content.json"
        if not mem_path.is_file():
            continue
        with open(mem_path) as f:
            mem_data = json.load(f)
        for m in mem_data.get("memories", []):
            links = m.get("subtopic_links", [])
            top = max(links, key=lambda l: l.get("importance", 0)) if links else None
            memories.append({
                "id": m["id"],
                "pid": pid,
                "title": m.get("title", ""),
                "text": m.get("text", ""),
                "top_subtopic": top["subtopic_id"] if top else "",
                # LLooM text: title + text for richer signal
                "doc": f"{m.get('title', '')}. {m.get('text', '')}",
            })
    return memories


async def run_lloom(df):
    import sys
    import text_lloom.workbench as wb
    import text_lloom.concept_induction as ci

    print(f"  Creating LLooM workbench with {len(df)} documents...")
    l = wb.lloom(
        df=df,
        text_col="doc",
        id_col="id",
    )

    # Request more concepts: suggest params with higher target, then override synth_n_concepts
    params = l.auto_suggest_parameters(target_n_concepts=40)
    params["synth_n_concepts"] = 8   # concepts per cluster
    params["filter_n_quotes"] = 1    # skip distill-filter (avoids timeouts on large corpus)
    print(f"  Params: {params}")

    async def _synthesize_from_cluster(df_cluster, seed, label):
        """Synthesize concepts from a cluster df, returning count of new concepts added."""
        before = len(l.concepts)
        await ci.synthesize(
            cluster_df=df_cluster[df_cluster["cluster_id"] >= 0],
            doc_col=l.doc_col,
            doc_id_col=l.doc_id_col,
            model=l.synth_model,
            concept_col_prefix="concept",
            n_concepts=params["synth_n_concepts"],
            pattern_phrase="unique topic",
            seed=seed,
            sess=l,
            return_logs=True,
        )
        n_new = len(l.concepts) - before
        print(f"  {label}: {n_new} concepts (total {len(l.concepts)})", flush=True)
        return n_new

    async def _noise_recovery(df_cluster, label):
        """Re-cluster HDBSCAN noise points (cluster_id == -1) with a smaller min_cluster_size."""
        noise_df = df_cluster[df_cluster["cluster_id"] == -1][[l.doc_id_col, l.doc_col]].copy()
        n_noise = len(noise_df)
        pct_noise = round(100 * n_noise / max(len(df_cluster), 1), 1)
        print(f"\n  Noise recovery ({label}): {n_noise} unclustered bullets ({pct_noise}%)", flush=True)
        if n_noise < 5:
            print("  Noise recovery: too few noise points, skipping", flush=True)
            return
        df_cluster_noise = await ci.cluster(
            text_df=noise_df,
            doc_col=l.doc_col,
            doc_id_col=l.doc_id_col,
            embed_model=l.cluster_model,
            min_cluster_size=3,
            sess=l,
        )
        n_sub = (df_cluster_noise["cluster_id"] >= 0).sum()
        print(f"  Noise recovery: {n_sub}/{n_noise} bullets re-assigned to "
              f"{df_cluster_noise[df_cluster_noise['cluster_id'] >= 0]['cluster_id'].nunique()} sub-clusters",
              flush=True)
        if n_sub > 0:
            await _synthesize_from_cluster(df_cluster_noise, seed="AI adoption at work", label="noise concepts")

    # Check if we have cached bullet summaries to skip the slow distill step
    if BULLETS_CACHE.is_file():
        print(f"\n  Loading cached bullets from {BULLETS_CACHE}...")
        df_bullets = pd.read_parquet(BULLETS_CACHE)
        l.df_bullets = df_bullets
        l.df_filtered = df[["id", "doc"]]
        print(f"  Loaded {len(df_bullets)} bullet rows")

        # Still need to run cluster+synthesize (pass 1) to initialize l.concepts
        print("\n  Running cluster+synthesize pass 1 (from cache)...")
        df_cluster = await ci.cluster(
            text_df=l.df_bullets,
            doc_col=l.doc_col,
            doc_id_col=l.doc_id_col,
            embed_model=l.cluster_model,
            sess=l,
        )
        l.concepts = {}
        await _synthesize_from_cluster(df_cluster, seed="AI adoption at work", label="pass 1")
        # Recover concepts from bullets HDBSCAN labelled as noise
        await _noise_recovery(df_cluster, label="pass 1")
    else:
        # Full run: distill + cluster + synthesize
        print("\n  Running gen() pass 1 — distill + cluster + synthesize...")
        await l.gen(seed="AI adoption at work", params=params, n_synth=1, auto_review=False, debug=True)
        print(f"  Pass 1: {len(l.concepts)} concepts")
        # Cache bullets for future runs
        print(f"  Caching bullets → {BULLETS_CACHE}")
        l.df_bullets.to_parquet(BULLETS_CACHE, index=False)
        # gen() doesn't expose df_cluster, so run a fine-grained cluster pass
        # on all bullets (min_cluster_size=3) to surface micro-clusters that the
        # larger threshold missed — equivalent to noise recovery for the first run.
        print("\n  Running fine-grained noise-recovery pass (first run)...")
        df_cluster_fine = await ci.cluster(
            text_df=l.df_bullets,
            doc_col=l.doc_col,
            doc_id_col=l.doc_id_col,
            embed_model=l.cluster_model,
            min_cluster_size=3,
            sess=l,
        )
        await _synthesize_from_cluster(df_cluster_fine, seed="AI adoption at work", label="fine-grained pass")

    all_concepts = dict(l.concepts)
    # Write to stderr so spinner doesn't swallow it
    import sys
    print(f"PASS1_CONCEPTS={len(all_concepts)}", file=sys.stderr, flush=True)

    # Passes 2 & 3: reuse cached bullets, random batching → fresh groupings each time
    for pass_i in range(2, 4):
        print(f"\n  Running cluster+synthesize pass {pass_i}...", flush=True)
        sys.stdout.flush()
        try:
            df_cluster = await ci.cluster(
                text_df=l.df_bullets,
                doc_col=l.doc_col,
                doc_id_col=l.doc_id_col,
                embed_model=l.cluster_model,
                randomize=True,
                sess=l,
            )
            print(f"PASS{pass_i}_CLUSTERS={df_cluster['cluster_id'].nunique()}", file=sys.stderr, flush=True)
            await _synthesize_from_cluster(
                df_cluster,
                seed=f"AI adoption at work pass {pass_i}",
                label=f"pass {pass_i}",
            )
            # Also recover noise from each randomized pass
            await _noise_recovery(df_cluster, label=f"pass {pass_i}")
            new_concepts = {k: v for k, v in l.concepts.items() if k not in all_concepts}
            all_concepts.update(new_concepts)
            print(f"PASS{pass_i}_NEW={len(new_concepts)}_TOTAL={len(all_concepts)}", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"PASS{pass_i}_ERROR={e}", file=sys.stderr, flush=True)
            import traceback; traceback.print_exc(file=sys.stderr)

    l.concepts = all_concepts
    print(f"\n  Total concepts before scoring: {len(l.concepts)}", flush=True)

    print("\n  Running score() — scoring all documents against concepts...")
    score_df = await l.score(score_all=True, batch_size=5, get_highlights=True, debug=True)

    return l, score_df


def save_results(l, score_df, memories_df):
    # Extract concepts
    concepts = []
    for c_id, concept in l.concepts.items():
        concepts.append({
            "id": c_id,
            "name": getattr(concept, "name", str(c_id)),
            "prompt": getattr(concept, "prompt", ""),
            "summary": getattr(concept, "summary", ""),
        })

    # score_df is long-format: one row per (doc, concept)
    # columns: doc_id, text, concept_id, concept_name, concept_prompt, score, rationale, highlight, concept_seed
    if score_df is not None and len(score_df) > 0:
        sdf = score_df.copy()
        sdf["doc_id"] = sdf["doc_id"].astype(str)
        meta = memories_df[["id", "pid", "title", "top_subtopic"]].copy()
        meta["id"] = meta["id"].astype(str)
        merged = sdf.merge(meta, left_on="doc_id", right_on="id", how="left").drop(columns=["id"])
        records = merged.to_dict(orient="records")
    else:
        records = []

    out = {
        "n_concepts": len(concepts),
        "concepts": concepts,
        "scores": records,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Results → {OUT_PATH}")
    print(f"  {len(concepts)} concepts discovered")
    for c in concepts:
        print(f"    [{c['id'][:8]}] {c['name']}")


async def main():
    print("=" * 55)
    print("LLooM CONCEPT INDUCTION")
    print("=" * 55)

    print("\n[1/3] Loading memories...")
    memories = load_memories()
    print(f"  {len(memories)} memories from {len(set(m['pid'] for m in memories))} participants")

    df = pd.DataFrame(memories)

    print("\n[2/3] Running LLooM...")
    l, score_df = await run_lloom(df)

    print("\n[3/3] Saving results...")
    save_results(l, score_df, df)


if __name__ == "__main__":
    asyncio.run(main())
