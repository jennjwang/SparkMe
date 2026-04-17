"""
Incremental Divisive Clustering for the online task clustering pipeline.

Hierarchy direction (read bottom-up):
  leaves        = specific subtask instances  (most granular)
  internal nodes = progressively broader task groupings
  root(s)       = broadest task categories

**Incremental**: after each item is assigned to a leaf cluster, that specific
cluster is checked for a split (via `try_split`).  There is no periodic
full-scan — the pipeline only pays for an LLM split call on the single
cluster that just grew.  A bulk `check_and_split` is still available for
one-time catch-up (e.g. after loading a saved state or a taxonomy warmup).

Splitting is triggered when a leaf cluster's member count reaches
state.max_split_size AND its depth is below state.split_depth_limit.
The LLM is asked whether 2–4 meaningful sub-types exist; if yes, child
clusters are created and the original becomes an internal node.

New items are never directly assigned to internal nodes — only to leaves.
"""

from __future__ import annotations
import uuid
from datetime import datetime

from .models import Cluster, ClusterState
from . import llm_ops


def _build_child(
    parent: Cluster,
    label: str,
    member_ids: list[str],
    state: ClusterState,
    now: datetime,
) -> Cluster:
    child = Cluster(
        id=str(uuid.uuid4()),
        leader=label,
        members=member_ids,
        weight=float(len(member_ids)),
        last_updated=now,
        created_at=now,
        parent_id=parent.id,
        children=[],
        level=parent.level + 1,
        anchored=False,
    )
    state.clusters[child.id] = child
    return child


def _split(cluster: Cluster, state: ClusterState, llm, model: str) -> list[str]:
    """
    Attempt to divisively split `cluster` into 2–4 sub-clusters.

    Returns the list of new child cluster ids, or [] if the cluster
    was deemed homogeneous by the LLM.
    """
    member_texts = [
        state.items[mid].text
        for mid in cluster.members
        if mid in state.items
    ]
    if not member_texts:
        return []

    groups = llm_ops.split_cluster(
        leader=cluster.leader,
        member_texts=member_texts,
        llm=llm,
        model=model,
    )

    if len(groups) <= 1:
        return []  # LLM says homogeneous — no split

    now = datetime.now()
    child_ids = []
    for group in groups:
        indices = group.get("member_indices", [])
        label = group.get("label", cluster.leader)
        # Map indices back to member ids (indices may be 0-based from llm_ops)
        child_member_ids = [
            cluster.members[i]
            for i in indices
            if 0 <= i < len(cluster.members)
        ]
        if not child_member_ids:
            continue
        child = _build_child(cluster, label, child_member_ids, state, now)
        child_ids.append(child.id)

    if child_ids:
        # Add new children to existing ones (don't replace — old children are still valid)
        cluster.children.extend(child_ids)
        # Clear direct members — they've been redistributed into new children
        cluster.members = []

    return child_ids


def try_split(
    cluster_id: str,
    state: ClusterState,
    llm,
    model: str = "gpt-4.1",
) -> list[str]:
    """
    Incrementally check a root (level-0) cluster for a split after it receives
    a direct member (an item that didn't fit any existing child).

    Splits only the root's direct members into new children — existing children
    are untouched. The root's member list is cleared after the split (items move
    to new children).

    Returns newly created child cluster ids, or [] if no split occurred.
    """
    if cluster_id not in state.clusters:
        return []
    cluster = state.clusters[cluster_id]
    # Only split level-0 roots; level-1 clusters are the bottom of the tree
    if cluster.level != 0:
        return []
    if len(cluster.members) < state.max_split_size:
        return []
    return _split(cluster, state, llm, model)


def check_and_split(
    state: ClusterState,
    llm,
    model: str = "gpt-4.1",
) -> list[str]:
    """
    Scan all leaf clusters; split any that exceed state.max_split_size
    and are within the allowed depth.

    Returns a flat list of newly created child cluster ids.
    """
    new_ids: list[str] = []

    # Snapshot keys — state.clusters may grow during iteration
    candidate_ids = [
        cid for cid, c in list(state.clusters.items())
        if c.is_leaf
        and len(c.members) >= state.max_split_size
        and c.level < state.split_depth_limit
    ]

    for cid in candidate_ids:
        if cid not in state.clusters:
            continue
        cluster = state.clusters[cid]
        new_ids.extend(_split(cluster, state, llm, model))

    return new_ids


def get_hierarchy(state: ClusterState) -> list[dict]:
    """
    Return the cluster forest as a list of nested dicts (one per root cluster).

    Each node: {"id", "leader", "weight", "member_count", "level", "anchored", "children": [...]}
    """
    def _to_dict(cluster: Cluster) -> dict:
        return {
            "id": cluster.id,
            "leader": cluster.leader,
            "weight": round(cluster.weight, 4),
            "member_count": len(cluster.members),
            "level": cluster.level,
            "anchored": cluster.anchored,
            "children": [
                _to_dict(state.clusters[cid])
                for cid in cluster.children
                if cid in state.clusters
            ],
        }

    roots = [c for c in state.clusters.values() if c.parent_id is None]
    roots.sort(key=lambda c: -c.weight)
    return [_to_dict(r) for r in roots]
