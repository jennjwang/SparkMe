"""
DenStream-inspired fading and pruning for the online task clustering pipeline.

Cluster weights decay exponentially per interview (ingestion event):
    w(n) = w * 2^(-lambda_ * delta_interviews)

lambda_ is now per-interview, not per-hour:
    lambda=0.1  → weight halves every ~10 interviews
    lambda=0.01 → weight halves every ~100 interviews
    lambda=1.0  → weight halves every interview

Fading is applied every `fade_interval_interviews` ingested items
(tracked via state.total_processed).

References:
    Cao, F. et al. (2006). Density-based clustering over an evolving data stream
    with noise. Proc. SDM.
"""

from __future__ import annotations

from .models import ClusterState


def fade_all(state: ClusterState) -> None:
    """
    Apply exponential weight decay to every cluster and outlier buffer item
    since the last fade.

    Uses state.total_processed - state.last_fade_count as delta_interviews.
    Modifies state in-place. Updates state.last_fade_count.
    """
    delta = state.total_processed - state.last_fade_count
    if delta <= 0:
        return

    factor = 2.0 ** (-state.lambda_ * delta)
    for cluster in state.clusters.values():
        cluster.weight *= factor
    for item in state.outlier_buffer:
        item.weight *= factor

    state.last_fade_count = state.total_processed


def prune(state: ClusterState) -> int:
    """
    Remove leaf clusters and outlier buffer items whose weight has fallen
    below state.eps.

    Anchored clusters (taxonomy seeds) are never pruned.
    Also cleans up dangling child references from parent clusters.

    Returns the total number of objects removed (clusters + buffer items).
    """
    to_prune = [
        c for c in state.clusters.values()
        if c.is_leaf
        and not c.anchored
        and c.weight < state.eps
    ]

    for cluster in to_prune:
        del state.clusters[cluster.id]
        if cluster.parent_id and cluster.parent_id in state.clusters:
            parent = state.clusters[cluster.parent_id]
            if cluster.id in parent.children:
                parent.children.remove(cluster.id)

    before_buf = len(state.outlier_buffer)
    state.outlier_buffer = [
        item for item in state.outlier_buffer if item.weight >= state.eps
    ]
    pruned_buf = before_buf - len(state.outlier_buffer)

    return len(to_prune) + pruned_buf


def fade_and_prune(state: ClusterState) -> int:
    """Convenience: fade then prune. Returns count of pruned clusters."""
    fade_all(state)
    return prune(state)
