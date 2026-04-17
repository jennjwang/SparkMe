"""
Two-level routing with LLM similarity.

Level 0 (roots)  — broad categories. Items land here when they don't fit any sub-type.
Level 1 (children) — specific sub-types. Items land here when they clearly match.

Routing per item:
  1. LLM picks a root cluster (or "new" → create new root).
  2. If the root has children → LLM picks a child (or "none" → item stays at root).
  3. If the root has no children → item assigned directly to root.

New children are only ever created by divisive splitting (divisive.py), not by routing.
Routing at level 1 only assigns to existing children or falls back to the root.
"""

from __future__ import annotations
import random
import uuid
from datetime import datetime

from .models import Cluster, ClusterState, OutlierItem, TaskItem
from . import llm_ops

LEADER_REFRESH_EVERY = 5


def _sample_texts(cluster: Cluster, state: ClusterState, k: int = 10) -> list[str]:
    sample_ids = cluster.members if len(cluster.members) <= k else random.sample(cluster.members, k)
    return [state.items[mid].text for mid in sample_ids if mid in state.items]


def _new_cluster(task: TaskItem, state: ClusterState, parent_id: str | None, level: int,
                 llm=None, model: str = "gpt-4.1") -> str:
    # Generalize the leader immediately rather than using raw task text
    if llm is not None:
        leader = llm_ops.update_leader(
            leader=task.text,
            member_texts=[task.text],
            llm=llm,
            model=model,
        )
    else:
        leader = task.text

    cluster_id = str(uuid.uuid4())
    cluster = Cluster(
        id=cluster_id,
        leader=leader,
        members=[task.id],
        weight=1.0,
        last_updated=task.timestamp,
        created_at=task.timestamp,
        parent_id=parent_id,
        children=[],
        level=level,
        anchored=False,
    )
    state.clusters[cluster_id] = cluster
    if parent_id and parent_id in state.clusters:
        state.clusters[parent_id].children.append(cluster_id)
    return cluster_id


def _assign_to_cluster(
    task: TaskItem,
    cluster: Cluster,
    state: ClusterState,
    llm,
    model: str,
) -> str:
    cluster.members.append(task.id)
    cluster.weight += 1.0
    cluster.last_updated = task.timestamp
    if not cluster.anchored and len(cluster.members) % LEADER_REFRESH_EVERY == 0:
        cluster.leader = llm_ops.update_leader(
            leader=cluster.leader,
            member_texts=_sample_texts(cluster, state),
            llm=llm,
            model=model,
        )
    return cluster.id


def _promote_from_buffer(
    task: TaskItem,
    matched: OutlierItem,
    state: ClusterState,
    llm,
    model: str,
) -> str:
    """
    Promote a buffered item + incoming task into a new real cluster.

    Uses both texts to generalize the leader, and sets the initial weight
    to the sum of the matched buffer item's weight and 1 (for the new task).
    """
    combined_texts = [matched.task.text, task.text]
    if llm is not None:
        leader = llm_ops.update_leader(
            leader=matched.task.text,
            member_texts=combined_texts,
            llm=llm,
            model=model,
        )
    else:
        leader = matched.task.text

    cluster_id = str(uuid.uuid4())
    cluster = Cluster(
        id=cluster_id,
        leader=leader,
        members=[matched.task.id, task.id],
        weight=matched.weight + 1.0,
        last_updated=task.timestamp,
        created_at=task.timestamp,
        parent_id=None,
        children=[],
        level=0,
        anchored=False,
    )
    state.clusters[cluster_id] = cluster
    return cluster_id


def _check_buffer(
    task: TaskItem,
    state: ClusterState,
    llm,
    model: str,
    return_reasoning: bool,
    reasonings: list[str],
) -> str | None:
    """
    Check whether `task` matches any item in the outlier buffer.

    If a match is found: remove that item from the buffer and promote both
    to a new cluster.  Returns the new cluster_id.

    If no match: add `task` to the buffer and return None.
    """
    buf = state.outlier_buffer

    if not buf:
        state.outlier_buffer.append(OutlierItem(task=task, weight=1.0))
        if return_reasoning:
            reasonings.append("[buffer] Buffer empty — added to outlier buffer.")
        return None

    buffer_leaders = [item.task.text for item in buf]
    result = llm_ops.assign_task(
        task_text=task.text,
        leaders=buffer_leaders,
        llm=llm,
        criteria=state.criteria,
        model=model,
        allow_none=True,
        return_reasoning=return_reasoning,
    )
    if return_reasoning:
        choice, reason = result
        reasonings.append(f"[buffer] {reason}")
    else:
        choice = result

    if choice == "none":
        state.outlier_buffer.append(OutlierItem(task=task, weight=1.0))
        return None

    matched = buf.pop(int(choice) - 1)
    cid = _promote_from_buffer(task, matched, state, llm, model)
    if return_reasoning:
        reasonings.append(
            f"[buffer] Promoted '{matched.task.text[:60]}' + new task → cluster {cid[:8]}"
        )
    return cid


def process_task(
    task: TaskItem,
    state: ClusterState,
    llm,
    model: str = "gpt-4.1",
    return_reasoning: bool = False,
) -> str | tuple[str, list[str]]:
    """
    Route a task through the two-level hierarchy and assign it.

    Returns the cluster_id, or (cluster_id, [reasoning strings]) if return_reasoning=True.
    """
    state.items[task.id] = task
    reasonings: list[str] = []

    roots = [c for c in state.clusters.values() if c.parent_id is None]

    # No established clusters yet — go straight to buffer
    if not roots:
        cid = _check_buffer(task, state, llm, model, return_reasoning, reasonings)
        return (cid, reasonings) if return_reasoning else cid

    # --- Level 0: pick a root ---
    root_leaders = [(c.id, c.leader) for c in roots]
    result = llm_ops.assign_task(
        task_text=task.text,
        leaders=[l for _, l in root_leaders],
        llm=llm,
        criteria=state.criteria,
        model=model,
        allow_none=False,
        return_reasoning=return_reasoning,
    )
    if return_reasoning:
        choice, reason = result
        reasonings.append(f"[L0] {reason}")
    else:
        choice = result

    if choice == "new":
        cid = _check_buffer(task, state, llm, model, return_reasoning, reasonings)
        return (cid, reasonings) if return_reasoning else cid

    root_id = root_leaders[int(choice) - 1][0]
    root = state.clusters[root_id]
    children = [state.clusters[cid] for cid in root.children if cid in state.clusters]

    # Root has no children yet — assign directly
    if not children:
        cid = _assign_to_cluster(task, root, state, llm, model)
        return (cid, reasonings) if return_reasoning else cid

    # --- Level 1: try to fit a child, or stay at root ---
    child_leaders = [(c.id, c.leader) for c in children]
    result2 = llm_ops.assign_task(
        task_text=task.text,
        leaders=[l for _, l in child_leaders],
        llm=llm,
        criteria=state.criteria,
        model=model,
        allow_none=True,
        return_reasoning=return_reasoning,
    )
    if return_reasoning:
        choice2, reason2 = result2
        reasonings.append(f"[L1] {reason2}")
    else:
        choice2 = result2

    if choice2 == "none":
        cid = _assign_to_cluster(task, root, state, llm, model)
    else:
        child_id = child_leaders[int(choice2) - 1][0]
        cid = _assign_to_cluster(task, state.clusters[child_id], state, llm, model)

    return (cid, reasonings) if return_reasoning else cid
