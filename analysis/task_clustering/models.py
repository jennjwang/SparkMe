"""
Data structures for the online task clustering pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TaskItem:
    """A single task observation ingested into the pipeline."""
    id: str
    text: str
    source: str          # participant / session id
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TaskItem":
        return cls(
            id=d["id"],
            text=d["text"],
            source=d.get("source", ""),
            timestamp=datetime.fromisoformat(d["timestamp"]),
            metadata=d.get("metadata", {}),
        )


@dataclass
class Cluster:
    """
    A cluster of semantically similar task items.

    Hierarchy: parent_id=None → top-level cluster (broadest).
    Children are set when divisive splitting occurs; the cluster then becomes
    an internal node and new items are routed to its leaves.

    Reading the tree bottom-up:  leaves = specific subtasks,
    internal nodes = broader task groupings, root(s) = broadest categories.
    """
    id: str
    leader: str              # canonical statement representing the cluster center
    members: list[str]       # ordered list of TaskItem ids
    weight: float            # DenStream weight; incremented on assignment, fades over time
    last_updated: datetime
    created_at: datetime
    parent_id: Optional[str]  # None = root
    children: list[str]       # child cluster ids (non-empty → internal node)
    level: int                # 0 = root level, deeper = more specific
    anchored: bool = False    # if True, DenStream never prunes (used for taxonomy seeds)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "leader": self.leader,
            "members": self.members,
            "weight": self.weight,
            "last_updated": self.last_updated.isoformat(),
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
            "children": self.children,
            "level": self.level,
            "anchored": self.anchored,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Cluster":
        return cls(
            id=d["id"],
            leader=d["leader"],
            members=d["members"],
            weight=d["weight"],
            last_updated=datetime.fromisoformat(d["last_updated"]),
            created_at=datetime.fromisoformat(d["created_at"]),
            parent_id=d.get("parent_id"),
            children=d.get("children", []),
            level=d.get("level", 0),
            anchored=d.get("anchored", False),
        )


@dataclass
class OutlierItem:
    """
    A task item held in the outlier buffer — not yet part of any cluster.

    An item is promoted to a real cluster only when a second item arrives
    that the LLM judges to match it.  Items that never accumulate support
    decay and are pruned by DenStream, same as weak clusters.
    """
    task: TaskItem
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {"task": self.task.to_dict(), "weight": self.weight}

    @classmethod
    def from_dict(cls, d: dict) -> "OutlierItem":
        return cls(task=TaskItem.from_dict(d["task"]), weight=d["weight"])


DEFAULT_CRITERIA = [
    "Same core action verb and object domain",
    "Would be described as the same task by a job analyst",
    "Could appear as the same bullet point in a job description",
]


@dataclass
class ClusterState:
    """Full mutable state of the pipeline — serialisable to JSON."""
    clusters: dict[str, Cluster]    # cluster_id → Cluster
    items: dict[str, TaskItem]      # item_id → TaskItem

    # Algorithm parameters
    lambda_: float        # DenStream fading factor (e.g. 0.1 per hour)
    eps: float            # prune threshold weight (e.g. 0.5)
    max_split_size: int   # divisive split trigger (cluster member count)
    split_depth_limit: int  # max hierarchy depth allowed
    criteria: list[str]   # similarity criteria injected into assign_task prompt

    # Runtime counters
    total_processed: int
    last_fade_count: int   # total_processed value at the last fade pass

    # Outlier buffer: items waiting for a second match before becoming a cluster
    outlier_buffer: list = field(default_factory=list)  # list[OutlierItem]

    def to_dict(self) -> dict:
        return {
            "clusters": {k: v.to_dict() for k, v in self.clusters.items()},
            "items": {k: v.to_dict() for k, v in self.items.items()},
            "lambda_": self.lambda_,
            "eps": self.eps,
            "max_split_size": self.max_split_size,
            "split_depth_limit": self.split_depth_limit,
            "criteria": self.criteria,
            "total_processed": self.total_processed,
            "last_fade_count": self.last_fade_count,
            "outlier_buffer": [item.to_dict() for item in self.outlier_buffer],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ClusterState":
        return cls(
            clusters={k: Cluster.from_dict(v) for k, v in d["clusters"].items()},
            items={k: TaskItem.from_dict(v) for k, v in d["items"].items()},
            lambda_=d.get("lambda_", 0.1),
            eps=d.get("eps", 0.5),
            max_split_size=d.get("max_split_size", 10),
            split_depth_limit=d.get("split_depth_limit", 3),
            criteria=d.get("criteria", DEFAULT_CRITERIA),
            total_processed=d.get("total_processed", 0),
            last_fade_count=d.get("last_fade_count", 0),
            outlier_buffer=[OutlierItem.from_dict(i) for i in d.get("outlier_buffer", [])],
        )

    @classmethod
    def new(
        cls,
        lambda_: float = 0.1,
        eps: float = 0.5,
        max_split_size: int = 10,
        split_depth_limit: int = 3,
        criteria: Optional[list[str]] = None,
    ) -> "ClusterState":
        return cls(
            clusters={},
            items={},
            lambda_=lambda_,
            eps=eps,
            max_split_size=max_split_size,
            split_depth_limit=split_depth_limit,
            criteria=criteria if criteria is not None else list(DEFAULT_CRITERIA),
            total_processed=0,
            last_fade_count=0,
        )
