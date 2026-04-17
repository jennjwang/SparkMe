"""
OnlineClusteringPipeline — orchestrates Leader + DenStream + Divisive.

Usage
-----
from analysis.task_clustering.pipeline import OnlineClusteringPipeline
from analysis.task_clustering.models import TaskItem
from datetime import datetime

pipe = OnlineClusteringPipeline(max_split_size=8)
pipe.load_taxonomy("my_taxonomy.json")

for task in tasks:
    pipe.ingest(TaskItem(id=..., text=..., source=..., timestamp=datetime.now()))

pipe.save("clusters.json")
hierarchy = pipe.get_hierarchy()
"""

from __future__ import annotations
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import Cluster, ClusterState, TaskItem, DEFAULT_CRITERIA
from . import leader as leader_mod
from . import denstream
from . import divisive

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dataset_gen"))
from llm_client import LLMClient

# Initial weight given to taxonomy seed clusters
_TAXONOMY_SEED_WEIGHT = 1.0


class OnlineClusteringPipeline:
    """
    General-purpose online task clustering pipeline.

    Three algorithm layers:
      1. Leader algorithm  — assigns each new item to a leaf cluster (or creates one)
      2. DenStream fading  — cluster weights decay; old unused clusters are pruned
      3. Hierarchical Divisive — large/heterogeneous clusters are recursively split
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        eps: float = 0.5,
        max_split_size: int = 10,
        split_depth_limit: int = 3,
        fade_interval_interviews: int = 10,
        criteria: Optional[list[str]] = None,
        model: str = "gpt-4.1",
    ):
        """
        Parameters
        ----------
        lambda_ : float
            DenStream fading factor. Higher = faster decay.
            E.g. 0.1 means weight halves every ~10 hours.
        eps : float
            Prune threshold. Leaf clusters with weight < eps are removed.
        max_split_size : int
            Divisive split is triggered when a leaf cluster reaches this many members.
        split_depth_limit : int
            Maximum depth of the cluster hierarchy (0 = no splits allowed).
        fade_interval_hours : float
            How often (wall-clock hours) fading is applied.
        criteria : list[str] | None
            Custom similarity criteria injected into the LLM assign prompt.
        model : str
            LLM model name passed to all LLM ops.
        """
        self.state = ClusterState.new(
            lambda_=lambda_,
            eps=eps,
            max_split_size=max_split_size,
            split_depth_limit=split_depth_limit,
            criteria=criteria if criteria is not None else list(DEFAULT_CRITERIA),
        )
        self.llm = LLMClient()
        self.fade_interval_interviews = fade_interval_interviews
        self.model = model

    # ------------------------------------------------------------------
    # Taxonomy warmup
    # ------------------------------------------------------------------

    def load_taxonomy(self, taxonomy: list[dict] | str | Path) -> int:
        """
        Pre-seed the pipeline with a known taxonomy before ingesting items.

        Taxonomy format (list of dicts):
            [
              {"id": "code-review", "label": "Code Review",
               "description": "Review code changes...", "parent": null},
              {"id": "pr-review", "label": "PR Review",
               "description": "Review pull requests on GitHub.", "parent": "code-review"}
            ]

        - `description` is used as the cluster leader; falls back to `label`.
        - `parent` is an id string referencing another entry; None = root.
        - All taxonomy clusters are marked `anchored=True` (immune to DenStream pruning).

        Parameters
        ----------
        taxonomy : list[dict] | str | Path
            The taxonomy as a list of dicts, or a path to a JSON file.

        Returns
        -------
        int
            Number of taxonomy nodes loaded.
        """
        if isinstance(taxonomy, (str, Path)):
            with open(taxonomy) as f:
                taxonomy = json.load(f)

        now = datetime.now()

        # First pass: create all clusters
        for node in taxonomy:
            node_id = str(node.get("id", uuid.uuid4()))
            leader_text = node.get("description") or node.get("label", node_id)
            parent_id = node.get("parent")  # may be None or an id string

            cluster = Cluster(
                id=node_id,
                leader=leader_text,
                members=[],
                weight=_TAXONOMY_SEED_WEIGHT,
                last_updated=now,
                created_at=now,
                parent_id=str(parent_id) if parent_id is not None else None,
                children=[],
                level=0,       # resolved in second pass
                anchored=True,
            )
            self.state.clusters[node_id] = cluster

        # Second pass: wire parent→children and compute levels
        for cluster in self.state.clusters.values():
            if cluster.parent_id and cluster.parent_id in self.state.clusters:
                parent = self.state.clusters[cluster.parent_id]
                if cluster.id not in parent.children:
                    parent.children.append(cluster.id)

        # Compute levels via BFS from roots
        roots = [c for c in self.state.clusters.values() if c.parent_id is None]
        queue = [(r, 0) for r in roots]
        while queue:
            node, depth = queue.pop(0)
            node.level = depth
            for child_id in node.children:
                if child_id in self.state.clusters:
                    queue.append((self.state.clusters[child_id], depth + 1))

        return len(taxonomy)

    # ------------------------------------------------------------------
    # Core ingestion
    # ------------------------------------------------------------------

    def ingest(self, task: TaskItem) -> str | None:
        """
        Process a single task: assign → maybe fade → incremental split.

        Returns the cluster_id the task was assigned to, or None if the task
        was added to the outlier buffer (waiting for a second matching item
        before a cluster is created).
        """
        cluster_id = leader_mod.process_task(task, self.state, self.llm, model=self.model)
        self.state.total_processed += 1

        if cluster_id is None:
            print(f"[buffer] '{task.text[:60]}' → outlier buffer "
                  f"(buffer size: {len(self.state.outlier_buffer)})")

        # Fading: apply every fade_interval_interviews ingested items
        since_last_fade = self.state.total_processed - self.state.last_fade_count
        if since_last_fade >= self.fade_interval_interviews:
            n_pruned = denstream.fade_and_prune(self.state)
            if n_pruned:
                print(f"[denstream] Pruned {n_pruned} objects (clusters+buffer, "
                      f"total clusters={len(self.state.clusters)})")

        # Incremental divisive split — only if the item was assigned to a cluster
        if cluster_id is not None:
            new_ids = divisive.try_split(cluster_id, self.state, self.llm, model=self.model)
            if new_ids:
                print(f"[divisive] Split cluster {cluster_id[:8]}… → {len(new_ids)} sub-clusters")

        return cluster_id

    def screen_items(
        self,
        tasks: list[TaskItem],
        batch_size: int = 30,
        verbose: bool = True,
    ) -> tuple[list[TaskItem], list[TaskItem]]:
        """
        Screen tasks for validity before ingestion (same checks as the ONET pipeline).

        Batches calls to avoid prompt overflow. Rewrites salvageable tasks in-place.
        Returns (kept, rejected) where kept includes passed + rewritten items.
        """
        from . import llm_ops
        kept, rejected = [], []

        for start in range(0, len(tasks), batch_size):
            batch = tasks[start:start + batch_size]
            texts = [t.text for t in batch]
            results = llm_ops.screen_tasks(texts, self.llm, model=self.model)

            for r in results:
                idx = r.get("index", 0)
                if idx >= len(batch):
                    continue
                task = batch[idx]
                status = r.get("status", "pass")
                if status == "rejected":
                    if verbose:
                        print(f"  [screen] rejected: {task.text[:70]!r}")
                        print(f"           reason: {r.get('reason', '')}")
                    rejected.append(task)
                else:
                    if status == "rewritten" and r.get("rewritten"):
                        if verbose:
                            print(f"  [screen] rewritten: {task.text[:60]!r}")
                            print(f"           → {r['rewritten'][:60]!r}")
                        task.text = r["rewritten"]
                    kept.append(task)

        if verbose:
            print(f"[screen] {len(kept)} kept ({len(tasks)-len(kept)-len(rejected)} rewritten), "
                  f"{len(rejected)} rejected out of {len(tasks)}")
        return kept, rejected

    def ingest_batch(self, tasks: list[TaskItem], verbose: bool = True) -> dict[str, str]:
        """
        Process a list of tasks sequentially.

        Returns a mapping {task_id → cluster_id}.
        """
        assignments: dict[str, str | None] = {}
        for i, task in enumerate(tasks):
            cluster_id = self.ingest(task)
            assignments[task.id] = cluster_id
            if verbose and (i + 1) % 10 == 0:
                n_leaves = sum(1 for c in self.state.clusters.values() if c.is_leaf)
                print(f"  [{i+1}/{len(tasks)}] {len(self.state.clusters)} clusters "
                      f"({n_leaves} leaves, {len(self.state.outlier_buffer)} buffered)")
        return assignments

    # ------------------------------------------------------------------
    # Manual controls
    # ------------------------------------------------------------------

    def force_fade(self) -> int:
        """Apply DenStream fading and pruning immediately. Returns pruned count."""
        return denstream.fade_and_prune(self.state)  # uses total_processed as current count

    def force_split_check(self) -> list[str]:
        """Trigger divisive split check on all eligible clusters. Returns new cluster ids."""
        return divisive.check_and_split(self.state, self.llm, model=self.model)

    def flush_buffer(self) -> int:
        """Route all remaining outlier buffer items into the nearest existing cluster.

        If established clusters exist, forces each buffer item to assign to the
        closest root (allow_none=False) rather than creating new singletons.
        Falls back to creating a singleton only when there are no established
        clusters at all.  Returns the count of items flushed.
        """
        from . import llm_ops
        flushed = 0
        roots = [c for c in self.state.clusters.values() if c.parent_id is None]
        for item in list(self.state.outlier_buffer):
            if roots:
                root_leaders = [(c.id, c.leader) for c in roots]
                choice = llm_ops.assign_task(
                    task_text=item.task.text,
                    leaders=[l for _, l in root_leaders],
                    llm=self.llm,
                    criteria=self.state.criteria,
                    model=self.model,
                    force=True,
                    return_reasoning=False,
                )
                if choice not in ("new", "none"):
                    root_id = root_leaders[int(choice) - 1][0]
                    leader_mod._assign_to_cluster(
                        item.task, self.state.clusters[root_id], self.state,
                        self.llm, self.model,
                    )
                    flushed += 1
                    continue
            # No clusters or LLM said "new" — create singleton
            leader_mod._new_cluster(
                task=item.task,
                state=self.state,
                parent_id=None,
                level=0,
                llm=self.llm,
                model=self.model,
            )
            flushed += 1
        self.state.outlier_buffer.clear()
        return flushed

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def get_hierarchy(self) -> list[dict]:
        """Return the cluster tree as a nested dict structure (roots first)."""
        return divisive.get_hierarchy(self.state)

    def get_leaf_clusters(self) -> list[Cluster]:
        """Return all current leaf clusters (active assignment targets)."""
        return [c for c in self.state.clusters.values() if c.is_leaf]

    def summary(self) -> dict:
        clusters = list(self.state.clusters.values())
        leaves = [c for c in clusters if c.is_leaf]
        return {
            "total_items": len(self.state.items),
            "total_clusters": len(clusters),
            "leaf_clusters": len(leaves),
            "internal_clusters": len(clusters) - len(leaves),
            "max_depth": max((c.level for c in clusters), default=0),
            "anchored_clusters": sum(1 for c in clusters if c.anchored),
            "total_processed": self.state.total_processed,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize pipeline state to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.state.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "OnlineClusteringPipeline":
        """
        Load a previously saved pipeline state from a JSON file.

        Any keyword args override the saved algorithm parameters.
        """
        with open(path) as f:
            data = json.load(f)

        state = ClusterState.from_dict(data)

        pipe = cls.__new__(cls)
        pipe.state = state
        pipe.llm = LLMClient()
        pipe.fade_interval_interviews = kwargs.get("fade_interval_interviews", 10)
        pipe.model = kwargs.get("model", "gpt-4.1")

        # Allow parameter overrides
        if "lambda_" in kwargs:
            pipe.state.lambda_ = kwargs["lambda_"]
        if "eps" in kwargs:
            pipe.state.eps = kwargs["eps"]
        if "max_split_size" in kwargs:
            pipe.state.max_split_size = kwargs["max_split_size"]
        if "split_depth_limit" in kwargs:
            pipe.state.split_depth_limit = kwargs["split_depth_limit"]
        if "criteria" in kwargs:
            pipe.state.criteria = kwargs["criteria"]

        return pipe
