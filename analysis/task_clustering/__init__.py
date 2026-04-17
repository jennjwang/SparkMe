"""
analysis.task_clustering — general-purpose online task clustering.

Algorithms:
  - Leader algorithm with LLM similarity (leader.py)
  - DenStream fading and pruning (denstream.py)
  - Hierarchical Divisive Clustering (divisive.py)

Entry points:
  - OnlineClusteringPipeline (pipeline.py)
  - CLI: python -m analysis.task_clustering.cli --help
"""

from .models import TaskItem, Cluster, ClusterState
from .pipeline import OnlineClusteringPipeline

__all__ = ["TaskItem", "Cluster", "ClusterState", "OnlineClusteringPipeline"]
