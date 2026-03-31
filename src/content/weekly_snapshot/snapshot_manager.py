import os
import json
from datetime import datetime
from typing import List, Optional

from src.content.weekly_snapshot.weekly_snapshot import WeeklySnapshot


class SnapshotManager:
    """Persists and retrieves weekly snapshots for a user.

    Snapshots are stored at:
        {LOGS_DIR}/{user_id}/weekly_snapshots/snapshot_week_{week_number}.json

    The latest snapshot is always the one with the highest week_number.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self._dir = os.path.join(
            os.getenv("LOGS_DIR", "logs"),
            user_id,
            "weekly_snapshots",
        )

    def _ensure_dir(self) -> None:
        os.makedirs(self._dir, exist_ok=True)

    def _path_for_week(self, week_number: int) -> str:
        return os.path.join(self._dir, f"snapshot_week_{week_number}.json")

    def save_snapshot(self, snapshot: WeeklySnapshot) -> str:
        """Save snapshot to disk, overwriting any existing file for that week.

        Returns the path the snapshot was written to.
        """
        self._ensure_dir()
        path = self._path_for_week(snapshot.week_number)
        with open(path, "w") as f:
            json.dump(snapshot.to_dict(), f, indent=2)
        return path

    def load_snapshot_by_week(self, week_number: int) -> Optional[WeeklySnapshot]:
        """Load the snapshot for a specific ISO week number, or None if it doesn't exist."""
        path = self._path_for_week(week_number)
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return WeeklySnapshot.from_dict(json.load(f))

    def load_latest_snapshot(self) -> Optional[WeeklySnapshot]:
        """Load the most recent snapshot by week number, or None if none exist."""
        if not os.path.isdir(self._dir):
            return None
        files = [
            f for f in os.listdir(self._dir)
            if f.startswith("snapshot_week_") and f.endswith(".json")
        ]
        if not files:
            return None
        # Extract week numbers and find max
        def _week_num(filename: str) -> int:
            return int(filename.replace("snapshot_week_", "").replace(".json", ""))

        latest_file = max(files, key=_week_num)
        with open(os.path.join(self._dir, latest_file)) as f:
            return WeeklySnapshot.from_dict(json.load(f))

    def get_all_snapshots(self) -> List[WeeklySnapshot]:
        """Return all snapshots sorted by week number (ascending)."""
        if not os.path.isdir(self._dir):
            return []
        snapshots = []
        for filename in sorted(os.listdir(self._dir)):
            if filename.startswith("snapshot_week_") and filename.endswith(".json"):
                with open(os.path.join(self._dir, filename)) as f:
                    snapshots.append(WeeklySnapshot.from_dict(json.load(f)))
        return snapshots

    @staticmethod
    def current_week_number() -> int:
        """Return the current ISO week number."""
        return datetime.utcnow().isocalendar()[1]
