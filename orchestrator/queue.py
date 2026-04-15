"""Simple file-based FIFO queue for submission IDs."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


class FileQueue:
    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def enqueue(self, submission_id: int) -> None:
        ts = time.time_ns()
        path = self.dir / f"{ts:020d}_{submission_id}.json"
        path.write_text(json.dumps({"submission_id": submission_id, "ts": ts}))

    def dequeue(self) -> Optional[int]:
        items = sorted(self.dir.glob("*.json"))
        if not items:
            return None
        item = items[0]
        try:
            data = json.loads(item.read_text())
        except json.JSONDecodeError:
            item.unlink(missing_ok=True)
            return None
        try:
            os.remove(item)
        except FileNotFoundError:
            return None
        return int(data["submission_id"])

    def size(self) -> int:
        return len(list(self.dir.glob("*.json")))
