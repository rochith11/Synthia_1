"""Audit logger — append-only file-based audit trail for user actions."""

import json
import os
from datetime import datetime, timezone
from typing import Optional


class AuditLogger:
    """Log all user actions (create, export, delete) to a JSON-lines file."""

    def __init__(self, log_path: str = "data/audit_log.jsonl"):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    def log(
        self,
        action: str,
        resource_id: Optional[str] = None,
        username: str = "default_user",
        details: Optional[dict] = None,
    ) -> None:
        """Append an audit entry.

        Args:
            action: Action performed (e.g. 'create', 'export', 'delete').
            resource_id: ID of the affected resource (dataset UUID, etc.).
            username: Who performed the action.
            details: Extra information about the action.
        """
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "resource_id": resource_id,
            "username": username,
            "details": details or {},
        }

        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def get_entries(self, limit: int = 100) -> list:
        """Read the most recent audit entries.

        Args:
            limit: Maximum number of entries to return.

        Returns:
            List of audit entry dicts (most recent last).
        """
        if not os.path.exists(self.log_path):
            return []

        entries = []
        with open(self.log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))

        return entries[-limit:]
