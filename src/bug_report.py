from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass(frozen=True)
class BugReport:
    bug_id: str
    title: str
    screenshot_path: Path | None
    description: str
    error_log: str
    created_at: datetime
    resolved_at: datetime | None = None
    project: str = ""
    resolution: str | None = None
    status: str = ""
    priority: str = ""
