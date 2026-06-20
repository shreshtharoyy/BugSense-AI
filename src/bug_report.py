from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

@dataclass
class BugReport:
    bug_id: str
    screenshot_path: Path | None
    description: str
    error_log: str
    created_at: datetime