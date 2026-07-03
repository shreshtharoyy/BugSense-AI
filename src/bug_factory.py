from datetime import datetime
from pathlib import Path
from uuid import uuid4
from src.bug_report import BugReport

class BugFactory:
    @staticmethod
    def create(
        title: str,
        description: str,
        error_log: str,
        screenshot_path: Path | None = None,
        bug_id: str | None = None,
        created_at: datetime | None = None,
        resolved_at: datetime | None = None,
        resolution: str | None = None,
        project: str = "",
    ) -> BugReport:
        
        if bug_id is None:
            bug_id = f"BUG-{uuid4().hex[:8].upper()}"

        if created_at is None:
            created_at = datetime.now()

        return BugReport(
            bug_id=bug_id,
            screenshot_path=screenshot_path,
            description=description,
            error_log=error_log,
            created_at=created_at,
            title=title,
            project=project,
            resolution=resolution,
            resolved_at=resolved_at
        )