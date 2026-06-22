from datetime import datetime
from pathlib import Path
from uuid import uuid4
from src.bug_report import BugReport

class BugFactory:
    @staticmethod
    def create(
        screenshot_path: Path | None,
        description: str,
        error_log: str,
    ) -> BugReport:
        
        bug_id = f"BUG-{uuid4().hex[:8].upper()}"

        return BugReport(
            bug_id=bug_id,
            screenshot_path=screenshot_path,
            description=description,
            error_log=error_log,
            created_at=datetime.now(),
        )