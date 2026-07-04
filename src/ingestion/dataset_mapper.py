from datetime import datetime
from src.bug_factory import BugFactory
from src.bug_report import BugReport
import pandas as pd

class DatasetMapper:
    @staticmethod
    def map_row(row: dict, project: str)-> BugReport:
        created_at = datetime.fromisoformat(str(row["Created"]))
        
        resolved_value = row["Resolved"]

        if pd.notna(resolved_value):
            resolved_at = datetime.fromisoformat(str(resolved_value))
        else:
            resolved_at = None
        
        return BugFactory.create(
            bug_id=str(row["Issue id"]),
            title=row["Summary"],
            description=row["Description"],
            error_log=row["Description"],      
            created_at=created_at,
            resolved_at=resolved_at,
            resolution=row.get("Resolution"),
            project=project,
        )