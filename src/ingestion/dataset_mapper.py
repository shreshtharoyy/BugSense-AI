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

        resolution_value = row["Resolution"]

        if pd.notna(resolution_value):
            resolution = str(resolution_value)
        else:
            resolution = None

        description = (
            str(row["Description"])
            if pd.notna(row["Description"])
            else ""
        )
       
        return BugFactory.create(
            bug_id=str(row["Issue id"]),
            title=row["Summary"],
            description=description,
            error_log=description,      
            created_at=created_at,
            resolved_at=resolved_at,
            resolution=resolution,
            project=project,
        )