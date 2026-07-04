from datetime import datetime
from src.bug_factory import BugFactory
from src.bug_report import BugReport

class DatasetMapper:
    @staticmethod
    def map_row(row: dict, project: str)-> BugReport:
        created_at = (datetime.fromisoformat(row["Created"])
                      if row.get("Created")
                      else None
                      )
        
        resolved_at = (datetime.fromisoformat(row["Resolved"])
                      if row.get("Resolved")
                      else None
                      )
        
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