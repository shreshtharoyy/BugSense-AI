from datetime import datetime

import pandas as pd

from src.bug_factory import BugFactory
from src.bug_report import BugReport
from src.preprocessing.log_extractor import LogExtractor
from src.preprocessing.text_cleaner import TextCleaner


class RowMappingError(ValueError):
    """A single CSV row could not be mapped. The caller counts it and moves on."""


def _optional_str(value) -> str | None:
    return str(value) if pd.notna(value) else None


def _parse_datetime(value, field: str) -> datetime:
    if pd.isna(value):
        raise RowMappingError(f"{field} is empty")
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise RowMappingError(f"{field} is not an ISO timestamp: {value!r}") from exc


def _normalize_id(value) -> str:
    # pandas types an integer id column as int64 or float64. str() on the float form
    # yields "1606532.0", which silently forks the id space against the CSV.
    if pd.isna(value):
        raise RowMappingError("Issue id is empty")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


class DatasetMapper:
    @staticmethod
    def map_row(row: dict, project: str) -> BugReport:
        try:
            bug_id = _normalize_id(row["Issue id"])
            created_at = _parse_datetime(row["Created"], "Created")

            resolved_value = row.get("Resolved")
            resolved_at = (
                _parse_datetime(resolved_value, "Resolved")
                if pd.notna(resolved_value)
                else None
            )

            description = TextCleaner.clean_text(row.get("Description"))
            title = TextCleaner.clean_text(row.get("Summary"))
        except KeyError as exc:
            raise RowMappingError(f"missing column {exc}") from exc

        return BugFactory.create(
            bug_id=bug_id,
            title=title,
            description=description,
            # The dataset has no error-log column. Recover the signature from the
            # description rather than embedding the description twice.
            error_log=LogExtractor.extract(description),
            created_at=created_at,
            resolved_at=resolved_at,
            resolution=_optional_str(row.get("Resolution")),
            status=TextCleaner.clean_text(row.get("Status")),
            priority=TextCleaner.clean_text(row.get("Priority")),
            project=project,
        )
