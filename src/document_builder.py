from src.bug_report import BugReport
from src.config import MAX_DESCRIPTION_LENGTH, MAX_ERROR_LOG_LENGTH


def build_retrieval_text(bug: BugReport) -> str:
    """Compose the text that gets embedded.

    Index time and query time must both call this. When the template was inlined in
    two places it drifted, and query vectors landed in a different region of the
    space than the documents they were meant to match.

    Error Log precedes Description because the embedding model truncates the tail at
    512 tokens without warning, and roughly a third of these documents exceed that.
    The error signature is the shortest, highest-signal field, so it goes first.
    """
    error_log = (bug.error_log or "")[:MAX_ERROR_LOG_LENGTH]
    description = (bug.description or "")[:MAX_DESCRIPTION_LENGTH]

    return (
        f"Title: {bug.title}. Error Log: {error_log}. Description: {description}"
    ).strip()


def is_indexable(bug: BugReport) -> bool:
    """A bug with neither title nor description embeds to a degenerate vector.

    597 Firefox rows have a NaN description. Indexed, they all land in nearly the same
    spot and pollute top-k for unrelated queries.
    """
    return bool((bug.title or "").strip() or (bug.description or "").strip())
