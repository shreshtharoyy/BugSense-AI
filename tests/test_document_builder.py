from src.bug_factory import BugFactory
from src.config import MAX_DESCRIPTION_LENGTH, MAX_ERROR_LOG_LENGTH
from src.document_builder import build_retrieval_text, is_indexable


def test_error_log_precedes_description(bug):
    text = build_retrieval_text(bug)
    # The embedder truncates the tail at 512 tokens and ~38% of real documents exceed
    # that. The error signature must sit ahead of the long free-text description.
    assert text.index("Error Log:") < text.index("Description:")


def test_fields_are_truncated_to_their_budgets():
    bug = BugFactory.create(
        title="t", description="d" * 5000, error_log="e" * 5000
    )
    text = build_retrieval_text(bug)

    assert "d" * MAX_DESCRIPTION_LENGTH in text
    assert "d" * (MAX_DESCRIPTION_LENGTH + 1) not in text
    assert "e" * MAX_ERROR_LOG_LENGTH in text
    assert "e" * (MAX_ERROR_LOG_LENGTH + 1) not in text


def test_bug_with_title_only_is_indexable():
    assert is_indexable(BugFactory.create(title="t", description="", error_log=""))


def test_bug_with_neither_title_nor_description_is_not_indexable():
    # 597 Firefox rows have a NaN description; indexed they form a degenerate cluster.
    assert not is_indexable(BugFactory.create(title="  ", description="", error_log=""))
