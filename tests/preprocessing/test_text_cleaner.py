import pytest

from src.bug_factory import BugFactory
from src.preprocessing.preprocessor import Preprocessor
from src.preprocessing.text_cleaner import TextCleaner


def test_collapses_whitespace():
    assert TextCleaner.clean_text("  Login   button\n\n crashes  ") == (
        "Login button crashes"
    )


def test_strips_zero_width_characters():
    assert TextCleaner.clean_text(f"a{chr(0x200B)}b{chr(0xFEFF)}c") == "abc"


@pytest.mark.parametrize("value", [None, float("nan"), 42])
def test_non_string_input_returns_empty(value):
    # pandas yields float('nan') for empty cells; .strip() used to raise AttributeError.
    assert TextCleaner.clean_text(value) == ""


def test_preprocessor_cleans_title(bug):
    dirty = BugFactory.create(
        title="  Crash   on\nlogin ", description="d", error_log="e"
    )
    assert Preprocessor.process(dirty).title == "Crash on login"


def test_preprocessor_does_not_mutate_input(bug):
    original_description = bug.description
    processed = Preprocessor.process(bug)

    assert bug.description == original_description
    assert processed is not bug
