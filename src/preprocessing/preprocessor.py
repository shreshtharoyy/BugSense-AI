from dataclasses import replace

from src.bug_report import BugReport
from src.preprocessing.text_cleaner import TextCleaner


class Preprocessor:
    @staticmethod
    def process(bug_report: BugReport) -> BugReport:
        # A copy, not a mutation: BugReport is frozen so the caller's object cannot
        # change underneath it.
        return replace(
            bug_report,
            title=TextCleaner.clean_text(bug_report.title),
            description=TextCleaner.clean_text(bug_report.description),
            error_log=TextCleaner.clean_text(bug_report.error_log),
        )
