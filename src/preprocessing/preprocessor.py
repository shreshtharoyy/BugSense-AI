from src.bug_report import BugReport
from src.preprocessing.text_cleaner import TextCleaner

class Preprocessor:
    @staticmethod
    def process(bug_report: BugReport) -> BugReport:
        bug_report.description = TextCleaner.clean_text(bug_report.description)
        bug_report.error_log = TextCleaner.clean_text(bug_report.error_log)

        return bug_report