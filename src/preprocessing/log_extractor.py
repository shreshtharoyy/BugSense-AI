import re

from src.config import MAX_ERROR_LOG_LENGTH

# The GitBugs CSVs carry no error-log column, so the error signature has to be
# recovered from the free-text description. These are the shapes that actually
# recur in Bugzilla reports.
PATTERNS = (
    # Exception/error class plus its message.
    re.compile(r"\b[A-Z][A-Za-z0-9_.]*(?:Error|Exception)\b[^\n]{0,200}"),
    re.compile(r"\bNS_ERROR_[A-Z_]+"),
    # Stack frame. The frame token must contain a letter, otherwise clock times
    # ("at 10:51:48") match and swamp the output.
    re.compile(r"\bat\s+[^\s:(]*[A-Za-z][^\s(]*[:(]\d+\)?"),
    re.compile(r"Assertion failure:[^\n]{0,200}"),
    re.compile(r"panicked at[^\n]{0,200}"),
    # nsresult / HRESULT failure codes only. A bare 0x[0-9a-f]{8} also matches raw
    # stack addresses, and a single crash dump then fills the whole budget.
    re.compile(r"\b0x8[0-9A-Fa-f]{7}\b"),
    re.compile(r"\b(?:SIGSEGV|SIGABRT|EXCEPTION_ACCESS_VIOLATION)\b"),
)

MAX_MATCHES_PER_PATTERN = 5


class LogExtractor:
    @staticmethod
    def extract(text: str, max_length: int = MAX_ERROR_LOG_LENGTH) -> str:
        if not isinstance(text, str) or not text:
            return ""

        matches: list[str] = []
        for pattern in PATTERNS:
            kept = 0
            for match in pattern.findall(text):
                if kept >= MAX_MATCHES_PER_PATTERN:
                    break
                match = " ".join(match.split())
                if match and match not in matches:
                    matches.append(match)
                    kept += 1

        return " | ".join(matches)[:max_length]
