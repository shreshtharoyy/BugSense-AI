import re

# Zero-width space, ZWNJ, ZWJ, BOM.
ZERO_WIDTH = re.compile(r"[\u200B-\u200D\uFEFF]")
WHITESPACE = re.compile(r"\s+")


class TextCleaner:
    @staticmethod
    def clean_text(text) -> str:
        # pandas yields float('nan') for empty cells, which has no .strip().
        if not isinstance(text, str):
            return ""
        text = ZERO_WIDTH.sub("", text)
        text = WHITESPACE.sub(" ", text)
        return text.strip()
