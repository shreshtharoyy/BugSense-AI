import re

class TextCleaner:
    @staticmethod
    def clean_text(text: str) -> str:
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)
        return text