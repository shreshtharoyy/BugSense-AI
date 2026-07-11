import pytest

from src.preprocessing.log_extractor import LogExtractor


@pytest.mark.parametrize(
    "text, expected",
    [
        ("got NS_ERROR_FAILURE here", "NS_ERROR_FAILURE"),
        (
            "console shows TypeError: info.PDFFormatVersion is undefined",
            "TypeError: info.PDFFormatVersion is undefined",
        ),
        ("trace: at LoginService.java:45", "at LoginService.java:45"),
        ("failed with 0x80040111", "0x80040111"),
        ("crash was SIGSEGV", "SIGSEGV"),
    ],
)
def test_extracts_known_signatures(text, expected):
    assert expected in LogExtractor.extract(text)


def test_clock_times_are_not_stack_frames():
    # `\bat\s+\S+[:(]\d+` matched "at 10:51:48" and swamped the real signal.
    assert LogExtractor.extract("it happened at 10:51:48 today") == ""


def test_raw_stack_addresses_are_ignored():
    # A crash dump of bare addresses used to consume the whole length budget.
    dump = "0x0026eca8 0x5b0b5281 0x6a895c08 0x5eca7e68"
    assert LogExtractor.extract(dump) == ""


def test_hresult_style_codes_are_kept():
    assert LogExtractor.extract("0x80004005") == "0x80004005"


def test_output_is_deduplicated():
    result = LogExtractor.extract("NS_ERROR_FAILURE and again NS_ERROR_FAILURE")
    assert result == "NS_ERROR_FAILURE"


def test_respects_max_length():
    text = " ".join(f"NS_ERROR_CODE_{i}" for i in range(200))
    assert len(LogExtractor.extract(text, max_length=50)) <= 50


@pytest.mark.parametrize("value", [None, float("nan"), "", 42])
def test_non_text_input_returns_empty(value):
    assert LogExtractor.extract(value) == ""
