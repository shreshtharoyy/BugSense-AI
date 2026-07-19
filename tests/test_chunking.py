from src.chunking import chunk_id, chunk_retrieval_text, parent_id_from_chunk_id
from src.config import MAX_CHUNKS_PER_BUG
from src.document_builder import build_retrieval_text
from src.bug_factory import BugFactory


class _FakeTokenizer:
    """Whitespace tokenizer so chunk tests do not load the real BGE model."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(f"w{i}" for i in token_ids)


def test_chunk_id_format():
    assert chunk_id("1606532", 0) == "1606532::0"
    assert parent_id_from_chunk_id("1606532::3") == "1606532"
    assert parent_id_from_chunk_id("1606532") == "1606532"


def test_short_text_stays_one_chunk():
    text = "Title: t. Error Log: e. Description: short body"
    chunks = chunk_retrieval_text(text, _FakeTokenizer(), chunk_tokens=64, overlap=8)
    assert chunks == [text]


def test_long_description_caps_chunks_and_keeps_error_on_first_only():
    body = " ".join(f"word{i}" for i in range(400))
    text = f"Title: crash. Error Log: NS_ERROR. Description: {body}"
    chunks = chunk_retrieval_text(
        text, _FakeTokenizer(), chunk_tokens=40, overlap=5, max_chunks=4
    )

    assert 1 < len(chunks) <= 4
    assert "Error Log: NS_ERROR" in chunks[0]
    for later in chunks[1:]:
        assert "Error Log:" not in later
        assert later.startswith("Title: crash. Description: ")


def test_max_chunks_hard_cap():
    body = " ".join(f"word{i}" for i in range(2000))
    text = f"Title: t. Error Log: e. Description: {body}"
    chunks = chunk_retrieval_text(
        text, _FakeTokenizer(), chunk_tokens=30, overlap=2, max_chunks=3
    )
    assert len(chunks) <= 3


def test_real_builder_output_respects_cap(embedder):
    bug = BugFactory.create(
        title="Long bug",
        description=("failed assertion in layout " * 400),
        error_log="NS_ERROR_FAILURE",
    )
    text = build_retrieval_text(bug)
    chunks = chunk_retrieval_text(text, embedder.tokenizer)
    assert 1 <= len(chunks) <= MAX_CHUNKS_PER_BUG
    assert "Error Log:" in chunks[0]
    assert all(chunk.startswith("Title:") for chunk in chunks)
