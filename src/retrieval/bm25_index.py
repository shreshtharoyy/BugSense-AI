from __future__ import annotations

import bm25s


class BM25Index:
    """In-memory BM25 over the same documents held in Chroma.

    Dense embeddings blur rare exact tokens (NS_ERROR_FAILURE, error codes, API names);
    BM25 ranks precisely on them. Built from the corpus at startup, so it never drifts
    from the vector index and needs no separate on-disk artifact. Over ~25k short docs
    this indexes in a second or two.
    """

    def __init__(self, ids: list[str], bm25: bm25s.BM25) -> None:
        self._ids = ids
        self._bm25 = bm25

    @classmethod
    def from_documents(cls, ids: list[str], documents: list[str]) -> BM25Index:
        tokens = bm25s.tokenize(documents, show_progress=False)
        bm25 = bm25s.BM25()
        bm25.index(tokens, show_progress=False)
        return cls(list(ids), bm25)

    def __len__(self) -> int:
        return len(self._ids)

    def query(self, text: str, top_n: int) -> list[str]:
        """Return bug ids ranked best-first. RRF needs the order, not the raw scores."""
        top_n = min(top_n, len(self._ids))
        if top_n <= 0:
            return []
        # Same tokenizer as the corpus, so exact tokens line up on both sides.
        query_tokens = bm25s.tokenize([text], show_progress=False)
        results, _ = self._bm25.retrieve(query_tokens, k=top_n, show_progress=False)
        return [self._ids[int(row_index)] for row_index in results[0]]
