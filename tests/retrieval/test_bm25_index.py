from src.retrieval.bm25_index import BM25Index

IDS = ["A", "B", "C"]
DOCS = [
    "login button crashes with NS_ERROR_FAILURE on click",
    "bookmarks fail to sync after signing in",
    "pdf viewer shows a blank page",
]


def test_exact_token_ranks_its_document_first():
    # The whole thesis of hybrid: a rare exact token that dense embeddings would blur,
    # BM25 pins to the one document that contains it.
    index = BM25Index.from_documents(IDS, DOCS)
    assert index.query("crash NS_ERROR_FAILURE", top_n=3)[0] == "A"


def test_query_returns_ids_best_first_and_respects_top_n():
    index = BM25Index.from_documents(IDS, DOCS)
    ranked = index.query("sync bookmarks", top_n=2)
    assert len(ranked) == 2
    assert ranked[0] == "B"


def test_len_matches_corpus():
    assert len(BM25Index.from_documents(IDS, DOCS)) == 3


def test_top_n_larger_than_corpus_is_clamped():
    index = BM25Index.from_documents(IDS, DOCS)
    assert len(index.query("pdf", top_n=99)) == 3
