from datetime import datetime

import pytest

from src.bug_factory import BugFactory
from src.document_builder import build_retrieval_text
from src.indexing.indexing_pipeline import IndexingPipeline
from src.retrieval.retriever import Retriever

CORPUS = [
    ("BUG-1", "Address bar does not elide origins", "The URL bar shows the origin elided from the right."),
    ("BUG-2", "PDF viewer crash", "TypeError: info.PDFFormatVersion is undefined when the header is invalid."),
    ("BUG-3", "Bookmark sync fails", "Sync of bookmarks stops after signing in."),
]


@pytest.fixture
def indexed_store(store, embedder):
    bugs = [
        BugFactory.create(
            bug_id=bug_id,
            title=title,
            description=description,
            error_log="",
            created_at=datetime(2020, 1, 1),
            project="firefox",
        )
        for bug_id, title, description in CORPUS
    ]
    IndexingPipeline(embedder=embedder, store=store).index(iter(bugs))
    return store


def test_indexes_every_bug(indexed_store):
    assert indexed_store.existing_parent_ids() == {"BUG-1", "BUG-2", "BUG-3"}
    # Short corpus bugs are one chunk each.
    assert indexed_store.count() == 3


def test_indexes_long_bug_as_multiple_chunks(store, embedder):
    bugs = [
        BugFactory.create(
            bug_id="LONG",
            title="Overflow",
            description=("layout assert " * 400),
            error_log="NS_ERROR_FAILURE",
            created_at=datetime(2020, 1, 1),
            project="firefox",
        )
    ]
    indexed = IndexingPipeline(embedder=embedder, store=store).index(iter(bugs))

    assert indexed == 1
    assert store.existing_parent_ids() == {"LONG"}
    assert store.count() > 1
    assert all(chunk_id.startswith("LONG::") for chunk_id in store.existing_ids())


def test_skips_bugs_with_no_title_and_no_description(store, embedder):
    bugs = [
        BugFactory.create(bug_id="GOOD", title="real bug", description="d", error_log=""),
        BugFactory.create(bug_id="EMPTY", title="", description="", error_log=""),
    ]
    indexed = IndexingPipeline(embedder=embedder, store=store).index(iter(bugs))

    assert indexed == 1
    assert store.count() == 1
    assert store.existing_parent_ids() == {"GOOD"}


def test_exact_text_retrieves_its_own_bug_first(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder)
    target = BugFactory.create(
        bug_id="BUG-2", title=CORPUS[1][1], description=CORPUS[1][2], error_log=""
    )

    results = retriever.retrieve(build_retrieval_text(target), top_k=3)

    assert results[0].bug_id == "BUG-2"
    assert results[0].similarity > 0.9


def test_similarity_is_a_real_cosine_never_negative(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder)
    # min_similarity=-1 so nothing is filtered and we see every raw score.
    results = retriever.retrieve("url bar truncates domain", top_k=3, min_similarity=-1.0)

    assert len(results) == 3
    for result in results:
        assert -1.0 <= result.similarity <= 1.0
    # Under the old l2 space, unrelated hits scored near zero or negative.
    assert results[0].similarity > 0.5


def test_exclude_ids_removes_the_query_bug(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder)
    query = build_retrieval_text(
        BugFactory.create(bug_id="x", title=CORPUS[1][1], description=CORPUS[1][2], error_log="")
    )

    results = retriever.retrieve(query, top_k=2, exclude_ids=["BUG-2"], min_similarity=-1.0)

    assert "BUG-2" not in [result.bug_id for result in results]
    assert len(results) == 2


def test_min_similarity_filters_weak_matches(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder)
    assert retriever.retrieve("quantum chromodynamics lattice gauge", min_similarity=0.95) == []


def test_metadata_round_trips_through_retrieval(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder)
    result = retriever.retrieve("bookmark sync", top_k=1, min_similarity=-1.0)[0]

    assert result.project == "firefox"
    assert result.created_at == "2020-01-01T00:00:00"
    assert result.title


class _StubReranker:
    """Records the pool it was handed and reverses it, so the test can prove both that
    the reranker receives the dense candidates and that its order is what comes back.
    Avoids downloading a real cross-encoder in unit tests."""

    def __init__(self):
        self.received: list | None = None

    def rerank(self, query_text, candidates, top_k):
        self.received = candidates
        return list(reversed(candidates))[:top_k]


def test_reranker_receives_dense_pool_and_owns_final_order(indexed_store, embedder):
    dense = Retriever(indexed_store, embedder=embedder).retrieve(
        "url bar truncates domain", top_k=3, min_similarity=-1.0
    )

    stub = _StubReranker()
    reranked = Retriever(
        indexed_store, embedder=embedder, reranker=stub, rerank_pool=50
    ).retrieve("url bar truncates domain", top_k=3, min_similarity=-1.0)

    # The reranker saw the dense candidates...
    assert stub.received is not None
    assert [bug.bug_id for bug in stub.received] == [bug.bug_id for bug in dense]
    # ...and the returned order is the reranker's, not the dense order.
    assert [bug.bug_id for bug in reranked] == [bug.bug_id for bug in reversed(dense)]


def _bm25_over(store):
    from src.retrieval.bm25_index import BM25Index

    got = store.collection.get(include=["documents"])
    return BM25Index.from_documents(got["ids"], got["documents"])


def test_hybrid_returns_rrf_scored_rows(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder, bm25=_bm25_over(indexed_store))

    results = retriever.retrieve("url bar truncates domain", top_k=2)

    assert len(results) == 2
    # Hybrid rows carry an RRF score; dense-only rows would not.
    assert all(result.rrf_score is not None for result in results)


def test_hybrid_excludes_query_bug(indexed_store, embedder):
    retriever = Retriever(indexed_store, embedder=embedder, bm25=_bm25_over(indexed_store))

    results = retriever.retrieve("bookmark sync", top_k=3, exclude_ids=["BUG-3"])

    assert "BUG-3" not in [result.bug_id for result in results]
