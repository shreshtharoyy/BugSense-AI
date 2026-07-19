import numpy as np

from src.config import BGE_QUERY_PREFIX


def test_encode_documents_returns_matrix(embedder):
    dim = embedder.model.get_embedding_dimension()
    embeddings = embedder.encode_documents(["first bug", "second bug"])
    assert embeddings.shape == (2, dim)


def test_encode_query_returns_vector(embedder):
    # encode_query always returns (dim,), never (1, dim) — callers depend on that.
    dim = embedder.model.get_embedding_dimension()
    assert embedder.encode_query("login crash").shape == (dim,)


def test_embeddings_are_normalized(embedder):
    vector = embedder.encode_query("login crash")
    assert np.linalg.norm(vector) == np.float32(1.0).item() or np.isclose(
        np.linalg.norm(vector), 1.0, atol=1e-5
    )


def test_query_prefix_is_applied_and_document_prefix_is_not(embedder):
    """BGE is asymmetric: only the query side carries the instruction prefix."""
    query = "login crash"

    from_encode_query = embedder.encode_query(query)
    manually_prefixed = embedder.encode_documents([BGE_QUERY_PREFIX + query])[0]
    unprefixed = embedder.encode_documents([query])[0]

    assert np.allclose(from_encode_query, manually_prefixed, atol=1e-5)
    assert not np.allclose(from_encode_query, unprefixed, atol=1e-5)
