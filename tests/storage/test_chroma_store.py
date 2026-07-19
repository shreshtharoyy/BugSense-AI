from datetime import datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

import chromadb

from src.bug_factory import BugFactory
from src.retrieval.retriever import Retriever
from src.storage.chroma_store import ChromaStore, IncompatibleCollectionError


def test_collection_uses_cosine_space(store):
    # Under the default l2 space Chroma returns squared euclidean distance and every
    # similarity downstream is wrong. This assertion pins the space.
    assert store.collection.configuration_json["hnsw"]["space"] == "cosine"


def test_cosine_invariant_one_minus_distance(store):
    """The single assertion that would have caught the original l2 bug."""
    indexed = np.array([0.6, 0.8, 0.0, 0.0])
    query = np.array([1.0, 0.0, 0.0, 0.0])
    expected_cosine = float(indexed @ query)  # 0.6

    store.collection.upsert(
        ids=["b"], embeddings=[indexed.tolist()], documents=["doc"]
    )
    results = store.collection.query(query_embeddings=[query.tolist()], n_results=1)
    distance = results["distances"][0][0]

    assert Retriever._to_similarity(distance) == pytest.approx(expected_cosine, abs=1e-4)


def test_add_batch_serializes_datetime_and_path(store, bug):
    # Chroma raises ValueError on a raw datetime or Path in metadata.
    store.add_batch(bugs=[bug], embeddings=np.zeros((1, 4)), documents=["doc"])

    stored = store.collection.get(ids=["BUG-1::0"])
    metadata = stored["metadatas"][0]

    assert metadata["created_at"] == "2020-01-01T05:10:54"
    assert metadata["resolved_at"] == "2020-02-06T10:01:56"
    assert metadata["screenshot_path"] == str(Path("shots/login.png"))
    assert metadata["status"] == "RESOLVED"
    assert metadata["priority"] == "P1"
    assert metadata["parent_bug_id"] == "BUG-1"
    assert metadata["chunk_index"] == 0
    assert stored["documents"][0] == "doc"


def test_absent_optional_metadata_is_omitted_not_blank(store):
    bug = BugFactory.create(
        bug_id="BUG-2",
        title="No resolution yet",
        description="d",
        error_log="",
        created_at=datetime(2021, 5, 5),
    )
    store.add_batch(bugs=[bug], embeddings=np.zeros((1, 4)), documents=["doc"])

    metadata = store.collection.get(ids=["BUG-2::0"])["metadatas"][0]
    assert "resolved_at" not in metadata
    assert "resolution" not in metadata
    assert "screenshot_path" not in metadata


def test_add_batch_is_idempotent(store, bug):
    for _ in range(2):
        store.add_batch(bugs=[bug], embeddings=np.zeros((1, 4)), documents=["doc"])
    assert store.count() == 1


def test_add_batch_writes_multiple_chunks_for_one_parent(store, bug):
    store.add_batch(
        bugs=[bug, bug],
        embeddings=np.zeros((2, 4)),
        documents=["chunk-a", "chunk-b"],
        chunk_indices=[0, 1],
    )
    assert store.count() == 2
    assert store.existing_parent_ids() == {"BUG-1"}
    assert set(store.existing_ids()) == {"BUG-1::0", "BUG-1::1"}


def test_reset_empties_the_collection(store, bug):
    store.add_batch(bugs=[bug], embeddings=np.zeros((1, 4)), documents=["doc"])
    assert store.count() == 1

    store.reset()
    assert store.count() == 0
    assert store.collection.configuration_json["hnsw"]["space"] == "cosine"


def test_stale_l2_collection_is_rejected_not_silently_reused():
    """get_or_create_collection hands back an existing collection unchanged.

    Passing a cosine configuration does NOT convert a collection that was built with
    l2, and Chroma raises nothing. Without this guard a stale index would silently
    reintroduce the similarity bug.
    """
    client = chromadb.EphemeralClient()
    name = f"stale-{uuid4().hex}"
    client.get_or_create_collection(name)  # defaults to l2

    with pytest.raises(IncompatibleCollectionError, match="--reset"):
        ChromaStore(client=client, collection_name=name)


def test_reset_repairs_a_stale_l2_collection():
    client = chromadb.EphemeralClient()
    name = f"stale-{uuid4().hex}"
    client.get_or_create_collection(name)

    store = ChromaStore(client=client, collection_name=name, require_cosine=False)
    store.reset()

    assert store.collection.configuration_json["hnsw"]["space"] == "cosine"
