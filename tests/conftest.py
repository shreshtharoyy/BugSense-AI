from datetime import datetime
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest

from src.bug_factory import BugFactory
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore


@pytest.fixture
def store():
    """An in-memory ChromaStore. Never touches data/chromadb.

    chromadb.EphemeralClient() hands back a shared in-memory system, so every test
    would otherwise write into the same collection. A unique name per test isolates them.
    """
    return ChromaStore(
        client=chromadb.EphemeralClient(),
        collection_name=f"test-{uuid4().hex}",
    )


@pytest.fixture(scope="session")
def embedder():
    # Session-scoped: loading BGE per test would dominate the run.
    return TextEmbedder()


@pytest.fixture
def bug():
    return BugFactory.create(
        bug_id="BUG-1",
        title="Login button crashes",
        description="The app crashes when logging in with valid credentials.",
        error_log="java.lang.NullPointerException at LoginService.java:45",
        screenshot_path=Path("shots/login.png"),
        created_at=datetime(2020, 1, 1, 5, 10, 54),
        resolved_at=datetime(2020, 2, 6, 10, 1, 56),
        resolution="FIXED",
        status="RESOLVED",
        priority="P1",
        project="firefox",
    )
