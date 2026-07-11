from chromadb import PersistentClient
from chromadb.api import ClientAPI
from chromadb.api.collection_configuration import CreateCollectionConfiguration
from chromadb.api.types import Metadata
import numpy as np
from src.bug_report import BugReport
from src.config import CHROMA_PATH, COLLECTION_NAME


class IncompatibleCollectionError(RuntimeError):
    """An existing collection was built with the wrong distance space."""


class ChromaStore:
    # Cosine space is required. Under the default "l2" space Chroma returns squared
    # euclidean distance, and `1 - distance` is then not cosine similarity -- it goes
    # negative below cos 0.5. In cosine space Chroma returns `1 - cos`, so the
    # conversion in Retriever is exact.
    CONFIGURATION: CreateCollectionConfiguration = {"hnsw": {"space": "cosine"}}
    SPACE = "cosine"

    def __init__(
        self,
        db_path: str = CHROMA_PATH,
        collection_name: str = COLLECTION_NAME,
        client: ClientAPI | None = None,
        require_cosine: bool = True,
    ) -> None:
        self.client = client if client is not None else PersistentClient(path=db_path)
        self.collection_name = collection_name
        self.collection = self._get_or_create()

        if require_cosine:
            self._assert_cosine_space()

    def _get_or_create(self):
        return self.client.get_or_create_collection(
            name=self.collection_name,
            configuration=self.CONFIGURATION,
            embedding_function=None,
        )

    def _assert_cosine_space(self) -> None:
        # get_or_create_collection does NOT apply this configuration to a collection
        # that already exists -- it silently hands back the old one, wrong space and
        # all. Without this check a stale l2 index reintroduces the similarity bug
        # with no error anywhere.
        space = self.collection.configuration_json["hnsw"]["space"]
        if space != self.SPACE:
            raise IncompatibleCollectionError(
                f"Collection {self.collection_name!r} uses hnsw space {space!r}, "
                f"not {self.SPACE!r}. Chroma cannot convert the distance space of an "
                "existing collection, and the similarity conversion in Retriever is "
                "only valid for cosine. Rebuild the index:\n"
                "    python scripts/build_vector_index.py --reset"
            )

    def reset(self) -> None:
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self._get_or_create()
        self._assert_cosine_space()

    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self) -> set[str]:
        return set(self.collection.get(include=[])["ids"])

    @staticmethod
    def _build_metadata(bug: BugReport) -> Metadata:
        # Chroma accepts only str/int/float/bool/None. A datetime or a Path raises
        # ValueError, so both are serialized here. Absent values are dropped rather than
        # coerced to "", so a missing field stays distinguishable from an empty one.
        metadata = {
            "title": bug.title,
            "project": bug.project,
            "status": bug.status,
            "priority": bug.priority,
            "resolution": bug.resolution,
            "created_at": bug.created_at.isoformat() if bug.created_at else None,
            "resolved_at": bug.resolved_at.isoformat() if bug.resolved_at else None,
            "screenshot_path": str(bug.screenshot_path) if bug.screenshot_path else None,
        }
        return {key: value for key, value in metadata.items() if value is not None}

    def add_batch(
        self, bugs: list[BugReport], embeddings: np.ndarray, documents: list[str]
    ) -> None:
        ids = [bug.bug_id for bug in bugs]
        metadata_list = [self._build_metadata(bug) for bug in bugs]
        embedding_list = [embedding.tolist() for embedding in embeddings]

        # upsert, not add: re-indexing must be idempotent rather than collide on ids.
        self.collection.upsert(
            ids=ids,
            embeddings=embedding_list,
            documents=documents,
            metadatas=metadata_list,
        )
