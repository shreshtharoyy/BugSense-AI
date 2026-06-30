from src.storage.chroma_store import ChromaStore
from chromadb.api.types import QueryResult
import numpy as np

class Retriever:
    def __init__(self, store: ChromaStore) -> None:
        self.store = store

    def retrieve(self, embedding: np.ndarray, top_k: int = 5) -> QueryResult:
        results = self.store.collection.query(
            query_embeddings=[embedding.tolist()], n_results=top_k
        )

        return results