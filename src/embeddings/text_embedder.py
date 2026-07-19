from sentence_transformers import SentenceTransformer
import numpy as np

from src.config import BATCH_SIZE, BGE_QUERY_PREFIX, EMBEDDING_MODEL


class TextEmbedder:
    def __init__(
        self, model_name: str = EMBEDDING_MODEL, device: str | None = None
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def tokenizer(self):
        """Hugging Face tokenizer used for embedding-aware chunk windows."""
        return self.model.tokenizer

    def encode_documents(
        self, texts: list[str], batch_size: int = BATCH_SIZE
    ) -> np.ndarray:
        """Encode corpus documents. Always shape (N, dim). No prefix."""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a search query. Always shape (dim,).

        BGE is an asymmetric retrieval model: queries carry an instruction prefix and
        documents do not. Embedding a query as though it were a document costs recall,
        so the prefix lives here rather than at the call site where it can be forgotten.
        """
        return self.model.encode(
            BGE_QUERY_PREFIX + text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
