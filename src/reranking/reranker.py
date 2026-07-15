import dataclasses

from sentence_transformers import CrossEncoder

from src.config import RERANK_BATCH_SIZE, RERANK_MAX_CHARS, RERANKER_MODEL
from src.retrieval.retriever import RetrievedBug


class CrossEncoderReranker:
    """Second-stage reranker over dense candidates.

    The bi-encoder embeds query and document separately, so at query time it never sees
    the two together -- it cannot tell that a candidate matches on the exact error token
    that matters. A cross-encoder feeds the (query, document) pair through one model and
    scores relevance directly. It is too slow to run over the whole corpus, but over the
    top RERANK_POOL dense hits it typically lifts Recall@k for little added latency.
    """

    def __init__(
        self,
        model_name: str = RERANKER_MODEL,
        device: str | None = None,
        batch_size: int = RERANK_BATCH_SIZE,
    ) -> None:
        # CrossEncoder, like SentenceTransformer, selects CUDA automatically when device
        # is None and a GPU is present.
        self.model = CrossEncoder(model_name, device=device)
        self.batch_size = batch_size

        # Half precision on GPU: roughly 2x faster inference at negligible ranking-quality
        # cost, and bge-reranker-base fits a 6GB card comfortably in fp16.
        if self.model.model.device.type == "cuda":
            self.model.model.half()

    def rerank(
        self, query_text: str, candidates: list[RetrievedBug], top_k: int
    ) -> list[RetrievedBug]:
        if not candidates:
            return []

        # Cap both sides so the pair fits 512 tokens without each being squeezed to half.
        query = query_text[:RERANK_MAX_CHARS]
        pairs = [
            (query, candidate.document[:RERANK_MAX_CHARS]) for candidate in candidates
        ]
        scores = self.model.predict(
            pairs, convert_to_numpy=True, batch_size=self.batch_size
        )

        reranked = [
            dataclasses.replace(candidate, rerank_score=float(score))
            for candidate, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda bug: bug.rerank_score, reverse=True)
        return reranked[:top_k]
