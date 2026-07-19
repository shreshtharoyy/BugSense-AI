import logging
from typing import Iterator

from src.bug_report import BugReport
from src.chunking import chunk_retrieval_text
from src.config import BATCH_SIZE
from src.document_builder import build_retrieval_text, is_indexable
from src.embeddings.text_embedder import TextEmbedder
from src.preprocessing.preprocessor import Preprocessor
from src.storage.chroma_store import ChromaStore

logger = logging.getLogger(__name__)


class IndexingPipeline:
    def __init__(
        self, embedder: TextEmbedder | None = None, store: ChromaStore | None = None
    ) -> None:
        self.embedder = embedder if embedder is not None else TextEmbedder()
        self.store = store if store is not None else ChromaStore()

    def _flush(
        self,
        batch_bugs: list[BugReport],
        batch_texts: list[str],
        batch_indices: list[int],
    ) -> int:
        if not batch_texts:
            return 0
        embeddings = self.embedder.encode_documents(batch_texts)
        self.store.add_batch(
            bugs=batch_bugs,
            embeddings=embeddings,
            documents=batch_texts,
            chunk_indices=batch_indices,
        )
        return len(batch_texts)

    def index(
        self,
        bugs_generator: Iterator[BugReport],
        total_count: int | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> int:
        """Index bugs as one or more chunk vectors. Returns the number of parent bugs written."""
        batch_bugs: list[BugReport] = []
        batch_texts: list[str] = []
        batch_indices: list[int] = []
        bugs_indexed = 0
        chunks_indexed = 0
        skipped_count = 0

        for bug in bugs_generator:
            processed_bug = Preprocessor.process(bug)

            if not is_indexable(processed_bug):
                skipped_count += 1
                continue

            text = build_retrieval_text(processed_bug)
            chunks = chunk_retrieval_text(text, self.embedder.tokenizer)
            if not chunks:
                skipped_count += 1
                continue

            for index, chunk in enumerate(chunks):
                batch_bugs.append(processed_bug)
                batch_texts.append(chunk)
                batch_indices.append(index)

                if len(batch_texts) == batch_size:
                    chunks_indexed += self._flush(batch_bugs, batch_texts, batch_indices)
                    batch_bugs.clear()
                    batch_texts.clear()
                    batch_indices.clear()

            bugs_indexed += 1
            if total_count:
                print(
                    f"Indexed {bugs_indexed}/{total_count} bug reports "
                    f"({chunks_indexed + len(batch_texts)} chunks)",
                    end="\r",
                )

        chunks_indexed += self._flush(batch_bugs, batch_texts, batch_indices)

        logger.info(
            "Indexed %d bug reports as %d chunks, skipped %d",
            bugs_indexed,
            chunks_indexed,
            skipped_count,
        )
        print(
            f"Indexed {bugs_indexed} bug reports as {chunks_indexed} chunks, "
            f"skipped {skipped_count} empty"
        )

        return bugs_indexed
