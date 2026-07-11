import logging
from typing import Iterator

from src.bug_report import BugReport
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

    def _flush(self, batch_bugs: list[BugReport], batch_texts: list[str]) -> int:
        if not batch_texts:
            return 0
        embeddings = self.embedder.encode_documents(batch_texts)
        self.store.add_batch(
            bugs=batch_bugs, embeddings=embeddings, documents=batch_texts
        )
        return len(batch_texts)

    def index(
        self,
        bugs_generator: Iterator[BugReport],
        total_count: int | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> int:
        batch_bugs: list[BugReport] = []
        batch_texts: list[str] = []
        indexed_count = 0
        skipped_count = 0

        for bug in bugs_generator:
            processed_bug = Preprocessor.process(bug)

            if not is_indexable(processed_bug):
                skipped_count += 1
                continue

            batch_bugs.append(processed_bug)
            batch_texts.append(build_retrieval_text(processed_bug))

            if len(batch_texts) == batch_size:
                indexed_count += self._flush(batch_bugs, batch_texts)
                batch_bugs.clear()
                batch_texts.clear()
                if total_count:
                    print(f"Indexed {indexed_count}/{total_count} bug reports", end="\r")

        indexed_count += self._flush(batch_bugs, batch_texts)

        logger.info("Indexed %d bug reports, skipped %d", indexed_count, skipped_count)
        print(f"Indexed {indexed_count} bug reports, skipped {skipped_count} empty")

        return indexed_count
