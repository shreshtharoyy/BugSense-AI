from src.bug_report import BugReport
from src.embeddings.text_embedder import TextEmbedder
from src.preprocessing.preprocessor import Preprocessor
from src.storage.chroma_store import ChromaStore
from typing import Iterator
import gc

class IndexingPipeline:
    def __init__(self)-> None:
        self.embedder=TextEmbedder()
        self.store=ChromaStore()

    def index(self, bugs_generator: Iterator[BugReport], total_count: int, batch_size: int=16)->None:
        batch_bugs = []
        batch_texts = []
        indexed_count = 0
        for bug in bugs_generator:
            processed_bug = Preprocessor.process(bug)
            description = processed_bug.description or ""

            batch_bugs.append(processed_bug)
            batch_texts.append(description)

            if len(batch_texts) == batch_size:
                embedding = self.embedder.encode(batch_texts, batch_size=batch_size)
                self.store.add_batch(
                    bugs=batch_bugs,
                    embeddings=embedding,
                    document=batch_texts, 
                )
                indexed_count += len(batch_texts)

                print(
                    f"Indexed {indexed_count}/{total_count} bug reports",
                    end="\r",
                )

                batch_bugs.clear()
                batch_texts.clear()

                gc.collect()

        if batch_texts:

            embeddings = self.embedder.encode(
                batch_texts,
                batch_size=batch_size,
            )

            self.store.add_batch(
                bugs=batch_bugs,
                embeddings=embeddings,
                document=batch_texts,
            )

            indexed_count += len(batch_texts)

            print(
                f"Indexed {indexed_count}/{total_count} bug reports"
            )

            gc.collect()  
