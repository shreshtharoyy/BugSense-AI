from src.bug_report import BugReport
from src.embeddings.text_embedder import TextEmbedder
from src.preprocessing.preprocessor import Preprocessor
from src.storage.chroma_store import ChromaStore

class IndexingPipeline:
    def __init__(self)-> None:
        self.embedder=TextEmbedder()
        self.store=ChromaStore()

    def index(self, bugs: list[BugReport])->None:
        for bug in bugs:
            processed_bug = Preprocessor.process(bug)
            embedding = self.embedder.encode(processed_bug.description)
            self.store.add(
                bug_id= processed_bug.bug_id,
                embedding= embedding.tolist(),
                document= processed_bug.description,
                metadata= {
                    "title": processed_bug.title or "",
                    "project": processed_bug.project or "",
                    "resolution": processed_bug.resolution or "",
                }, 
            )

