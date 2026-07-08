from chromadb import PersistentClient
import numpy as np
from src.bug_report import BugReport

class ChromaStore:
    def __init__(self, db_path: str= "./data/chromadb", collection_name: str = "bug_reports") -> None:
        self.client = PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_batch(self, bugs: list[BugReport], embeddings: np.ndarray, document: list[str]) -> None:
        ids=[]
        embedding_list=[]
        metadata_list=[]

        for bug, embedding in zip(bugs, embeddings):
            ids.append(bug.bug_id)
            embedding_list.append(embedding.tolist())
            metadata_list.append(
                {
                    "title": bug.title or "",
                    "project": bug.project or "",
                    "resolution": bug.resolution or "",
                }
            )

        self.collection.add(
            ids=ids,
            embeddings=embedding_list,
            documents=document,
            metadatas=metadata_list,
        )
        