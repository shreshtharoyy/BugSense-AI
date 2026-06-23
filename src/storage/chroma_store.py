from chromadb import PersistentClient

class ChromaStore:
    def __init__(self, db_path: str= "./data/chromadb", collection_name: str = "bug_reports") -> None:
        self.client = PersistentClient(path=db_path)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add(self, bug_id: str, embedding: list[float], document: str, metadata: dict) -> None:
        self.collection.add(
            ids=[bug_id],
            embeddings=[embedding],
            documents=[document],
            metadatas=[metadata],
        )
        