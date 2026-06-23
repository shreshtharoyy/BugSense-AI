from src.bug_factory import BugFactory
from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore


def main():

    bug = BugFactory.create(
        screenshot_path=None,
        description="Login button crashes after clicking",
        error_log="java.lang.NullPointerException",
    )

    bug = Preprocessor.process(bug)

    embedder = TextEmbedder()

    embedding = embedder.encode(
        bug.description
    )

    store = ChromaStore()

    store.add(
        bug_id=bug.bug_id,
        embedding=embedding.tolist(),
        document=bug.description,
        metadata={
            "source": "test"
        },
    )

    print("Stored Bug:")
    print(bug.bug_id)

    print("\nEmbedding Shape:")
    print(embedding.shape)

    print("\nSuccessfully inserted into ChromaDB.")


if __name__ == "__main__":
    main()