from src.bug_factory import BugFactory
from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever


def main():

    embedder = TextEmbedder()
    store = ChromaStore()

    # -------------------------
    # Create sample bugs
    # -------------------------

    bug_descriptions = [
        "Login button crashes after clicking",
        "Logout button freezes on homepage",
        "Payment gateway timeout during checkout",
    ]

    for description in bug_descriptions:

        bug = BugFactory.create(
            screenshot_path=None,
            description=description,
            error_log="Test Error Log",
        )

        bug = Preprocessor.process(bug)

        embedding = embedder.encode(bug.description)

        store.add(
            bug_id=bug.bug_id,
            embedding=embedding.tolist(),
            document=bug.description,
            metadata={"source": "test"},
        )

    # -------------------------
    # Query
    # -------------------------

    query = "Login page crashes"

    query_embedding = embedder.encode(query)

    retriever = Retriever(store)

    results = retriever.retrieve(
        query_embedding,
        top_k=3,
    )

    print("=" * 60)
    print("Query:")
    print(query)

    print("\nRetrieved Documents:")
    print(results)
    print(type(results))

    for i, document in enumerate(results["documents"][0], start=1):
        print(f"{i}. {document}")

    print("=" * 60)


if __name__ == "__main__":
    main()