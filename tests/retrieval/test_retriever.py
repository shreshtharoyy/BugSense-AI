from src.bug_factory import BugFactory
from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever


def main():

    embedder = TextEmbedder()
    store = ChromaStore()

    bug_descriptions = [
        "Login button crashes after clicking",
        "Logout button freezes on homepage",
        "Payment gateway timeout during checkout",
    ]

    for description in bug_descriptions:

        bug = BugFactory.create(
        title=description,
        description=description,
        error_log="Test Error Log",
        screenshot_path=None,
    )

        bug = Preprocessor.process(bug)

        embedding = embedder.encode(bug.description)

        store.add(
            bug_id=bug.bug_id,
            embedding=embedding.tolist(),
            document=bug.description,
            metadata={"source": "test"},
        )

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

    if results is None:
        raise RuntimeError("Retriever returned no results.")

    documents = results.get("documents", [])

    if documents:
        for i, document in enumerate(documents[0], start=1):
            print(f"{i}. {document}")
    else:
        print("No documents retrieved.")

    print("=" * 60)


if __name__ == "__main__":
    main()