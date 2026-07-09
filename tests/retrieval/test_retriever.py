from src.bug_factory import BugFactory
from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.text_embedder import TextEmbedder
from src.storage.chroma_store import ChromaStore
from src.retrieval.retriever import Retriever


def main():
    store = ChromaStore()
    embedder = TextEmbedder()
    retriever = Retriever(store)
    
    query_bug = BugFactory.create(
        title="Login Crash",
        description="The application crashes when attempting to log in with valid credentials.",
        error_log="NS_ERROR_FAILURE",
        screenshot_path=None,
    )
    
    query_bug = Preprocessor.process(query_bug)
    query_text = f"""Title: {query_bug.title}. Description: {query_bug.description}. Error Log: {query_bug.error_log}""".strip()
    query_embedding = embedder.encode(query_text)
    
    results = retriever.retrieve(query_embedding, top_k=5)
    
    print(f"Query {query_text} \n")
    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, (bug_id, doc, meta, distance) in enumerate(zip(ids, documents, metadatas, distances),start=1,):
        similarity = 1 - distance
        print(f"Rank       : {i}")
        print(f"Bug ID     : {bug_id}")
        # print(f"Distance   : {distance:.4f}")
        print(f"Similarity : {similarity:.2%}")
        print(f"Project    : {meta.get('project')}")
        print(f"Title      : {meta.get('title')}")
        print(f"Resolution : {meta.get('resolution')}")
        print(f"Created At : {meta.get('created_at')}")
        print(f"Resolved At: {meta.get('resolved_at')} \n")
        print("Retrieved Document")
        print(f"{doc}\n")

if __name__ == "__main__":
    main()