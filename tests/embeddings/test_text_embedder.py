from pathlib import Path

from src.bug_factory import BugFactory
from src.preprocessing.preprocessor import Preprocessor
from src.embeddings.text_embedder import TextEmbedder


def main():

    bug = BugFactory.create(
        screenshot_path=None,
        description="""
            Login      button      crashes
            after clicking.
        """,
        error_log="""
            java.lang.NullPointerException

            at LoginService.java:45
        """,
    )

    bug = Preprocessor.process(bug)

    embedder = TextEmbedder()

    embedding = embedder.encode(bug.description)

    print("Bug ID:")
    print(bug.bug_id)

    print("\nClean Description:")
    print(bug.description)

    print("\nClean Error Log:")
    print(bug.error_log)

    print("\nEmbedding Shape:")
    print(embedding.shape)

    print("\nFirst 10 Values:")
    print(embedding[:10])

if __name__ == "__main__":
    main()