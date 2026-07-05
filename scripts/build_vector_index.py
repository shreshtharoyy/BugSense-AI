from pathlib import Path
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_mapper import DatasetMapper
from src.indexing.indexing_pipeline import IndexingPipeline

dataset_root = Path("dataset/gitbugs")
project_names = ["firefox", "cassandra", "vscode"]

def main():
    pipeline = IndexingPipeline()
    total_bugs = 0

    print("Building BugSense vector index")

    for project in project_names:
        print(f"Indexing {project}")
        csv_path = dataset_root/project/f"{project}_bugs.csv"

        rows = DatasetLoader.load(csv_path)

        bugs = [DatasetMapper.map_row(row=row, project=project) for row in rows]

        pipeline.index(bugs)

        total_bugs += len(bugs)

        print(f"Indexed {len(bugs)} bugs")

    print(f"Vector index created and total bugs indexed: {total_bugs}")

if __name__ == "__main__":
    main()