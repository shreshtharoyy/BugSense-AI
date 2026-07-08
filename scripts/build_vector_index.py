from pathlib import Path
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_mapper import DatasetMapper
from src.indexing.indexing_pipeline import IndexingPipeline

dataset_root = Path("dataset/gitbugs")
project_name = ["firefox"]

def main():
    pipeline = IndexingPipeline()
    total_bugs = 0

    print("Building BugSense vector index")

    for project in project_name:
        print(f"Indexing {project}")
        csv_path = dataset_root/project/f"{project}_bugs.csv"

        total_count = DatasetLoader.count(csv_path)

        bugs = (DatasetMapper.map_row(row=row, project=project) for row in DatasetLoader.load(csv_path))

        pipeline.index(bugs_generator=bugs, total_count=total_count, batch_size=16)

        total_indexed += total_count
        print(f"Indexed {total_count} bugs")
        print(f"Vector index created. Total bugs indexed: {total_indexed}")

if __name__ == "__main__":
    main()