from pathlib import Path
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_mapper import DatasetMapper
from src.indexing.indexing_pipeline import IndexingPipeline

dataset_root = Path("dataset/gitbugs")
project_name = "firefox"

def main():
    csv_path = dataset_root/project_name/f"{project_name}_bugs.csv"
    rows = DatasetLoader.load(csv_path)
    print(f"Rows {len(rows)}")

    bugs = []
    for row in rows[:10]:
        bug = DatasetMapper.map_row(row, project=project_name)
        bugs.append(bug)

    pipeline = IndexingPipeline()
    pipeline.index(bugs)

    print(f"Successfully indexed {len(bugs)} bug reports.")
    print("Knowledge Base updated.")

if __name__ == "__main__":
    main()