from pathlib import Path
from src.ingestion.dataset_loader import DatasetLoader
from src.ingestion.dataset_mapper import DatasetMapper

def main():
    csv_path=Path('dataset/gitbugs/ms_vscode_bugs/vscode_bugs.csv')

    rows = DatasetLoader.load(csv_path)
    print(f"Rows loaded: {len(rows)}")

    bug = DatasetMapper.map_row(rows[0], project="ms_vscode_bugs")
    print("BugReport:")
    print(f"Bug ID: {bug.bug_id}, title: {bug.title}, description: {bug.description}, error_log: {bug.error_log}, created at: {bug.created_at}, resolved at: {bug.resolved_at}, resolution: {bug.resolution}, project: {bug.project}")

if __name__ == "__main__":
    main()