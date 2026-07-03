import subprocess
from pathlib import Path

dataset_name = "av9ash/gitbugs"
base_dir = Path(__file__).parent
gitbugs_dir = base_dir / "gitbugs" 

def dataset_exists() -> bool:
    return gitbugs_dir.exists()

def download_dataset() -> None:
    print("Downloading GitBug dataset")
    subprocess.run(
        ["kaggle", "datasets", "download", dataset_name, "-p", str(base_dir), "--unzip"], check=True
    )
    print("Downloaded successfully")

def main():

    if dataset_exists():
        print("GitBugs dataset already exists.")
        print(f"Location: {gitbugs_dir}")
        return

    download_dataset()

    print(f"\nDataset available at:\n{gitbugs_dir}")

if __name__ == "__main__":
    main()
