from pathlib import Path
import pandas as pd

dataset_root = Path("dataset/gitbugs")
projects = ["firefox"]

for project in projects:
    print(f"Project: {project}")

    csv_path = dataset_root / project / f"{project}_bugs.csv"
    df = pd.read_csv(csv_path)

    # df.info() prints as a side effect and returns None, so it cannot be interpolated.
    print(f"Shape {df.shape}")
    df.info()
    print(f"\nMissing values\n{df.isna().sum()}")
    print(f"\nData types\n{df.dtypes}\n")
