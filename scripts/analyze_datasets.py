from pathlib import Path
import pandas as pd

dataset_root = Path("dataset/gitbugs")
projects = ["firefox"]

for project in projects:
    print(f"Project: {project}")

    csv_path = dataset_root / project / f"{project}_bugs.csv"
    df = pd.read_csv(csv_path)
    print(f"Shape {df.shape}, Info {df.info()}, Missing values {df.isna().sum()}, Data types {df.dtypes} \n")
