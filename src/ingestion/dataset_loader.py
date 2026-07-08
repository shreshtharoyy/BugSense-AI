from pathlib import Path
import pandas as pd
from typing import Iterator

class DatasetLoader:
    @staticmethod
    def load(csv_path: Path)-> Iterator[dict]:
        dataframe = pd.read_csv(csv_path)
        for row in dataframe.to_dict(orient="records"):
            yield row

    @staticmethod
    def count(csv_path:Path)->int:
        return sum(1 for _ in pd.read_csv(csv_path, usecols=[0]).itertuples())