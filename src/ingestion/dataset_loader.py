from pathlib import Path
import pandas as pd

class DatasetLoader:
    @staticmethod
    def load(csv_path: Path)-> list[dict]:
        dataframe = pd.read_csv(csv_path)
        return dataframe.to_dict(orient="records")