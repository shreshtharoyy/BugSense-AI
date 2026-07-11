from pathlib import Path
import pandas as pd
from typing import Iterator

CHUNK_SIZE = 2000


class DatasetLoader:
    @staticmethod
    def load(csv_path: Path, chunk_size: int = CHUNK_SIZE) -> Iterator[dict]:
        # Genuinely streamed. The previous implementation read the whole CSV and then
        # called to_dict(orient="records"), materializing every row before the first
        # one was yielded -- roughly 56 MB of strings for the Firefox corpus.
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            for row in chunk.to_dict(orient="records"):
                yield row

    @staticmethod
    def count(csv_path: Path, chunk_size: int = CHUNK_SIZE) -> int:
        return sum(
            len(chunk)
            for chunk in pd.read_csv(
                csv_path, usecols=["Issue id"], chunksize=chunk_size
            )
        )
