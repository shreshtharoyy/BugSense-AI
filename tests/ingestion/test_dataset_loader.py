import inspect

import pandas as pd
import pytest

from src.ingestion.dataset_loader import DatasetLoader


@pytest.fixture
def csv_path(tmp_path):
    path = tmp_path / "bugs.csv"
    pd.DataFrame(
        {"Issue id": range(50), "Summary": [f"bug {i}" for i in range(50)]}
    ).to_csv(path, index=False)
    return path


def test_load_yields_every_row(csv_path):
    rows = list(DatasetLoader.load(csv_path, chunk_size=7))
    assert len(rows) == 50
    assert rows[0]["Summary"] == "bug 0"


def test_count_matches_row_count(csv_path):
    assert DatasetLoader.count(csv_path, chunk_size=7) == 50


def test_load_is_lazy(csv_path):
    """The old implementation materialized every row before yielding the first."""
    generator = DatasetLoader.load(csv_path, chunk_size=7)

    assert inspect.isgenerator(generator)
    assert next(generator)["Issue id"] == 0
