import numpy as np
import pytest

from src.ingestion.dataset_mapper import DatasetMapper, RowMappingError


def make_row(**overrides) -> dict:
    row = {
        "Issue id": 1606532,
        "Summary": "Address bar does not elide origins",
        "Status": "RESOLVED",
        "Priority": "P3",
        "Resolution": "DUPLICATE",
        "Created": "2020-01-01 05:10:54+00:00",
        "Resolved": "2023-06-06 00:44:25+00:00",
        "Description": (
            "Steps to reproduce: open the address bar and type a long domain. "
            "The console then reports TypeError: element is undefined and the "
            "page stops rendering."
        ),
    }
    row.update(overrides)
    return row


def test_maps_a_well_formed_row():
    bug = DatasetMapper.map_row(make_row(), project="firefox")

    assert bug.bug_id == "1606532"
    assert bug.project == "firefox"
    assert bug.status == "RESOLVED"
    assert bug.priority == "P3"
    assert bug.resolution == "DUPLICATE"
    assert bug.created_at.year == 2020


def test_error_log_is_extracted_not_copied_from_description():
    # Previously error_log = description, embedding the same text twice.
    bug = DatasetMapper.map_row(make_row(), project="firefox")

    assert "TypeError: element is undefined" in bug.error_log
    assert "Steps to reproduce" in bug.description
    # The error log is a distilled signature, not a copy of the description.
    assert len(bug.error_log) < len(bug.description)


def test_nan_description_becomes_empty_string():
    bug = DatasetMapper.map_row(make_row(Description=np.nan), project="firefox")
    assert bug.description == ""
    assert bug.error_log == ""


def test_nan_resolution_becomes_none():
    bug = DatasetMapper.map_row(make_row(Resolution=np.nan), project="firefox")
    assert bug.resolution is None


def test_nan_resolved_becomes_none():
    bug = DatasetMapper.map_row(make_row(Resolved=np.nan), project="firefox")
    assert bug.resolved_at is None


def test_float_issue_id_does_not_become_dot_zero():
    # pandas types an int column as float64 when nulls are present; str(1606532.0)
    # would silently fork the id space against the CSV.
    bug = DatasetMapper.map_row(make_row(**{"Issue id": 1606532.0}), project="firefox")
    assert bug.bug_id == "1606532"


@pytest.mark.parametrize("bad", ["not-a-date", np.nan])
def test_malformed_created_raises_row_mapping_error(bad):
    # One bad row must be skippable, not fatal to a 40-minute index run.
    with pytest.raises(RowMappingError):
        DatasetMapper.map_row(make_row(Created=bad), project="firefox")


def test_missing_column_raises_row_mapping_error():
    row = make_row()
    del row["Issue id"]
    with pytest.raises(RowMappingError):
        DatasetMapper.map_row(row, project="firefox")
