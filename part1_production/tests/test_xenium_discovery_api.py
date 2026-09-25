from pathlib import Path

from strata.io.discovery import (
    XeniumSampleAssets,
    discover_samples,
    discover_xenium_samples,
)


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")


def test_discover_xenium_samples_expected_api(tmp_path):
    s = tmp_path / "healthy_reference"
    _touch(s / "sample_cell_feature_matrix.h5")
    _touch(s / "sample_cells.parquet")
    _touch(s / "sample_cell_boundaries.parquet")
    _touch(s / "sample_nucleus_boundaries.parquet")
    _touch(s / "sample_transcripts.parquet")

    x = discover_xenium_samples(tmp_path)
    assert len(x) == 1
    a = x[0]
    assert isinstance(a, XeniumSampleAssets)
    assert a.sample_id == "healthy_reference"
    assert a.matrix.name == "sample_cell_feature_matrix.h5"
    assert a.cells.name == "sample_cells.parquet"
    assert a.to_dict()["sample_id"] == "healthy_reference"


def test_legacy_discovery_is_preserved(tmp_path):
    s = tmp_path / "sample"
    _touch(s / "cell_boundaries.parquet")
    x = discover_samples(tmp_path)
    assert x[0]["sample"] == "sample"


def test_helper_directory_without_measured_assets_is_ignored(tmp_path):
    (tmp_path / "resources").mkdir()
    assert discover_xenium_samples(tmp_path) == []
