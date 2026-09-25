
from pathlib import Path
def test_no_scientific_change():
    root=Path(__file__).parents[1]
    t=(root/"src/strata_hierarchy/v1113/replay.py").read_text().lower()
    assert "merge_asof" not in t
