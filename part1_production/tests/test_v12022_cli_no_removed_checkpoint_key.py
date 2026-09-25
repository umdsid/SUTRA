
from pathlib import Path
def test_no_removed_checkpoint_key():
    root=Path(__file__).parents[1]
    t=(root/"src/strata/cli/merger_driver_atlas_v120.py").read_text()
    assert "r['checkpoint_candidates']" not in t
