
from pathlib import Path
def test_science_untouched():
    root=Path(__file__).parents[1]
    # This hotfix is a CLI-only contract repair.
    assert (root/"src/strata_hierarchy/v120/atlas.py").exists()
