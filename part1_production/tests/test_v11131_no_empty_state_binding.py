
from pathlib import Path
def test_no_empty_state_binding():
    root=Path(__file__).parents[1]
    t=(root/"src/strata_hierarchy/v1113/replay.py").read_text()
    assert "state={}" not in t
