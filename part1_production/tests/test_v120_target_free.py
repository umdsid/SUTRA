
import json
from pathlib import Path
def test_frozen():
    root=Path(__file__).parents[1]
    c=json.loads((root/"configs/hierarchy_v120_merger_driver_atlas.json").read_text())
    assert c["new_merges"] is False
    assert c["hierarchy_changed"] is False
    assert c["thresholds_modified"] is False
