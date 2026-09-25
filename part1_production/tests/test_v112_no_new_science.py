
import json
from pathlib import Path
def test_frozen():
    root=Path(__file__).parents[1]
    c=json.loads((root/"configs/hierarchy_v112_final_supported_hierarchy_landscape.json").read_text())
    assert c["new_merges"] is False
    assert c["node_count_target"] is None
    assert c["thresholds_modified"] is False
    assert c["nearest_imputation"] is False
