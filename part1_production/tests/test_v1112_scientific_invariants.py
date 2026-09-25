
import json
from pathlib import Path
def test_invariants():
    root=Path(__file__).parents[1]
    c=json.loads((root/"configs/hierarchy_v1112_support_provenance_recovery.json").read_text())
    assert c["allow_threshold_invention"] is False
    assert c["allow_nearest_imputation"] is False
    assert c["new_merges"] is False
    assert c["node_count_target"] is None
