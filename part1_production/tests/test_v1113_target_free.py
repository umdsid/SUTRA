
import json
from pathlib import Path
def test_target_free():
    root=Path(__file__).parents[1]
    c=json.loads((root/"configs/hierarchy_v1113_exact_frozen_terminal_replay.json").read_text())
    assert c["node_count_target"] is None
