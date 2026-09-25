
import json
from pathlib import Path
def test_hash_contract():
    root=Path(__file__).parents[1]
    c=json.loads((root/"configs/hierarchy_v1113_exact_frozen_terminal_replay.json").read_text())
    assert len(c["required_hashes"])==3
    assert c["allow_threshold_modification"] is False
    assert c["new_merges"] is False
