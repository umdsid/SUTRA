import json
from pathlib import Path
def test_integration_adds_no_target_count():
    root=Path(__file__).resolve().parents[1]
    d=json.loads((root/"configs/hierarchy_v1100_frozen_production_integration.json").read_text())
    s=json.dumps(d).lower()
    assert "target_node" not in s and "desired_node" not in s
