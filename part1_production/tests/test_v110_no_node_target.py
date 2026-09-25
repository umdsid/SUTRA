import json
from pathlib import Path
from strata_hierarchy.v110.terminal_rule import validate_frozen_config
def test_no_target_node_count():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/"configs/hierarchy_v110_production_terminal_rule.json").read_text())
    assert validate_frozen_config(c)==[]
    forbidden={"target_nodes","target_node_count","desired_nodes","stop_at_nodes","terminal_nodes"}
    assert not (forbidden & set(c))
    assert c["guard_can_trigger_success"] is False
