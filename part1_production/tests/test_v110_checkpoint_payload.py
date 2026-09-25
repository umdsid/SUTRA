import json
from pathlib import Path
def test_full_payload_is_frozen():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/"configs/hierarchy_v110_production_terminal_rule.json").read_text())
    p=set(c["required_checkpoint_payload"])
    for x in ["active_label_vector","merge_ancestry","holonomy_density",
              "first_scale_differences","second_scale_differences"]:
        assert x in p
