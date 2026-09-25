import json
from pathlib import Path
from strata_hierarchy.v110.terminal_rule import canonical_json_hash
from strata_hierarchy.v1100.production import EXPECTED_TERMINAL_RULE_SHA256
def test_frozen_config_hash_matches_inspected_contract():
    root=Path(__file__).resolve().parents[1]
    p=root/"configs"/"hierarchy_v110_production_terminal_rule.json"
    d=json.loads(p.read_text())
    assert canonical_json_hash(d)==EXPECTED_TERMINAL_RULE_SHA256
