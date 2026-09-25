import json,pandas as pd
from pathlib import Path
from strata_hierarchy.v110.terminal_rule import evaluate_once
def test_guard_cannot_trigger_stop():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/"configs/hierarchy_v110_production_terminal_rule.json").read_text())
    h=pd.DataFrame({"nodes":[c["minimum_tessellation_guard_nodes"]]})
    r=evaluate_once(h,c)
    assert not r["candidate"] and r["guard_reached"]
