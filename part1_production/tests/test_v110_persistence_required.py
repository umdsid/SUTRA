import json,numpy as np,pandas as pd
from pathlib import Path
from strata_hierarchy.v110.terminal_rule import evaluate_persistent
def test_insufficient_history_does_not_stop():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/"configs/hierarchy_v110_production_terminal_rule.json").read_text())
    h=pd.DataFrame({"nodes":[1000]*5})
    r=evaluate_persistent(h,c)
    assert not r["stop"]
