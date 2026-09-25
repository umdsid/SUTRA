
from pathlib import Path
def test_history_semantics():
    root=Path(__file__).parents[1]
    t=(root/"src/strata_hierarchy/v1113/replay.py").read_text()
    assert "records.append(rec)" in t
    assert "rule.evaluate_once,hist,rule_cfg" in t
    assert "rule.evaluate_persistent,hist,rule_cfg" in t
