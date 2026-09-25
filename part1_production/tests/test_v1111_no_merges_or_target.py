
from pathlib import Path
def test_no_merge_target():
    root=Path(__file__).parents[1]
    txt=(root/"src/strata_hierarchy/v1111/audit.py").read_text().lower()
    cfg=(root/"configs/hierarchy_v1111_terminal_landscape_repair.json").read_text()
    assert "merge_batch" not in txt
    assert '"node_count_target": null' in cfg
    assert '"new_merges": false' in cfg
