from strata_hierarchy.v0941.audit import block_for_column
def test_block_assignment():
    assert block_for_column("cellchat_total_mean")=="CellChat"
    assert block_for_column("holonomy_density_q95")=="holonomy"
    assert block_for_column("tension_confidence_mean")=="mechanics"
