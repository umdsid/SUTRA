from strata_hierarchy.v103.audit import block_for
def test_all_major_blocks_map():
    assert block_for("expression_total_variance")=="expression"
    assert block_for("cellchat_total_mean")=="CellChat"
    assert block_for("holonomy_density_q95")=="holonomy"
