from strata_hierarchy.v071.block_preflight import discover_functional_gmts


def test_signaling_named_gmt_is_not_functional(tmp_path):
    (tmp_path/"cellchat_signaling.gmt").write_text("S\\td\\tA\\tB\\n")
    (tmp_path/"h.all.test.Hs.symbols.gmt").write_text("S\\td\\tA\\tB\\n")
    x=discover_functional_gmts(tmp_path)
    assert len(x)==1
    assert x[0].name.startswith("h.all")
