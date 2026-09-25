import strata_hierarchy.v071.resource_intake as ri


def test_locator_always_returns_semantic_role_keys(monkeypatch,tmp_path):
    d=tmp_path/"r"; d.mkdir()
    (d/"h.all.v2025.1.Hs.symbols.gmt").write_text("H\\td\\tA\\tB\\n")
    (d/"c2.cp.reactome.v2025.1.Hs.symbols.gmt").write_text("R\\td\\tA\\tB\\n")
    (d/"c5.go.bp.v2025.1.Hs.symbols.gmt").write_text("G\\td\\tA\\tB\\n")
    monkeypatch.setattr(ri,"_mdfind_name",lambda name: [])
    chosen=ri.locate_functional_resources([d])
    assert set(chosen)=={"hallmark","reactome","go_bp"}
    assert chosen["hallmark"].name=="h.all.v2025.1.Hs.symbols.gmt"


def test_explicit_hallmark_copy_can_be_identified_by_role(monkeypatch,tmp_path):
    d=tmp_path/"r"; d.mkdir()
    g=d/"h.all.v2025.1.Hs.symbols.gmt"
    g.write_text("S\\tdesc\\tA\\tB\\n")
    monkeypatch.setattr(ri,"_mdfind_name",lambda name: [])
    chosen=ri.locate_functional_resources([d])
    copied,manifest=ri.copy_functional_snapshot(chosen,tmp_path/"out")
    rec=[x for x in manifest if x["role"]=="hallmark"][0]
    assert rec["source_sha256"]==ri.sha256_file(g)
    assert rec["snapshot_sha256"]==ri.sha256_file(g)
