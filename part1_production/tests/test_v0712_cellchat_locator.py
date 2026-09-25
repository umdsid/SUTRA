import strata_hierarchy.v071.cellchat_rdata as cc


def test_explicit_rda_is_found(monkeypatch,tmp_path):
    d=tmp_path/"cellchat"; d.mkdir()
    p=d/"CellChatDB.human.rda"; p.write_bytes(b"x")
    monkeypatch.setattr(cc,"_mdfind_name",lambda name: [])
    kind,path=cc.locate_cellchat_database([d])
    assert kind=="human_rda"
    assert path==p.resolve()


def test_explicit_csv_outranks_rda(monkeypatch,tmp_path):
    d=tmp_path/"cellchat"; d.mkdir()
    csv=d/"interaction_input_CellChatDB.csv"
    csv.write_text("ligand,receptor,annotation\\nL,R,Cell-Cell Contact\\n")
    (d/"CellChatDB.human.rda").write_bytes(b"x")
    monkeypatch.setattr(cc,"_mdfind_name",lambda name: [])
    kind,path=cc.locate_cellchat_database([d])
    assert kind=="interaction_csv"
    assert path==csv.resolve()
