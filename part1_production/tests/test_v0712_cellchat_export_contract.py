import pytest
import strata_hierarchy.v071.cellchat_rdata as cc


def test_missing_rscript_fails_cleanly(monkeypatch,tmp_path):
    monkeypatch.setattr(cc.shutil,"which",lambda name: None)
    src=tmp_path/"CellChatDB.human.rda"; src.write_bytes(b"x")
    script=tmp_path/"extract.R"; script.write_text("print('x')\\n")
    with pytest.raises(RuntimeError,match="Rscript"):
        cc.export_cellchat_interaction_with_r(
            src,"human_rda",tmp_path/"out.csv",script
        )
