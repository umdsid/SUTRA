import pandas as pd
import strata_hierarchy.v071.cellchat_rdata as cc


def test_python_export_precedes_rscript(monkeypatch,tmp_path):
    src=tmp_path/"db.rda";src.write_bytes(b"x")
    out=tmp_path/"out.csv"

    def fake_py(source,kind,out_csv):
        pd.DataFrame({"ligand":["L"]}).to_csv(out_csv,index=False)
        return {"export_method":"python_rdata"}

    monkeypatch.setattr(cc,"export_cellchat_interaction_with_python",fake_py)
    monkeypatch.setattr(
        cc,
        "export_cellchat_interaction_with_r",
        lambda *a,**k: (_ for _ in ()).throw(AssertionError("R should not run")),
    )
    meta=cc.export_cellchat_interaction(src,"human_rda",out)
    assert meta["export_method"]=="python_rdata"
