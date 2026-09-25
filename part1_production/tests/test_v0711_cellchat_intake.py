import pandas as pd

from strata_hierarchy.v071.resource_intake import canonicalize_cellchat


def test_cellchat_typing_is_conservative(tmp_path):
    p=tmp_path/"interaction_input_CellChatDB.csv"
    pd.DataFrame({
        "ligand":["L1","L2","L3"],
        "receptor":["R1","R2","R3"],
        "annotation":[
            "Cell-Cell Contact",
            "Secreted Signaling",
            "ECM-Receptor",
        ],
        "pathway_name":["A","B","C"],
    }).to_csv(p,index=False)
    out,a=canonicalize_cellchat(p)
    assert set(out.kind)=={"contact","diffusible"}
    assert len(out)==2
    assert a["n_unsupported_annotation"]==1


def test_complex_labels_are_not_exploded(tmp_path):
    p=tmp_path/"interaction_input_CellChatDB.csv"
    pd.DataFrame({
        "ligand":["L1"],
        "receptor":["R1+R2"],
        "annotation":["Cell-Cell Contact"],
    }).to_csv(p,index=False)
    out,a=canonicalize_cellchat(p)
    assert len(out)==0
    assert a["n_complex_or_nonsimple_entity"]==1
