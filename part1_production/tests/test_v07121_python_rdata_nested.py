import pandas as pd

from strata_hierarchy.v071.cellchat_rdata import _find_interaction


def test_nested_cellchat_list_finds_interaction():
    df=pd.DataFrame({
        "ligand":["L"],
        "receptor":["R"],
        "annotation":["Cell-Cell Contact"],
    })
    obj={"CellChatDB.human":{"interaction":df,"complex":{"x":[1]}}}
    got=_find_interaction(obj)
    assert got.equals(df)


def test_generic_nested_object_finds_interaction():
    df=pd.DataFrame({"a":[1,2]})
    obj={"foo":{"bar":{"interaction":df}}}
    got=_find_interaction(obj)
    assert got.equals(df)
