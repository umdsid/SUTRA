import pandas as pd
from strata_hierarchy.v091.completion import compile_cellchat

def test_generic_ligand_receptor_schema_still_supported():
    reg=pd.DataFrame([
        {"ligand":"LIG1","receptor":"REC1"}
    ])
    channels=compile_cellchat(reg,["REC1","LIG1"])
    assert len(channels)==1
    _,send,recv=channels[0]
    assert send==(1,)
    assert recv==(0,)
