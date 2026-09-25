import pandas as pd
from strata_hierarchy.v091.completion import compile_cellchat

def test_partial_complex_is_not_declared_panel_supported():
    reg=pd.DataFrame([
        {"sender_gene":"LIGA_LIGB","receiver_gene":"REC1"}
    ])
    channels=compile_cellchat(reg,["LIGA","REC1"])
    assert channels==[]
