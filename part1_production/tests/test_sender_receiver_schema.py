import pandas as pd
from strata_hierarchy.v091.completion import compile_cellchat

def test_canonical_sender_receiver_schema_is_supported():
    reg=pd.DataFrame([
        {
            "sender_gene":"LIG1",
            "receiver_gene":"REC1",
            "kind":"Secreted Signaling",
            "family":"demo",
        }
    ])
    channels=compile_cellchat(reg,["LIG1","REC1","OTHER"])
    assert len(channels)==1
    idx,send,recv=channels[0]
    assert send==(0,)
    assert recv==(1,)
