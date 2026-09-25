
import pandas as pd
from strata_hierarchy.v1111.audit import normalize_source
def test_support_aliases_resolve():
    cfg={
      "support_aliases":{"mass":["mass_status"],"expr":["expression_status"],"spatial":["spatial_status"]},
      "block_speed_prefixes":["speed_"]
    }
    d=pd.DataFrame({"eval":[0,1],"nodes":[10,9],"mass_status":["PASS","HOLD"],
                    "expression_status":["HOLD","PASS"],"spatial_status":["PASS","PASS"],
                    "speed_x":[None,.1]})
    o,s=normalize_source(d,cfg)
    assert o["mass"].tolist()==[1.0,0.0]
    assert o["expr"].tolist()==[0.0,1.0]
    assert o["spatial"].tolist()==[1.0,1.0]
