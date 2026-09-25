import numpy as np
import pandas as pd
from strata.cli.tranche2_2b_prepare import _normalize_xenium_csv

def test_bytes_are_decoded_and_numeric_semantics_preserved():
    x=pd.DataFrame({
        "transcript_id":np.array([1,2],dtype=np.uint64),
        "cell_id":[b"A",b"UNASSIGNED"],
        "overlaps_nucleus":np.array([1,0],dtype=np.uint8),
        "feature_name":[b"GENE1",b"GENE2"],
        "x_location":np.array([1.,2.],dtype=np.float32),
        "y_location":np.array([3.,4.],dtype=np.float32),
        "z_location":np.array([0.,0.],dtype=np.float32),
        "qv":np.array([40.,40.],dtype=np.float32),
        "fov_name":[b"0",b"0"],
    })
    y=_normalize_xenium_csv(x)
    assert y.loc[0,"feature_name"]=="GENE1"
    assert y.loc[0,"cell_id"]=="A"
    assert str(y["transcript_id"].dtype)=="uint64"
    assert str(y["overlaps_nucleus"].dtype)=="uint8"
