import numpy as np
import pandas as pd

from sutra.hierarchy.freeze_audit import validate_csr_npz


def test_csr_matches_interface_endpoints(tmp_path):
    p=tmp_path/"g.npz"
    np.savez(
        p,
        indptr=np.array([0,1,2],dtype=np.int64),
        indices=np.array([1,0],dtype=np.int32),
        interface_ids=np.array([7,7],dtype=np.int64),
        node_ids=np.array([0,1],dtype=np.int64),
    )
    E=pd.DataFrame({
        "interface_id":[7,8],
        "cell_i_index":[0,0],
        "cell_j_index":[1,None],
    })
    r=validate_csr_npz(p,2,E)
    assert r["reciprocal"]
    assert r["n_cell_cell_interfaces"]==1
