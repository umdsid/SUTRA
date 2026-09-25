import numpy as np, tempfile
from pathlib import Path
from strata_hierarchy.v1092.reconstruct import inspect_npz_partition_candidates
def test_wrong_n0_rejected():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/"x.npz";np.savez(p,labels=np.array([0,1,1]))
        rows,_=inspect_npz_partition_candidates(p,4,2)
        assert rows==[]
