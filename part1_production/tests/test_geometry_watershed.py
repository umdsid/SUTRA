import numpy as np
from strata.geometry_benchmark.baselines import transcript_watershed_completion

def test_scipy_watershed_preserves_markers():
    obs=np.zeros((20,20),dtype=np.int32)
    obs[5:8,4:7]=1
    obs[5:8,13:16]=2
    d=np.zeros((20,20),float)
    d[4:10,3:17]=1.0
    out,meta=transcript_watershed_completion(obs,d)
    assert np.all(out[obs>0]==obs[obs>0])
    assert meta["observed_preserved"]
