import numpy as np,pandas as pd
from scipy import sparse
from sutra.hierarchy.v076.geometry_hierarchy import geometry_costs

def test_pair_cost_is_storage_orientation_invariant():
    G=np.eye(3)
    ids=np.array([0,1])
    states=np.array([[0.,0.,0.],[1.,2.,.5]])
    B=sparse.csr_matrix(np.array([[.1,0.,0.],[-.05,.02,0.]]))
    a=pd.DataFrame([{"super_i":0,"super_j":1,"admissible":True,"ordering_merit":1.,"n_boundary_edges":1}])
    b=pd.DataFrame([{"super_i":1,"super_j":0,"admissible":True,"ordering_merit":1.,"n_boundary_edges":1}])
    ca=geometry_costs(a,ids,states,B,G).iloc[0].geometry_pair_cost
    cb=geometry_costs(b,ids,states,B,G).iloc[0].geometry_pair_cost
    assert np.isclose(ca,cb)
