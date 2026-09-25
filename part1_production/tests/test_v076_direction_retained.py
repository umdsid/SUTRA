import numpy as np,pandas as pd
from scipy import sparse
from strata_hierarchy.v076.geometry_hierarchy import geometry_costs

def test_directional_term_changes_pair_cost_when_covectors_differ():
    G=np.eye(2); ids=np.array([0,1]); states=np.array([[0.,0.],[1.,0.]])
    C=pd.DataFrame([{"super_i":0,"super_j":1,"admissible":True,"ordering_merit":1.,"n_boundary_edges":1}])
    B0=sparse.csr_matrix(np.zeros((2,2)))
    B1=sparse.csr_matrix(np.array([[.2,0.],[-.1,0.]]))
    c0=geometry_costs(C,ids,states,B0,G).iloc[0]
    c1=geometry_costs(C,ids,states,B1,G).iloc[0]
    assert not np.isclose(c0.geometry_pair_cost,c1.geometry_pair_cost)
    assert np.isclose(c1.directional_correction,.15)
