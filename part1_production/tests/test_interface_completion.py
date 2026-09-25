import numpy as np,pandas as pd
from scipy import ndimage
from strata_native_mechanics.interface_completion import InterfaceCompletionConfig,complete_interfaces_and_junctions

def test_rejects_when_not_exactly_one_missing_pair():
    m=np.array([[1,1,2],[1,0,2],[3,3,2]],dtype=np.int32)
    bg,n=ndimage.label(m==0,structure=np.ones((3,3),np.uint8));bc=int(bg[1,1])
    P=pd.DataFrame([{"background_component":bc,"mechanical_boundary_class":"unstable_micro_gap"}])
    E=pd.DataFrame(columns=["interface_id","kind","cell_i","cell_j","centroid_x","centroid_y"])
    J=pd.DataFrame(columns=["junction_id","x","y","incident_interfaces","n_regions"])
    E2,J2,D,A=complete_interfaces_and_junctions(m,bg,P,E,J,InterfaceCompletionConfig())
    assert A.get("accepted",0)==0
    assert A.get("reject_not_exactly_one_missing_pair",0)>=1

def test_biological_mask_not_modified():
    m=np.array([[1,1,2],[1,0,2],[3,3,2]],dtype=np.int32);orig=m.copy()
    bg,n=ndimage.label(m==0,structure=np.ones((3,3),np.uint8));bc=int(bg[1,1])
    P=pd.DataFrame([{"background_component":bc,"mechanical_boundary_class":"unstable_micro_gap"}])
    E=pd.DataFrame(columns=["interface_id","kind","cell_i","cell_j","centroid_x","centroid_y"])
    J=pd.DataFrame(columns=["junction_id","x","y","incident_interfaces","n_regions"])
    complete_interfaces_and_junctions(m,bg,P,E,J,InterfaceCompletionConfig())
    assert np.array_equal(m,orig)
