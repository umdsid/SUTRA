import numpy as np,pandas as pd
from scipy import ndimage
from strata_native_mechanics.junction_recovery import (
    JunctionRecoveryConfig,recover_microgap_junctions
)

def test_three_cell_microgap_can_recover():
    # Three cells surround a one-pixel gap and all pairs also contact locally.
    m=np.zeros((9,9),dtype=np.int32)
    m[1:8,1:4]=1
    m[1:4,4:8]=2
    m[4:8,4:8]=3
    m[4,4]=0
    # create pair contacts 1-2, 1-3, 2-3 near center
    bg,n=ndimage.label(m==0,structure=np.ones((3,3),np.uint8))
    bc=int(bg[4,4])
    P=pd.DataFrame([{"background_component":bc,
                     "mechanical_boundary_class":"unstable_micro_gap"}])
    E=pd.DataFrame([
        {"interface_id":0,"kind":"cell_cell","cell_i":1,"cell_j":2},
        {"interface_id":1,"kind":"cell_cell","cell_i":1,"cell_j":3},
        {"interface_id":2,"kind":"cell_cell","cell_i":2,"cell_j":3},
    ])
    J=pd.DataFrame(columns=["junction_id","x","y","incident_interfaces","n_regions"])
    J2,D,A=recover_microgap_junctions(
        m,bg,P,E,J,
        JunctionRecoveryConfig(local_pad=4,min_angular_separation_deg=0,
                               max_angular_sector_deg=360)
    )
    assert A.get("recovered",0)>=1
    assert len(D)>=1

def test_ambiguous_four_cell_gap_rejected():
    m=np.array([
        [1,1,2,2],
        [1,0,0,2],
        [3,0,0,4],
        [3,3,4,4],
    ],dtype=np.int32)
    bg,n=ndimage.label(m==0,structure=np.ones((3,3),np.uint8))
    bc=int(bg[1,1])
    P=pd.DataFrame([{"background_component":bc,
                     "mechanical_boundary_class":"unstable_micro_gap"}])
    E=pd.DataFrame(columns=["interface_id","kind","cell_i","cell_j"])
    J=pd.DataFrame(columns=["junction_id","x","y","incident_interfaces","n_regions"])
    J2,D,A=recover_microgap_junctions(m,bg,P,E,J,JunctionRecoveryConfig())
    assert A.get("recovered",0)==0
    assert A.get("reject_not_exactly_three_cells",0)>=1
