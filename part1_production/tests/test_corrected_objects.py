import pandas as pd
from strata_native_mechanics.corrections import CorrectionConfig,corrected_objects

def test_unstable_gap_interface_removed():
    E=pd.DataFrame([
        {"interface_id":0,"kind":"cell_cell","cell_i":1,"cell_j":2,
         "background_component":None,"length":10.,
         "curvature":.1,"curvature_confidence":1.0,
         "centroid_x":0.,"centroid_y":0.},
        {"interface_id":1,"kind":"cell_boundary","cell_i":1,"cell_j":None,
         "background_component":7,"length":2.,
         "curvature":.0,"curvature_confidence":0.0,
         "centroid_x":1.,"centroid_y":1.},
    ])
    J=pd.DataFrame(columns=["junction_id","x","y","incident_interfaces","n_regions"])
    P=pd.DataFrame([{"background_component":7,
                     "mechanical_boundary_class":"unstable_micro_gap"}])
    E2,J2,a=corrected_objects(E,J,P,CorrectionConfig(),"persistent_boundaries")
    assert len(E2)==1
    assert a["removed_microgap_interfaces"]==1
