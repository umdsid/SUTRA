import numpy as np
import pandas as pd
from strata_hierarchy.v074.effective_state import evaluate_effective_candidates


def test_unsupported_communication_rejects_merge():
    t={
        "min_mechanics_support_fraction":0.5,
        "abs_tension_z_max":1.0,
        "abs_delta_p_z_max":1.0,
    }
    x=pd.DataFrame([{
        "super_i":0,"super_j":1,"n_boundary_edges":1,
        "molecular_distance":0.1,
        "mechanics_support_fraction":1.0,
        "abs_tension_z":0.1,
        "abs_delta_p_z":0.1,
        "comm_support":0.0,
        "comm_supported":False,
        "comm_reciprocity":np.nan,
    }])
    y=evaluate_effective_candidates(
        x,t,molecular_distance_max=1.0,comm_reciprocity_min=0.5
    )
    assert not bool(y.iloc[0].admissible)
    assert not bool(y.iloc[0].pass_communication_support)
