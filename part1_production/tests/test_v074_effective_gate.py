import pandas as pd
from sutra.hierarchy.v074.effective_state_v074_frozen import evaluate_effective_candidates


def test_effective_gate_remains_conjunctive():
    t={
        "min_mechanics_support_fraction":0.5,
        "abs_tension_z_max":1.,
        "abs_delta_p_z_max":1.,
        "comm_strength_min":0.2,
        "comm_reciprocity_min":0.5,
    }
    x=pd.DataFrame([{
        "super_i":0,"super_j":1,
        "molecular_distance":0.1,
        "mechanics_support_fraction":0.9,
        "abs_tension_z":2.0,
        "abs_delta_p_z":0.2,
        "comm_strength":1.0,
        "comm_reciprocity":0.9,
        "n_boundary_edges":1,
    }])
    y=evaluate_effective_candidates(x,t,0.5)
    assert not bool(y.iloc[0].admissible)
