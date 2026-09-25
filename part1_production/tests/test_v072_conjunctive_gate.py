import pandas as pd
from strata_hierarchy.v072.pilot import evaluate_candidates


def thresholds():
    return {
        "molecular_z_max":-0.5,
        "abs_tension_z_max":1.0,
        "abs_delta_p_z_max":1.0,
        "comm_strength_min":0.5,
        "comm_reciprocity_min":0.5,
        "min_mechanics_support_fraction":0.5,
    }


def test_one_failed_block_rejects_candidate():
    x=pd.DataFrame([{
        "super_i":0,"super_j":1,"n_boundary_edges":1,
        "molecular_z":-1.0,
        "comm_strength":1.0,
        "comm_reciprocity":0.9,
        "mechanics_support_fraction":0.9,
        "abs_tension_z":2.0,  # only failure
        "abs_delta_p_z":0.2,
    }])
    y=evaluate_candidates(x,thresholds())
    assert not bool(y.iloc[0].admissible)


def test_all_blocks_must_pass():
    x=pd.DataFrame([{
        "super_i":0,"super_j":1,"n_boundary_edges":2,
        "molecular_z":-1.0,
        "comm_strength":1.0,
        "comm_reciprocity":0.9,
        "mechanics_support_fraction":0.9,
        "abs_tension_z":0.2,
        "abs_delta_p_z":0.2,
    }])
    y=evaluate_candidates(x,thresholds())
    assert bool(y.iloc[0].admissible)
