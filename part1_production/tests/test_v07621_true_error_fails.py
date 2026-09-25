import numpy as np,pandas as pd
from strata_hierarchy.v076.pressure_field_finalize import decomposition_summary

def test_true_relative_reconstruction_error_fails():
    E=pd.DataFrame({
        "pressure_component":[0],
        "delta_p_observed":[1.0],
        "delta_p_potential":[0.4],
        "delta_p_residual":[0.5],
        "delta_p_reconstructed":[0.9],
        "reconstruction_error":[-0.1],
        "reconstruction_scale":[1.0],
        "reconstruction_relative_error":[0.1],
        "pressure_consistency_ratio":[0.5],
    })
    s=decomposition_summary(E)
    assert not s["exact_additive_decomposition_float64"]
