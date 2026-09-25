import numpy as np
import pandas as pd
from strata_hierarchy.v076.pressure_field_finalize import decomposition_summary

def test_large_absolute_roundoff_can_still_be_float_exact():
    # Mimics catastrophic cancellation: an absolute ~1e-8 error at scale 1e8.
    obs=np.array([1.0])
    pot=np.array([1e8])
    res=obs-pot
    recon=pot+res
    err=recon-obs
    scale=np.maximum.reduce([np.abs(obs),np.abs(pot),np.abs(res),np.ones(1)])
    rel=np.abs(err)/scale
    E=pd.DataFrame({
        "pressure_component":[0],
        "delta_p_observed":obs,
        "delta_p_potential":pot,
        "delta_p_residual":res,
        "delta_p_reconstructed":recon,
        "reconstruction_error":err,
        "reconstruction_scale":scale,
        "reconstruction_relative_error":rel,
        "pressure_consistency_ratio":[abs(res[0])/(abs(obs[0])+1e-12)],
    })
    s=decomposition_summary(E)
    assert s["exact_additive_decomposition_float64"]
