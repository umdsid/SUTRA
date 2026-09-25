import numpy as np
import pandas as pd

from strata_native_mechanics.production_observable import (
    ProductionSolveConfig,
    core_owned_table,
    add_solver_stability_status,
)


def test_unresolved_is_na_not_fabricated():
    owners = pd.DataFrame([
        {"interface_id":1,"owner_patch_id":0,"owner_anchor_cell":10},
        {"interface_id":2,"owner_patch_id":1,"owner_anchor_cell":20},
    ])
    vals = pd.DataFrame([
        {
            "patch_id":0,"interface_id":1,"tension":2.0,
            "tension_lsqr":2.0,"tension_lsmr":2.0,
            "solver_disagreement":0.0,"rowspace_residual_ratio":0.0,
            "status":"OBSERVABLE",
        }
    ])
    out = core_owned_table(vals, owners, "tension")
    out = add_solver_stability_status(out, ProductionSolveConfig())
    a = out.set_index("interface_id")
    assert a.loc[1,"numerical_status"] == "CERTIFIED"
    assert np.isfinite(a.loc[1,"tension_production"])
    assert a.loc[2,"production_status"] == "UNRESOLVED"
    assert np.isnan(a.loc[2,"tension_production"])
