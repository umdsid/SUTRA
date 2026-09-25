import warnings
import numpy as np
import pandas as pd

from strata_hierarchy.v0911.contextual_flow import (
    ContextualFlowConfig,
    freeze_cost_thresholds,
    primitive_fields,
    specimen_scales,
)


def _frame(mult):
    return pd.DataFrame({
        "molecular_effective": np.array([1., 2., 4.]) * mult,
        "mechanics_effective": np.array([2., 3., 6.]) * mult,
        "communication_support_effective": np.array([1., 1.5, 3.]) * mult,
        "geometry_effective": np.array([1., 5., 9.]) * mult,
        "topology_effective": np.array([.1, .2, .4]) * mult,
    })


def test_specimen_scales_are_local_not_pooled():
    a = specimen_scales(_frame(1.0))
    b = specimen_scales(_frame(10.0))
    for key in a:
        assert np.isclose(b[key], 10.0 * a[key])


def test_thresholds_are_specimen_local_empirical_quantiles():
    cfg = ContextualFlowConfig(initial_cost_quantile=.35, maximum_cost_quantile=.90)
    x = np.array([.1, .2, .3, .8, 1.2, 2.0])
    scored = pd.DataFrame({"composite_merge_cost": x})
    f = freeze_cost_thresholds(scored, cfg)
    assert np.isclose(f["initial_threshold"], np.quantile(x, .35))
    assert np.isclose(f["maximum_threshold"], np.quantile(x, .90))
    assert f["specimen_local"] is True
    assert f["pooled_across_specimens"] is False


def test_all_missing_mechanics_is_explicit_and_warning_free():
    cand = pd.DataFrame({
        "abs_tension_z": [np.nan, np.nan],
        "abs_delta_p_z": [np.nan, np.nan],
        "mechanics_support_fraction": [0.0, 0.0],
    })
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fields = primitive_fields(cand)
    mech, rel = fields["mechanics"]
    assert np.isnan(mech).all()
    assert np.all(rel == 0)
    assert not any("Mean of empty slice" in str(w.message) for w in rec)
