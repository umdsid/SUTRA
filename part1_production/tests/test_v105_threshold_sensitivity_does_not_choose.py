import pandas as pd
from strata_hierarchy.v105.pareto_audit import threshold_sensitivity
def test_threshold_grid_only_reports_counts():
    x=pd.DataFrame({"pass_min_basin_landmarks":[True,True],
                    "pass_finite_strength":[True,True],
                    "cross_block_coherence":[.4,.6]})
    y=threshold_sensitivity(x,[.3,.5,.7])
    assert list(y.surviving_basins)==[2,1,0]
    assert "selected" not in y.columns
