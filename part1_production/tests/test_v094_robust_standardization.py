import pandas as pd, numpy as np
from strata_hierarchy.v094.diagnostics import robust_standardize
def test_standardization_is_finite_for_variable_column():
    d=pd.DataFrame({"expression_x":[1.,2.,3.,4.,5.]})
    z,m=robust_standardize(d,["expression_x"])
    assert np.isfinite(z).all()
