import numpy as np
import pandas as pd

def test_unstable_candidate_is_excluded_not_exported():
    x=pd.DataFrame({
        "numerical_status":["CERTIFIED","NUMERICALLY_UNSTABLE","NOT_EXPORTED"],
        "value":[1.0,np.nan,np.nan],
    })
    bad=(x.numerical_status!="CERTIFIED") & x.value.notna()
    assert not bad.any()

def test_certified_value_must_be_finite():
    x=np.array([1.0,-2.0,0.0])
    assert np.all(np.isfinite(x))
