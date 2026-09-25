import numpy as np
from strata_hierarchy.v107.domain_audit import classify_units
def test_quadrant_labels_do_not_drop_units():
    y,_,_=classify_units([1,2,10,20],[.1,.2,.1,5],
       {"large_mass_quantile":.5,"high_heterogeneity_quantile":.5})
    assert len(y)==4
