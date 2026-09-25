from strata_hierarchy.v108.mass_spectrum import empirical_ccdf
def test_ccdf_monotone():
    c=empirical_ccdf([1,1,2,4])
    assert all(c.ccdf.iloc[i]>=c.ccdf.iloc[i+1] for i in range(len(c)-1))
