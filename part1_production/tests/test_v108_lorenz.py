from strata_hierarchy.v108.mass_spectrum import lorenz_curve
def test_lorenz_endpoints():
    l=lorenz_curve([1,2,3])
    assert l.iloc[0].fraction_mass==0
    assert abs(l.iloc[-1].fraction_mass-1)<1e-12
