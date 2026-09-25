from strata_hierarchy.v103.audit import local_minima
def test_minima_are_separated():
    x=[5,2,4,1,4,2,5]
    m=local_minima(x,min_sep=2,top_k=3)
    assert len(m)>=2
