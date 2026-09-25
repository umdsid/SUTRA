from strata_hierarchy.v108.mass_spectrum import rank_size
def test_rank_descending():
    r=rank_size([1,5,2])
    assert list(r.mass)==[5,2,1]
