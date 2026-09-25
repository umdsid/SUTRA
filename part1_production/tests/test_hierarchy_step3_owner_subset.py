def test_mechanics_ownership_is_allowed_to_be_strict_subset_of_geometry():
    geometry={1,2,3,4}
    owned={1,3}
    nonmechanics=geometry-owned
    assert nonmechanics=={2,4}
    assert not (owned-geometry)
