from strata_hierarchy.v078.holonomy import enumerate_triangles

def test_triangle_enumeration_unique():
    adj={
        0:{1,2},
        1:{0,2},
        2:{0,1,3},
        3:{2},
    }
    assert enumerate_triangles(adj)==[(0,1,2)]
