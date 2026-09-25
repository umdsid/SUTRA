from sutra.hierarchy.v078.holonomy import enumerate_chordless_quads

def test_chordless_quad_found_once():
    adj={
        0:{1,3},
        1:{0,2},
        2:{1,3},
        3:{0,2},
    }
    q=enumerate_chordless_quads(adj,100)
    assert len(q)==1
    assert set(q[0])=={0,1,2,3}
