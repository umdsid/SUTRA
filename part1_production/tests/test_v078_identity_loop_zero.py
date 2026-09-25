import numpy as np
from sutra.hierarchy.v078.holonomy import compose_loop_holonomy,holonomy_metrics

def test_all_zero_directional_frames_have_zero_holonomy():
    Q=np.zeros((3,5))
    H,U=compose_loop_holonomy(Q,(0,1,2))
    m=holonomy_metrics(H)
    assert m["identity_deviation_normalized"]==0.0
    assert m["spectral_angle_max"]==0.0
