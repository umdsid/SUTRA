import numpy as np
from strata_hierarchy.v077.transport import rotate_minimal

def test_random_nonantipodal_rotations_preserve_norm_and_roundtrip():
    rng=np.random.default_rng(177)
    for _ in range(200):
        a=rng.normal(size=12); a/=np.linalg.norm(a)
        b=rng.normal(size=12); b/=np.linalg.norm(b)
        if a@b < -0.999999999:
            continue
        x=rng.normal(size=12)
        y=rotate_minimal(a,b,x)
        z=rotate_minimal(b,a,y)
        assert np.isclose(np.linalg.norm(y),np.linalg.norm(x),rtol=5e-13,atol=5e-13)
        assert np.allclose(z,x,rtol=2e-12,atol=2e-12)
