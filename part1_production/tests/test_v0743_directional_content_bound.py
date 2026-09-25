import numpy as np


def test_directional_content_dominates_net_by_triangle_inequality():
    rng=np.random.default_rng(4)
    for _ in range(100):
        d=rng.normal(size=12)
        w=np.abs(d)+rng.random(12)
        net=abs(d.sum())/w.sum()
        content=np.abs(d).sum()/w.sum()
        assert content+1e-12>=net
