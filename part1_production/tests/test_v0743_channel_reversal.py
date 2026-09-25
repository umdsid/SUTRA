import numpy as np


def test_reversal_flips_net_not_directional_content():
    f=np.array([3.,1.,4.])
    r=np.array([1.,4.,2.])
    d=f-r
    w=f+r

    signed=d.sum()/w.sum()
    content=np.abs(d).sum()/w.sum()

    d_rev=r-f
    signed_rev=d_rev.sum()/w.sum()
    content_rev=np.abs(d_rev).sum()/w.sum()

    assert np.isclose(signed_rev,-signed)
    assert np.isclose(content_rev,content)
