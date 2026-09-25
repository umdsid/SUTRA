import numpy as np


def test_symmetric_channels_have_no_directional_content():
    f=np.array([1.,2.,3.])
    r=f.copy()
    d=f-r
    w=f+r
    assert np.isclose(np.abs(d).sum()/w.sum(),0.)
