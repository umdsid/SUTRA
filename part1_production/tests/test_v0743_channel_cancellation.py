import numpy as np


def test_opposing_channels_have_zero_net_but_nonzero_content():
    # Channel 1: +1 directional contribution, channel 2: -1.
    delta=np.array([1.,-1.])
    weight=np.array([1.,1.])
    net=abs(delta.sum())/weight.sum()
    content=np.abs(delta).sum()/weight.sum()
    cancellation=1-abs(delta.sum())/np.abs(delta).sum()
    assert np.isclose(net,0.)
    assert np.isclose(content,1.)
    assert np.isclose(cancellation,1.)
