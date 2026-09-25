import numpy as np
from strata.vmsi.projection import fill_background_within_envelope


def test_preserves_observed_labels_and_fills_only_envelope():
    mask=np.zeros((20,20),dtype=np.int32)
    mask[5:15,3:8]=1
    mask[5:15,12:17]=2

    # Synthetic "tissue envelope" spanning both cells and the internal gap,
    # but not the exterior background.
    envelope=np.zeros_like(mask,dtype=bool)
    envelope[5:15,3:17]=True

    out,stats=fill_background_within_envelope(mask,envelope)

    assert np.all(out[mask>0]==mask[mask>0])
    assert np.all(out[5:15,8:12]>0)
    assert np.all(out[:4,:]==0)
    assert np.all(out[16:,:]==0)
    assert stats["observed_pixels_preserved"]


def test_shape_mismatch_rejected():
    mask=np.zeros((3,3),dtype=np.int32)
    envelope=np.zeros((4,4),dtype=bool)
    try:
        fill_background_within_envelope(mask,envelope)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
