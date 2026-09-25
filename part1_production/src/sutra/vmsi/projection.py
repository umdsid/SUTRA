from __future__ import annotations
import numpy as np
from scipy import ndimage


def fill_background_within_envelope(mask: np.ndarray, envelope: np.ndarray):
    """
    Pure STRATA helper used to test the confluence projection semantics.

    Parameters
    ----------
    mask
        Integer label image. Zero is background.
    envelope
        Boolean mask defining the region in which background may be assigned
        to the nearest observed cell label.

    Returns
    -------
    projected, stats
        Observed labeled pixels are preserved exactly. Only zero-valued pixels
        inside `envelope` are filled.
    """
    mask=np.asarray(mask)
    envelope=np.asarray(envelope,dtype=bool)
    if mask.shape != envelope.shape:
        raise ValueError("mask and envelope must have identical shape")

    occupied=mask>0
    background=~occupied

    # For background pixels, indices point to nearest occupied pixel because
    # distance_transform_edt measures distance to zeros of `background`.
    _, inds=ndimage.distance_transform_edt(background, return_indices=True)
    nearest=mask[tuple(inds)]

    projected=mask.copy()
    fill=envelope & background
    projected[fill]=nearest[fill]

    stats={
        "filled_pixel_fraction_of_image":float(np.mean(fill)),
        "filled_pixel_fraction_of_envelope":float(np.sum(fill)/max(np.sum(envelope),1)),
        "observed_pixels_preserved":bool(np.all(projected[occupied]==mask[occupied])),
    }
    return projected,stats
