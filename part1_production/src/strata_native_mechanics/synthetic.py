from __future__ import annotations
import numpy as np


def three_cell_with_hole_mask():
    """
    Small deterministic tissue with three cells and one internal background hole.
    """
    x=np.zeros((40,40),dtype=np.int32)
    x[5:35,5:18]=1
    x[5:20,18:35]=2
    x[20:35,18:35]=3
    # internal hole crossing cells 2/3 boundary vicinity
    yy,xx=np.ogrid[:40,:40]
    hole=(xx-26)**2+(yy-20)**2<=9
    x[hole]=0
    return x
