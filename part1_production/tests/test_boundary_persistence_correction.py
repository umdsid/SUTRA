import numpy as np,pandas as pd
from scipy import ndimage
from strata_native_mechanics.corrections import (
    CorrectionConfig,classify_background_persistence
)

def test_tiny_hole_is_unstable():
    m=np.ones((30,30),dtype=np.int32)
    m[:2,:]=0;m[-2:,:]=0;m[:,:2]=0;m[:,-2:]=0
    m[14:16,14:16]=0
    bg,n=ndimage.label(m==0,structure=np.ones((3,3),np.uint8))
    rows=[]
    for k in range(1,n+1):
        ys,xs=np.nonzero(bg==k)
        touch=(ys==0).any() or (xs==0).any() or (ys==29).any() or (xs==29).any()
        rows.append({"background_component":k,
                     "kind":"exterior" if touch else "internal_gap",
                     "pixels":len(ys)})
    df=pd.DataFrame(rows)
    p=classify_background_persistence(m,bg,df,CorrectionConfig())
    assert "unstable_micro_gap" in set(p.mechanical_boundary_class)
