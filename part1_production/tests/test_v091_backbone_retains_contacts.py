import numpy as np
from strata_hierarchy.v091.completion import build_backbone
def test_contacts_are_never_removed():
    xy=np.array([[0.,0.],[1.,0.],[100.,0.]])
    c=np.array([[0,2]])
    b=build_backbone(xy,c,k=1,gap_factor=1.1,local_k=1)
    q=b[(b.cell_i_index==0)&(b.cell_j_index==2)]
    assert len(q)==1 and bool(q.iloc[0].source_contact)
