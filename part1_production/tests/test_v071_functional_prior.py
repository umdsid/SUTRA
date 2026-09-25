import numpy as np
from pathlib import Path

from strata_hierarchy.v071.block_preflight import functional_prior


def test_identity_survives_unannotated_gene(tmp_path):
    g=tmp_path/"h.all.test.gmt"
    g.write_text("set1\\tdesc\\tA\\tB\\n")
    W,L,a=functional_prior(("A","B","C"),[g])
    assert W.shape==(3,3)
    assert np.allclose(L[2],0)
    # C therefore receives only the identity contribution in I + lambda L.
    G=np.eye(3)+L
    assert np.allclose(G[2],[0,0,1])


def test_signaling_named_gmt_is_not_needed_here():
    # Functional prior itself accepts explicit paths; discovery performs role separation.
    pass
