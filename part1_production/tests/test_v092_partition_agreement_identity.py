import numpy as np
from strata_hierarchy.v092.sweep import partition_pair_agreement,label_overlap_jaccard

def test_identical_partitions_score_one():
    a=np.array([0,0,2,3,3])
    assert partition_pair_agreement(a,a)==1.0
    assert label_overlap_jaccard(a,a)==1.0
