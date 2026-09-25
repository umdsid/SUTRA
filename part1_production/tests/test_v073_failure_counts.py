import pandas as pd
from sutra.hierarchy.v073.full import failure_counts


def test_failure_counts_overlap_by_design():
    x=pd.DataFrame({
        "pass_molecular":[True,False],
        "pass_mechanics_support":[False,False],
        "pass_tension":[True,True],
        "pass_delta_p":[True,False],
        "pass_communication_strength":[True,True],
        "pass_communication_reciprocity":[False,True],
    })
    d=failure_counts(x)
    assert d["fail_molecular"]==1
    assert d["fail_mechanics_support"]==2
    assert d["fail_delta_p"]==1
    assert d["fail_communication_reciprocity"]==1
