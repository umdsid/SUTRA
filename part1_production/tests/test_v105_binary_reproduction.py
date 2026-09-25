import pandas as pd
from strata_hierarchy.v105.pareto_audit import audit_v104_binary_failures
def test_binary_failure_reason_is_explicit():
    x=pd.DataFrame([{
      "basin_landmarks":5,"lifetime_ell":.1,"basin_depth":.2,
      "cross_block_coherence":.4,"landmark_strength":.8,
      "hierarchy_landmark":False
    }])
    c={"v104_min_basin_landmarks":3,"v104_min_cross_block_coherence":.5}
    y=audit_v104_binary_failures(x,c).iloc[0]
    assert y.binary_failure_reasons=="cross_block_coherence"
    assert y.v104_flag_agrees
