import numpy as np,pandas as pd
from strata_hierarchy.v091.contextual_flow import *
def toy():
    return pd.DataFrame({"super_i":[0,1,2],"super_j":[1,2,3],"n_boundary_edges":[4,4,4],"molecular_distance":[.1,.12,.11],"mechanics_support_fraction":[1.,0.,1.],"abs_tension_z":[.2,np.nan,.25],"abs_delta_p_z":[.1,np.nan,.15],"comm_support":[1.,0.,1.2],"geometry_pair_cost":[.3,np.nan,.35],"geometry_state_resolved":[True,False,True],"alpha_cost":[.3,np.nan,.35]})
def test_missing_local_fields_get_context():
    c=ContextualFlowConfig().validate();f=contextual_block_table(toy(),np.array([0,1,2,3]),1,c)
    assert np.isfinite(f.loc[1,"mechanics_effective"]) and f.loc[1,"mechanics_reliability_effective"]>0
    assert np.isfinite(f.loc[1,"geometry_effective"]) and f.loc[1,"geometry_reliability_effective"]>0
def test_zero_comm_not_boolean_veto():
    c=ContextualFlowConfig().validate();f=contextual_block_table(toy(),np.array([0,1,2,3]),1,c);s=score_candidates(f,pooled_scales([f]),c)
    assert np.isfinite(s.loc[1,"composite_merge_cost"])
def test_hops_grow():
    c=ContextualFlowConfig();assert adaptive_hops(16,c)>adaptive_hops(1,c)
def test_threshold_monotone_capped():
    c=ContextualFlowConfig();fr={"initial_threshold":1.,"maximum_threshold":2.}
    assert scale_threshold(1,fr,c)<=scale_threshold(8,fr,c)<=scale_threshold(1e9,fr,c)<=2
def test_no_target_nodes():
    c=ContextualFlowConfig();f=contextual_block_table(toy(),np.array([0,1,2,3]),1,c);s=score_candidates(f,pooled_scales([f]),c);th=freeze_cost_thresholds([s],c);a=apply_merge_admissibility(s,1,th,c)
    assert "target_nodes" not in a.columns
def test_extreme_mechanics_stays_costly():
    c=ContextualFlowConfig(context_mix=0).validate();x=toy();x.loc[0,"abs_tension_z"]=50;x.loc[0,"abs_delta_p_z"]=50
    f=contextual_block_table(x,np.array([0,1,2,3]),1,c);s=score_candidates(f,pooled_scales([f]),c)
    assert s.loc[0,"composite_merge_cost"]>s.loc[2,"composite_merge_cost"]
