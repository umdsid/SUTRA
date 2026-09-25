import json,numpy as np,pandas as pd
from pathlib import Path
from strata_hierarchy.v110.terminal_rule import evaluate_persistent
def build(c,n=14):
    d={"nodes":np.arange(1000,1000-n,-1),
       "effective_domain_number":np.ones(n)*50,
       "K80":np.ones(n)*100,
       "mass_gini":np.ones(n)*.5,
       "dominant_expression_coherence":np.ones(n),
       "dominant_spatial_coherence":np.ones(n)}
    # Previous window speed 2, recent window speed 1 -> slowdown.
    for b in c["scientific_blocks"]:
        x=np.ones(n)
        x[:5]=2.0
        d["speed_"+b]=x
    return pd.DataFrame(d)
def test_appending_future_cannot_change_past_prefix_evaluation():
    root=Path(__file__).resolve().parents[1]
    c=json.loads((root/"configs/hierarchy_v110_production_terminal_rule.json").read_text())
    h=build(c)
    a=evaluate_persistent(h.iloc[:13].copy(),c)
    h2=pd.concat([h,pd.DataFrame([{k:(999 if k!="nodes" else 900) for k in h.columns}])],ignore_index=True)
    b=evaluate_persistent(h2.iloc[:13].copy(),c)
    assert a["stop"]==b["stop"] and a["confirmations"]==b["confirmations"]
