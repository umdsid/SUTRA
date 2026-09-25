import numpy as np
import pandas as pd
from strata.mechanics.gatec_v2 import paired_bootstrap_improvement
from strata.mechanics.junction_calibration import cluster_for_tolerance

def test_bootstrap_detects_improvement():
    base=np.ones(100)
    fit=np.full(100,0.5)
    x=paired_bootstrap_improvement(fit,base,n_boot=50,seed=1)
    assert x["ci95_low"]>0
    assert x["fraction_improved"]==1.0

def test_junction_clustering_merges_near_endpoints():
    g=pd.DataFrame([
        {"endpoint0_x":0.0,"endpoint0_y":0.0,"endpoint1_x":1.0,"endpoint1_y":0.0},
        {"endpoint0_x":0.05,"endpoint0_y":0.03,"endpoint1_x":0.0,"endpoint1_y":1.0},
        {"endpoint0_x":0.02,"endpoint0_y":0.04,"endpoint1_x":-1.0,"endpoint1_y":0.0},
    ])
    gg,j,s=cluster_for_tolerance(g,0.1)
    assert (j.n_incident_interfaces>=3).any()
    assert s["fraction_degree_ge3"]>0
