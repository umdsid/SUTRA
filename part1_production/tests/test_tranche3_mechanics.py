import numpy as np, pandas as pd
from strata.mechanics.inference import MechanicsConfig, select_mechanics_edges, solve_mechanics, attach
from strata.mechanics.stress import derive_stress

def toy():
    return pd.DataFrame([
        {"cell_i":"A","cell_j":"B","shared_support_at_chosen":1.0,"normal_i_to_j_x":1.0,"normal_i_to_j_y":0.0,"confidence_class":"high_confidence"},
        {"cell_i":"B","cell_j":"C","shared_support_at_chosen":1.0,"normal_i_to_j_x":1.0,"normal_i_to_j_y":0.0,"confidence_class":"admissible"},
    ])

def test_confidence_selection():
    assert len(select_mechanics_edges(toy(),MechanicsConfig(min_confidence_class="high_confidence")))==1
    assert len(select_mechanics_edges(toy(),MechanicsConfig(min_confidence_class="admissible")))==2

def test_solver_finite():
    s=solve_mechanics(toy(),["A","B","C"],MechanicsConfig())
    assert np.isfinite(s["tension"]).all()
    assert np.isfinite(s["pressure"]).all()
    assert np.isfinite(s["cell_residual"]).all()

def test_stress_derived():
    e=toy(); s=solve_mechanics(e,["A","B","C"],MechanicsConfig())
    me,p=attach(e,["A","B","C"],s)
    cells=pd.DataFrame({"cell_id":["A","B","C"],"cell_area":[10.,10.,10.]})
    st=derive_stress(cells,me,p)
    assert len(st)==3 and st["stress_valid"].all()
    assert np.isfinite(st["stress_anisotropy"]).all()
