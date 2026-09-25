import numpy as np, pandas as pd
from strata.mechanics.identifiability import effective_support_nonnegative,gini_nonnegative,top_overlap,compare_fields,certify_identifiability

def test_effective_support_uniform():
    x=np.ones(100); assert abs(effective_support_nonnegative(x)-100)<1e-9; assert abs(gini_nonnegative(x))<1e-12

def test_top_overlap_identical():
    x=np.arange(100,dtype=float); assert top_overlap(x,x,0.05)==1.0

def test_compare_identical_fields():
    x=np.linspace(0,1,100); p=np.linspace(-1,1,50); c=compare_fields(x,p,x.copy(),p.copy()); assert c['tension_spearman']>0.999 and c['pressure_spearman']>0.999 and c['top5_tension_overlap']==1.0 and c['relative_tension_l2']<1e-12

def test_audit_passes_stable_case():
    s={'tension_effective_support_fraction':0.2,'tension_mass_top1pct':0.2}; st={'structural_rank_fraction_of_unknowns':0.8}; p=pd.DataFrame({'tension_spearman':[0.95,0.92],'pressure_spearman':[0.90,0.88],'top5_tension_overlap':[0.85,0.82],'relative_tension_l2':[0.2,0.3],'relative_pressure_l2':[0.4,0.5]}); assert certify_identifiability(s,st,p)['status']=='PASS'
