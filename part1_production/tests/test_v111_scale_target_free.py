import numpy as np,pandas as pd
from strata_hierarchy.v111.audit import _norm
def test_scale():
 c={'node_column_aliases':['nodes'],'eval_column_aliases':['eval'],'scale_column_aliases':['ell'],'stability_columns':['mass','expr','spatial'],'block_prefixes':['speed_']};o=_norm(pd.DataFrame({'eval':[0,1,2],'nodes':[100,80,50]}),c);assert np.isclose(o.loc[0,'ell'],0);assert o.loc[2,'ell']>o.loc[1,'ell']>0
