import numpy as np
import pandas as pd
from strata.geometry_benchmark.components import select_primary_component,ComponentConfig

def test_detached_island_excluded():
    main=np.c_[np.arange(100)%10,np.arange(100)//10].astype(float)
    island=np.array([[100.,100.],[101.,100.],[100.,101.]])
    xy=np.vstack([main,island])
    c=pd.DataFrame({
        "cell_id":[f"C{i}" for i in range(len(xy))],
        "x_centroid":xy[:,0],
        "y_centroid":xy[:,1],
    })
    primary,meta=select_primary_component(
        c,ComponentConfig(factors=(2.,3.,4.,5.),q_nn=.99,min_plateau_jaccard=.95)
    )
    assert len(primary)>=95
    assert all(f"C{i}" not in primary for i in range(100,103))
