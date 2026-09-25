import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from strata.geometry_reconstruction.reconstruct import (
    build_adjacency, reconstruct_partition, ReconstructionConfig
)


def test_observed_pixels_are_preserved():
    polys={
        "A":Polygon([(0,0),(1,0),(1,1),(0,1)]),
        "B":Polygon([(2,0),(3,0),(3,1),(2,1)]),
    }
    edges=pd.DataFrame([{"cell_i":"A","cell_j":"B"}])
    # morphology support supplied after rasterization internally via a first pass is
    # awkward in unit tests, so call with None: envelope comes from observed mask.
    out,obs,evidence,meta,stats=reconstruct_partition(
        polys,edges,None,None,None,
        ReconstructionConfig(max_assignment_distance=5.0,raster_max_pixels=5000)
    )
    assert stats["observed_pixels_preserved"]
    assert np.all(out[obs>0]==obs[obs>0])


def test_adjacency():
    e=pd.DataFrame([{"cell_i":"A","cell_j":"B"},{"cell_i":"B","cell_j":"C"}])
    a=build_adjacency(e)
    assert "B" in a["A"]
    assert "C" in a["B"]
