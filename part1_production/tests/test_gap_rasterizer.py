from shapely.geometry import Polygon
from strata.gap_audit.core import rasterize_polygons

def test_rasterizer_without_matplotlib():
    polys={"A":Polygon([(0,0),(2,0),(2,2),(0,2)])}
    mask,meta=rasterize_polygons(polys,(0,0,2,2),10000)
    assert mask.any()
    assert mask.sum()>1
