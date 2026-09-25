from pathlib import Path
import tempfile
import numpy as np
import tifffile
from strata_native_mechanics.synthetic import three_cell_with_hole_mask
from strata_native_mechanics.geometry_fast import FastGeometryConfig,build_or_load_cache

def test_geometry_cache_hits_second_time():
    with tempfile.TemporaryDirectory() as td:
        td=Path(td)
        p=td/"mask.tif"
        tifffile.imwrite(p,three_cell_with_hole_mask().astype(np.uint32))
        c=td/"cache"
        *_,hit1=build_or_load_cache(p,c,FastGeometryConfig(min_interface_pixels=2))
        *_,hit2=build_or_load_cache(p,c,FastGeometryConfig(min_interface_pixels=2))
        assert hit1 is False
        assert hit2 is True
