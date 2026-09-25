import numpy as np
import pandas as pd
from strata.preflight.core import (
    synthetic_recovery_suite, graph_components, bbox_overlap_fraction
)


def test_synthetic_suite():
    r=synthetic_recovery_suite(1)
    assert r["recovery_fraction"]==1.0


def test_graph_components():
    e={("A","B"),("B","C")}
    r=graph_components(e,["A","B","C","D"])
    assert r["n_components"]==2
    assert abs(r["isolated_fraction"]-0.25)<1e-12


def test_bbox_overlap():
    a={"available":True,"x_min":0,"x_max":10,"y_min":0,"y_max":10}
    b={"available":True,"x_min":0,"x_max":10,"y_min":0,"y_max":10}
    assert abs(bbox_overlap_fraction(a,b)-1.0)<1e-12
