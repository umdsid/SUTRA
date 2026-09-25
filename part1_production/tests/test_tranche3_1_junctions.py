import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from strata.mechanics.junctions import materialize_interface_geometry,cluster_endpoints
from strata.mechanics.vertex_solver import VertexMechanicsConfig,solve_vertex_mechanics


def sq(x0,y0,x1,y1):
    return Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])


def test_interface_arc_and_junctions():
    polys={"A":sq(0,0,1,1),"B":sq(1,0,2,1)}
    ci=pd.DataFrame([{
        "cell_i":"A","cell_j":"B","confidence_class":"high_confidence",
        "min_gap":0.0,"global_persistence_fraction":1.0
    }])
    g=materialize_interface_geometry(ci,polys,0.01)
    assert len(g)==1
    assert abs(g.interface_length.iloc[0]-1.0)<0.05
    g,j=cluster_endpoints(g,0.05)
    assert len(j)==2


def test_scale_constraint_produces_nonzero_tension():
    # nit=0 is allowed: scipy may accept the unconstrained least-squares
    # solution immediately when it already satisfies bounds.
    e=pd.DataFrame([
        {"cell_i":"A","cell_j":"B","interface_length":1.0,
         "tangent_x":0.0,"tangent_y":1.0,
         "normal_i_to_j_x":1.0,"normal_i_to_j_y":0.0,
         "junction0":0,"junction1":1}
    ])
    s=solve_vertex_mechanics(e,["A","B"],2,VertexMechanicsConfig(solver_max_iter=50))
    assert s["success"]
    assert np.isfinite(s["tension"]).all()
    assert np.isfinite(s["pressure"]).all()
    assert float(np.mean(s["tension"])) > 0.5
    assert float(np.linalg.norm(s["tension"])) > 1e-6


def test_asymmetric_network_has_finite_field():
    e=pd.DataFrame([
        {"cell_i":"A","cell_j":"B","interface_length":1.0,
         "tangent_x":1.0,"tangent_y":0.0,
         "normal_i_to_j_x":0.0,"normal_i_to_j_y":1.0,
         "junction0":0,"junction1":1},
        {"cell_i":"B","cell_j":"C","interface_length":1.4,
         "tangent_x":0.0,"tangent_y":1.0,
         "normal_i_to_j_x":1.0,"normal_i_to_j_y":0.0,
         "junction0":1,"junction1":2},
        {"cell_i":"C","cell_j":"A","interface_length":0.8,
         "tangent_x":-0.70710678,"tangent_y":0.70710678,
         "normal_i_to_j_x":-0.70710678,"normal_i_to_j_y":-0.70710678,
         "junction0":2,"junction1":0},
    ])
    s=solve_vertex_mechanics(e,["A","B","C"],3,VertexMechanicsConfig(solver_max_iter=100))
    assert s["success"]
    assert np.isfinite(s["tension"]).all()
    assert np.isfinite(s["pressure"]).all()
    assert np.isfinite(s["normalized_junction_residual"]).all()
