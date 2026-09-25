from strata_native_mechanics.synthetic import three_cell_with_hole_mask
from strata_native_mechanics.geometry import (
    GeometryConfig,extract_interfaces,extract_junctions,extract_cell_centroids
)
from strata_native_mechanics.solver import SolverConfig,solve_patch

def test_solver_builds_and_runs():
    m=three_cell_with_hole_mask()
    E,bg,bgl=extract_interfaces(m,GeometryConfig(min_interface_pixels=2))
    J=extract_junctions(m,E,bgl)
    C=extract_cell_centroids(m)
    r=solve_patch([1,2,3],E,J,C,SolverConfig(max_iter=50,min_curvature_confidence=0.0))
    assert r["identifiability"]["n_variables"]>0
    assert r["status"] in {"PASS","HOLD"}
