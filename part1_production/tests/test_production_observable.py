import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from strata_native_mechanics.production_observable import (
    ProductionSolveConfig,
    solve_patch_pair,
    relative_solver_disagreement,
    interface_owner_map,
)


def test_observable_contrast_invariant_under_gauge():
    # p1 - p2 = 2; common pressure shift is free.
    A = csr_matrix(np.array([[1.0, -1.0]]))
    b = np.array([2.0])
    cfg = ProductionSolveConfig()
    x1, x2, meta = solve_patch_pair(A, b, cfg)
    d1 = x1[0] - x1[1]
    d2 = x2[0] - x2[1]
    assert abs(d1 - 2.0) < 1e-9
    assert abs(d2 - 2.0) < 1e-9
    assert relative_solver_disagreement(d1, d2) < 1e-10


def test_core_owner_does_not_depend_on_observability():
    cores = [{10, 20}, {30, 40}]
    E = pd.DataFrame([
        {"interface_id": 1, "kind": "cell_cell", "cell_i": 20, "cell_j": 30},
        {"interface_id": 2, "kind": "cell_cell", "cell_i": 40, "cell_j": 10},
        {"interface_id": 3, "kind": "cell_background", "cell_i": 30, "cell_j": -1},
    ])
    owners, cmap = interface_owner_map(cores, E)
    d = dict(zip(owners.interface_id, owners.owner_patch_id))
    assert d[1] == 0  # smaller incident cell is 20
    assert d[2] == 0  # smaller incident cell is 10
    assert d[3] == 1


def test_owner_unique():
    cores = [{1,2},{3,4}]
    E = pd.DataFrame([
        {"interface_id": 1, "kind": "cell_cell", "cell_i": 1, "cell_j": 3},
        {"interface_id": 2, "kind": "cell_cell", "cell_i": 2, "cell_j": 4},
    ])
    owners, _ = interface_owner_map(cores, E)
    assert owners.interface_id.is_unique
