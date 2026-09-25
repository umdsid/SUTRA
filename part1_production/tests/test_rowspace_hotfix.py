import ast
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from strata_native_mechanics.rowspace_observability import (
    RowspaceConfig,
    patch_rowspace_observability,
    residual_ratio_from_basis,
    rowspace_basis_qr,
)


def test_exact_common_gauge_pressure_contrast_is_observable():
    A = csr_matrix(np.array([[1.0, -1.0]]))
    meta = {"eidx": {}, "cidx": {1: 0, 2: 1}, "bidx": {}}
    E = pd.DataFrame([{
        "interface_id": 7,
        "kind": "cell_cell",
        "cell_i": 1,
        "cell_j": 2,
    }])
    sm, V, C = patch_rowspace_observability(A, meta, E, RowspaceConfig())
    assert sm["numerical_nullity"] == 1
    assert C.iloc[0].rowspace_status == "OBSERVABLE"
    assert C.iloc[0].rowspace_residual_ratio < 1e-12


def test_explicit_residual_zero_for_rowspace_vector():
    A = csr_matrix(np.array([[1.0, -1.0]]))
    Q, meta = rowspace_basis_qr(A)
    c = np.array([1.0, -1.0])
    assert residual_ratio_from_basis(Q, c) < 1e-12


def test_cli_source_compiles():
    src = (
        Path(__file__).resolve().parents[1]
        / "src" / "strata" / "cli" / "native_rowspace_observability_v068.py"
    )
    ast.parse(src.read_text())
