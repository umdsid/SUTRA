"""Independent Level-0 freeze-audit helpers for STRATA v0.7.0."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


class Level0FreezeError(RuntimeError):
    """Raised when a frozen Level-0 invariant is violated."""


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(block_size), b""):
            h.update(block)
    return h.hexdigest()


def assert_exact_float_match(
    left: pd.DataFrame,
    right: pd.DataFrame,
    key: str,
    left_value: str,
    right_value: str,
    *,
    name: str,
) -> dict:
    """
    Exact source-to-Level0 value verification.

    These values were copied, not recomputed, so numeric tolerance is not used.
    NaNs are never accepted on the certified source side.
    """
    a = left[[key, left_value]].copy()
    b = right[[key, right_value]].copy()

    if a[key].duplicated().any():
        raise Level0FreezeError(f"{name}: duplicate Level-0 keys")
    if b[key].duplicated().any():
        raise Level0FreezeError(f"{name}: duplicate source keys")

    m = b.merge(a, on=key, how="left", validate="one_to_one")
    if m[left_value].isna().any():
        miss = int(m[left_value].isna().sum())
        raise Level0FreezeError(
            f"{name}: {miss} certified source values missing from Level-0"
        )

    x = m[left_value].to_numpy(np.float64)
    y = m[right_value].to_numpy(np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise Level0FreezeError(f"{name}: non-finite certified values")

    same = np.equal(x, y)
    if not np.all(same):
        idx = int(np.flatnonzero(~same)[0])
        raise Level0FreezeError(
            f"{name}: copied value mismatch at {key}={int(m.iloc[idx][key])}"
        )

    return {
        "n_source": int(len(b)),
        "n_matched": int(len(m)),
        "exact_match": True,
    }


def validate_csr_npz(path: str | Path, n_cells: int, interfaces: pd.DataFrame) -> dict:
    z = np.load(path)
    required = {"indptr", "indices", "interface_ids", "node_ids"}
    if not required.issubset(z.files):
        raise Level0FreezeError(
            f"CSR archive missing fields: {required - set(z.files)}"
        )

    indptr = np.asarray(z["indptr"])
    indices = np.asarray(z["indices"])
    eids = np.asarray(z["interface_ids"])
    node_ids = np.asarray(z["node_ids"])

    if len(node_ids) != n_cells:
        raise Level0FreezeError(
            f"CSR node count {len(node_ids)} != Level-0 cells {n_cells}"
        )
    if not np.array_equal(node_ids, np.arange(n_cells, dtype=node_ids.dtype)):
        raise Level0FreezeError("CSR node IDs are not canonical Level-0 cell indices")
    if len(indptr) != n_cells + 1:
        raise Level0FreezeError("CSR indptr length mismatch")
    if indptr[0] != 0 or indptr[-1] != len(indices):
        raise Level0FreezeError("CSR endpoint mismatch")
    if np.any(indptr[1:] < indptr[:-1]):
        raise Level0FreezeError("CSR indptr is not monotone")
    if len(indices) != len(eids):
        raise Level0FreezeError("CSR indices/interface_ids mismatch")
    if len(indices) and (indices.min() < 0 or indices.max() >= n_cells):
        raise Level0FreezeError("CSR neighbor index out of range")

    # Build interface lookup only for cell-cell interfaces.
    cci = interfaces[interfaces.cell_j_index.notna()].copy()
    iid_to_pair = {
        int(r.interface_id): (int(r.cell_i_index), int(r.cell_j_index))
        for r in cci.itertuples()
    }

    directed = set()
    for u in range(n_cells):
        for k in range(int(indptr[u]), int(indptr[u + 1])):
            v = int(indices[k])
            eid = int(eids[k])
            directed.add((u, v, eid))

            if eid not in iid_to_pair:
                raise Level0FreezeError(
                    f"CSR references non-cell-cell/missing interface {eid}"
                )
            a, b = iid_to_pair[eid]
            if {u, v} != {a, b}:
                raise Level0FreezeError(
                    f"CSR endpoints disagree with interface {eid}"
                )

    for u, v, eid in directed:
        if (v, u, eid) not in directed:
            raise Level0FreezeError(
                f"CSR lacks reciprocal edge for interface {eid}"
            )

    if len(directed) != 2 * len(cci):
        raise Level0FreezeError(
            "CSR directed edge count does not equal twice cell-cell interfaces"
        )

    return {
        "n_nodes": int(n_cells),
        "n_cell_cell_interfaces": int(len(cci)),
        "n_directed_edges": int(len(directed)),
        "reciprocal": True,
        "endpoint_exact": True,
    }


def validate_level0_interface_references(
    interfaces: pd.DataFrame,
    n_cells: int,
) -> dict:
    if interfaces.interface_id.duplicated().any():
        raise Level0FreezeError("duplicate Level-0 interface IDs")

    ci = interfaces.cell_i_index.to_numpy(np.int64)
    if np.any(ci < 0) or np.any(ci >= n_cells):
        raise Level0FreezeError("cell_i index out of Level-0 range")

    q = interfaces.cell_j_index.notna().to_numpy()
    cj = interfaces.loc[q, "cell_j_index"].to_numpy(np.int64)
    if len(cj) and (np.any(cj < 0) or np.any(cj >= n_cells)):
        raise Level0FreezeError("cell_j index out of Level-0 range")

    own = interfaces.owner_cell_index.to_numpy(np.int64)
    if np.any(own < 0) or np.any(own >= n_cells):
        raise Level0FreezeError("owner cell index out of Level-0 range")

    if np.any(q & (ci == interfaces.cell_j_index.fillna(-1).to_numpy(np.int64))):
        raise Level0FreezeError("self interface detected")

    return {
        "n_interfaces": int(len(interfaces)),
        "references_valid": True,
    }


def validate_retention_has_no_mechanics(
    cells: pd.DataFrame,
    interfaces: pd.DataFrame,
) -> dict:
    # Mechanics values may only occur on mechanics-owned interfaces.
    q = interfaces.owner_patch_id.astype(int) == -1
    bad_tau = q & interfaces.tension.notna()
    bad_dp = q & interfaces.delta_pressure.notna()

    if bad_tau.any():
        raise Level0FreezeError(
            f"{int(bad_tau.sum())} retention interfaces carry tension"
        )
    if bad_dp.any():
        raise Level0FreezeError(
            f"{int(bad_dp.sum())} retention interfaces carry delta-pressure"
        )

    return {
        "n_retention_cells": int((cells.patch_id.astype(int) == -1).sum()),
        "n_geometry_only_interfaces": int(q.sum()),
        "retention_mechanics_values": 0,
    }


def canonical_table_digest(df: pd.DataFrame, sort_by: list[str]) -> str:
    """
    Deterministic table-content digest independent of Parquet encoding details.
    """
    x = df.sort_values(sort_by).reset_index(drop=True)
    payload = x.to_json(
        orient="table",
        index=False,
        date_format="iso",
        double_precision=15,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


__all__ = [
    "Level0FreezeError",
    "sha256_file",
    "assert_exact_float_match",
    "validate_csr_npz",
    "validate_level0_interface_references",
    "validate_retention_has_no_mechanics",
    "canonical_table_digest",
]
