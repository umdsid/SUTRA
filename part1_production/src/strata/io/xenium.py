from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp


def _decode(values) -> list[str]:
    out: list[str] = []
    for x in values:
        if isinstance(x, bytes):
            out.append(x.decode("utf-8"))
        else:
            out.append(str(x))
    return out


@dataclass
class MatrixMetadata:
    barcodes: list[str]
    feature_ids: list[str]
    feature_names: list[str]
    feature_types: list[str]
    shape: tuple[int, int]
    nnz: int
    cell_totals: np.ndarray | None = None
    feature_totals: np.ndarray | None = None


def read_10x_h5_metadata(path: str | Path, audit_counts: bool = True) -> MatrixMetadata:
    p = Path(path)
    with h5py.File(p, "r") as h:
        if "matrix" not in h:
            raise ValueError(f"Expected /matrix group in {p}")
        g = h["matrix"]
        required = ["barcodes", "data", "indices", "indptr", "shape", "features"]
        missing = [x for x in required if x not in g]
        if missing:
            raise ValueError(f"Missing 10x matrix fields in {p}: {missing}")

        barcodes = _decode(g["barcodes"][:])
        shape_raw = tuple(int(x) for x in g["shape"][:])
        if len(shape_raw) != 2:
            raise ValueError(f"Invalid matrix shape in {p}: {shape_raw}")
        n_features, n_cells = shape_raw
        if n_cells != len(barcodes):
            raise ValueError(f"Barcode count {len(barcodes)} != matrix cell dimension {n_cells}")

        fg = g["features"]
        id_key = "id" if "id" in fg else None
        name_key = "name" if "name" in fg else None
        type_key = "feature_type" if "feature_type" in fg else None
        if id_key is None or name_key is None:
            raise ValueError(f"Feature id/name missing in {p}")

        feature_ids = _decode(fg[id_key][:])
        feature_names = _decode(fg[name_key][:])
        feature_types = _decode(fg[type_key][:]) if type_key else ["unknown"] * n_features
        if not (len(feature_ids) == len(feature_names) == len(feature_types) == n_features):
            raise ValueError("Feature metadata lengths do not match matrix feature dimension")

        data = g["data"][:]
        indices = g["indices"][:]
        indptr = g["indptr"][:]
        nnz = int(len(data))
        cell_totals = feature_totals = None
        if audit_counts:
            mat = sp.csc_matrix((data, indices, indptr), shape=(n_features, n_cells))
            cell_totals = np.asarray(mat.sum(axis=0)).ravel()
            feature_totals = np.asarray(mat.sum(axis=1)).ravel()

    return MatrixMetadata(
        barcodes=barcodes,
        feature_ids=feature_ids,
        feature_names=feature_names,
        feature_types=feature_types,
        shape=(n_features, n_cells),
        nnz=nnz,
        cell_totals=cell_totals,
        feature_totals=feature_totals,
    )


def read_cells(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if p.name.endswith(".parquet") or p.name.endswith(".parquet.gz"):
        return pd.read_parquet(p)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Unsupported cell-table format: {p}")


def identify_cell_id_column(df: pd.DataFrame) -> str:
    candidates = ("cell_id", "barcode", "cell", "cell_barcode")
    for c in candidates:
        if c in df.columns:
            return c
    if df.index.name in candidates:
        return df.index.name
    raise ValueError(f"Could not identify cell identifier column. Columns: {list(df.columns)}")
