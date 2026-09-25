"""Measured-cell materialization for STRATA hierarchy Level-0."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..biology import Cell
from ..identifiers import CellID, PatchID, InterfaceID
from .validation import validate_cells


def _pick_column(df: pd.DataFrame, names: tuple[str, ...], required: bool = True):
    for c in names:
        if c in df.columns:
            return c
    if required:
        raise ValueError(f"none of required columns present: {names}")
    return None


def build_cells(
    barcodes: list[str],
    measured_cells: pd.DataFrame,
    barcode_to_patch: dict[str, int],
    barcode_to_interfaces: dict[str, tuple[int, ...]],
) -> tuple[tuple[Cell, ...], dict[str, CellID]]:
    """
    Materialize every measured cell in 10x matrix barcode order.

    Cells outside the mechanics primary component are retained with patch_id=-1
    and no mechanics interfaces.  gene_index is the immutable matrix-column
    position, not a gene identifier.
    """
    df = measured_cells.copy()
    cid = _pick_column(df, ("cell_id", "barcode", "cell", "cell_barcode"))
    xcol = _pick_column(df, ("x_centroid", "centroid_x", "x"))
    ycol = _pick_column(df, ("y_centroid", "centroid_y", "y"))
    acol = _pick_column(
        df,
        ("cell_area", "area", "cell_area_um2", "area_um2"),
        required=False,
    )

    df[cid] = df[cid].astype(str)
    if df[cid].duplicated().any():
        raise ValueError("measured cell table contains duplicate cell IDs")
    d = df.set_index(cid, drop=False)

    missing = [b for b in barcodes if b not in d.index]
    extra = sorted(set(d.index) - set(barcodes))
    if missing or extra:
        raise ValueError(
            f"matrix/cell-table mismatch: missing={len(missing)} extra={len(extra)}"
        )

    barcode_to_id = {str(b): CellID(i) for i, b in enumerate(barcodes)}
    out = []

    for i, barcode in enumerate(barcodes):
        r = d.loc[str(barcode)]
        area = float(r[acol]) if acol is not None else 1.0
        # Some Xenium exports can carry zero/NaN area; the hierarchy should not
        # invent a physical area.  Use a positive sentinel only if no area
        # column exists; otherwise fail.
        if acol is not None and (not np.isfinite(area) or area <= 0):
            raise ValueError(f"cell {barcode} has invalid measured area={area}")

        interfaces = tuple(
            InterfaceID(int(x))
            for x in barcode_to_interfaces.get(str(barcode), ())
        )
        out.append(
            Cell(
                id=CellID(i),
                patch_id=PatchID(int(barcode_to_patch.get(str(barcode), -1))),
                centroid_x=float(r[xcol]),
                centroid_y=float(r[ycol]),
                area=area,
                gene_index=i,
                interface_ids=interfaces,
            )
        )

    cells = tuple(out)
    validate_cells(cells)
    return cells, barcode_to_id


__all__ = ["build_cells"]
