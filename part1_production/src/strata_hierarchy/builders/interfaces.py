"""Canonical interface materialization from frozen STRATA geometry/mechanics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..flags import (
    GEOMETRY_VALID,
    TENSION_CERTIFIED,
    DELTA_P_CERTIFIED,
    PROVENANCE_COMPLETE,
    CELL_CELL_INTERFACE,
    BACKGROUND_INTERFACE,
)
from ..identifiers import InterfaceID, CellID, PatchID
from ..interfaces import Interface
from .validation import validate_interfaces


def _canonical_anchor_label(row) -> int:
    """
    Geometry-only deterministic anchor used when an interface has no frozen
    mechanics owner.  This does not create mechanics ownership.
    """
    a=int(row.cell_i)
    if pd.isna(row.cell_j):
        return a
    b=int(row.cell_j)
    return min(a,b)


def build_interfaces(
    corrected_interfaces: pd.DataFrame,
    label_to_barcode: dict[int, str],
    barcode_to_cell_id: dict[str, CellID],
    certified_tensions: pd.DataFrame,
    certified_delta_p: pd.DataFrame,
    interface_owner_patch: dict[int, int],
    interface_owner_label: dict[int, int],
) -> tuple[Interface, ...]:
    """
    Build every corrected geometry interface.

    Interfaces omitted from v0.6.9 mechanics ownership are retained as
    geometry-only objects with patch_id=-1 and no mechanical value.  They are
    not promoted into the mechanics problem retroactively.
    """
    tmap={}
    if len(certified_tensions):
        for r in certified_tensions.itertuples():
            tmap[int(r.interface_id)]=float(r.tension_production)

    dpmap={}
    if len(certified_delta_p):
        for r in certified_delta_p.itertuples():
            dpmap[int(r.interface_id)]=float(r.delta_p_production)

    out=[]
    for r in corrected_interfaces.sort_values("interface_id").itertuples():
        eid=int(r.interface_id)
        li=int(r.cell_i)
        lj=None if pd.isna(r.cell_j) else int(r.cell_j)

        if li not in label_to_barcode:
            raise ValueError(f"interface {eid}: cell_i label {li} lacks barcode mapping")
        ci=barcode_to_cell_id[label_to_barcode[li]]

        if lj is None:
            cj=None
        else:
            if lj not in label_to_barcode:
                raise ValueError(f"interface {eid}: cell_j label {lj} lacks barcode mapping")
            cj=barcode_to_cell_id[label_to_barcode[lj]]

        mechanically_owned=eid in interface_owner_patch
        if mechanically_owned:
            if eid not in interface_owner_label:
                raise ValueError(f"interface {eid}: owner patch exists but owner label missing")
            owner_label=int(interface_owner_label[eid])
            patch_id=int(interface_owner_patch[eid])
        else:
            owner_label=_canonical_anchor_label(r)
            patch_id=-1

        if owner_label not in label_to_barcode:
            raise ValueError(f"interface {eid}: anchor label {owner_label} lacks barcode mapping")
        owner=barcode_to_cell_id[label_to_barcode[owner_label]]

        tau=tmap.get(eid)
        dp=dpmap.get(eid)

        # A mechanical value may only occur on an interface that the frozen
        # mechanics production stage owned.
        if not mechanically_owned and (tau is not None or dp is not None):
            raise ValueError(
                f"interface {eid}: certified mechanics found on non-owned interface"
            )

        flags=GEOMETRY_VALID|PROVENANCE_COMPLETE
        flags |= CELL_CELL_INTERFACE if cj is not None else BACKGROUND_INTERFACE
        if tau is not None:
            if not np.isfinite(tau):
                raise ValueError(f"interface {eid}: certified tension is nonfinite")
            flags |= TENSION_CERTIFIED
        if dp is not None:
            if cj is None:
                raise ValueError(f"interface {eid}: delta-p attached to background interface")
            if not np.isfinite(dp):
                raise ValueError(f"interface {eid}: certified delta-p is nonfinite")
            flags |= DELTA_P_CERTIFIED

        out.append(
            Interface(
                id=InterfaceID(eid),
                patch_id=PatchID(patch_id),
                cell_i=ci,
                cell_j=cj,
                length=float(r.length),
                curvature=float(r.curvature),
                owner=owner,
                tension=tau,
                delta_pressure=dp,
                flags=int(flags),
            )
        )

    interfaces=tuple(out)
    validate_interfaces(interfaces,{int(v) for v in barcode_to_cell_id.values()})
    return interfaces


__all__=["build_interfaces"]
