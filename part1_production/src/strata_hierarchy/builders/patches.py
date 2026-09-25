"""Patch materialization matching frozen mechanics cores plus Level-0 retention."""

from __future__ import annotations

from ..biology import Patch
from ..identifiers import CellID, PatchID, GraphID, InterfaceID
from .validation import validate_patch


def build_patches(
    core_labels: list[set[int]],
    label_to_barcode: dict[int, str],
    barcode_to_cell_id: dict[str, CellID],
    interface_owner_patch: dict[int, int],
    graph_id: GraphID,
    retention_cell_ids: tuple[CellID, ...] = (),
    retention_interface_ids: tuple[InterfaceID, ...] = (),
    *,
    detached_cell_ids: tuple[CellID, ...] | None = None,
) -> tuple[Patch, ...]:
    """
    Build one immutable Patch per mechanics core plus patch -1.

    `retention_cell_ids` is the current API.

    `detached_cell_ids` is retained as a backward-compatible alias for the
    original Step-3 API.  It has identical semantics when supplied alone.
    Supplying both non-empty forms is rejected to avoid ambiguous duplication.

    Patch -1 is a Level-0 retention container, not a mechanics patch.  It can
    hold:
      - measured cells outside the mechanics primary component;
      - measured primary cells absent from the cell-cell contact partition;
      - geometry-only interfaces with no mechanics owner.
    """
    if detached_cell_ids is not None:
        if retention_cell_ids:
            raise ValueError(
                "use either retention_cell_ids or detached_cell_ids, not both"
            )
        retention_cell_ids = tuple(detached_cell_ids)

    owned: dict[int, list[int]] = {}
    for eid, pid in interface_owner_patch.items():
        owned.setdefault(int(pid), []).append(int(eid))

    out = []
    for pid, labels in enumerate(core_labels):
        cells = tuple(
            sorted(
                (
                    barcode_to_cell_id[label_to_barcode[int(lab)]]
                    for lab in labels
                ),
                key=int,
            )
        )
        eids = tuple(
            InterfaceID(e) for e in sorted(owned.get(pid, []))
        )
        p = Patch(
            id=PatchID(pid),
            graph_id=graph_id,
            cell_ids=cells,
            interface_ids=eids,
            n_cells=len(cells),
            n_interfaces=len(eids),
        )
        validate_patch(p)
        out.append(p)

    retention_cells = tuple(sorted(set(retention_cell_ids), key=int))
    retention_interfaces = tuple(
        sorted(set(retention_interface_ids), key=int)
    )

    if retention_cells or retention_interfaces:
        p = Patch(
            id=PatchID(-1),
            graph_id=graph_id,
            cell_ids=retention_cells,
            interface_ids=retention_interfaces,
            n_cells=len(retention_cells),
            n_interfaces=len(retention_interfaces),
        )
        validate_patch(p)
        out.append(p)

    return tuple(out)


__all__ = ["build_patches"]
