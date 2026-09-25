"""Typed immutable identifiers for the STRATA hierarchy layer."""

from __future__ import annotations
from typing import NewType

SampleID = NewType("SampleID", int)
PatchID = NewType("PatchID", int)
CellID = NewType("CellID", int)
InterfaceID = NewType("InterfaceID", int)
GraphID = NewType("GraphID", int)
NodeID = NewType("NodeID", int)

__all__ = [
    "SampleID", "PatchID", "CellID", "InterfaceID", "GraphID", "NodeID",
]
