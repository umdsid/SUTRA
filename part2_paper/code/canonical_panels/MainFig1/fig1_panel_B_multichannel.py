#!/usr/bin/env python3

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import Normalize, LogNorm, PowerNorm


# ============================================================
# Paths
# ============================================================

ROOT = Path(
    os.environ.get(
        "SUTRA_ROOT",
        str(Path.home() / "Desktop" / "SUTRA"),
    )
)
SAMPLE = "healthy_reference"

CELL_FILE = ROOT / "data" / SAMPLE / "cells.parquet"
BOUNDARY_FILE = ROOT / "data" / SAMPLE / "cell_boundaries.parquet"

HIERARCHY_ROOT = Path(
    os.environ.get(
        "SUTRA_HIERARCHY_ROOT",
        str(
            ROOT
            / "results"
            / "hierarchy_v0911_specimen_local_contextual_flow"
        ),
    )
)

LEVEL0_FILE = (
    HIERARCHY_ROOT
    / "ledger"
    / SAMPLE
    / "candidate_boundaries"
    / "step_000000.parquet"
)

FIGROOT = Path(
    os.environ.get(
        "SUTRA_FIG1_OUT",
        str(ROOT / "figures" / "fig1"),
    )
)
PANELDIR = FIGROOT / "panels"
SOURCEDIR = FIGROOT / "source_data"

PANELDIR.mkdir(parents=True, exist_ok=True)
SOURCEDIR.mkdir(parents=True, exist_ok=True)

CROP_FILE = SOURCEDIR / "panel_A_crop.json"

OUT_PDF = PANELDIR / "panel_B.pdf"
OUT_PNG = PANELDIR / "panel_B.png"

SOURCE_CSV = SOURCEDIR / "panel_B_level0_crop_edges.csv"
SOURCE_JSON = SOURCEDIR / "panel_B_metadata.json"


# ============================================================
# Frozen production quantities
#
# These are deliberately the effective Level-0 quantities
# from the final v0.9.1.1 production candidate ledger.
# ============================================================

CHANNELS = [
    {
        "column": "molecular_effective",
        "label": "Molecular",
        "cmap": "plasma",
        "norm": "linear",
        "qlo": 0.02,
        "qhi": 0.98,
    },
    {
        "column": "mechanics_effective",
        "label": "Mechanics",
        "cmap": "inferno",
        "norm": "log",
        "qlo": 0.04,
        "qhi": 0.96,
    },
    {
        "column": "communication_support_effective",
        "label": "Communication",
        "cmap": "viridis",
        "norm": "linear",
        "qlo": 0.02,
        "qhi": 0.98,
    },
    {
        "column": "geometry_effective",
        "label": "Geometry",
        "cmap": "magma",
        "norm": "log",
        "qlo": 0.04,
        "qhi": 0.96,
    },
    {
        "column": "topology_effective",
        "label": "Topology",
        "cmap": "turbo",
        "norm": "linear",
        "qlo": 0.01,
        "qhi": 0.99,
    },
    {
        "column": "composite_merge_cost",
        "label": "Composite merger cost",
        "cmap": "YlOrRd",
        "norm": "power",
        "qlo": 0.02,
        "qhi": 0.98,
        "gamma": 0.70,
    },
]


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


# ============================================================
# Load crop selected by Panel A
# ============================================================

if not CROP_FILE.exists():
    raise SystemExit(
        "ERROR: Panel-A crop not found.\n"
        "Run fig1_panel_A_level0_tissue.py first."
    )

crop = json.loads(CROP_FILE.read_text())

xmin = float(crop["xmin"])
xmax = float(crop["xmax"])
ymin = float(crop["ymin"])
ymax = float(crop["ymax"])

print("Using frozen Panel-A crop:")
print(f"  x: {xmin:.3f} .. {xmax:.3f}")
print(f"  y: {ymin:.3f} .. {ymax:.3f}")


# ============================================================
# Load Level-0 cells
#
# Production identity audits establish Level-0 supernode IDs
# as positional cell indices. We enforce the valid range here.
# ============================================================

cells = pq.read_table(
    CELL_FILE,
    columns=["x_centroid", "y_centroid"],
).to_pandas()

xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)

N0 = len(xy)

print(f"Level-0 cells: {N0:,}")

in_crop = (
    (xy[:, 0] >= xmin)
    & (xy[:, 0] <= xmax)
    & (xy[:, 1] >= ymin)
    & (xy[:, 1] <= ymax)
)

crop_indices = np.flatnonzero(in_crop)

print(f"Cells in Panel-B crop: {len(crop_indices):,}")


# ============================================================
# Load true cell polygons for background
# ============================================================

bd = pq.read_table(
    BOUNDARY_FILE,
    columns=["cell_id", "vertex_x", "vertex_y"],
).to_pandas()

# We only need polygons lying in/near the frozen crop.
# Cheap prefilter on vertices makes grouping faster.
margin_x = 0.03 * (xmax - xmin)
margin_y = 0.03 * (ymax - ymin)

bd = bd[
    (bd["vertex_x"] >= xmin - margin_x)
    & (bd["vertex_x"] <= xmax + margin_x)
    & (bd["vertex_y"] >= ymin - margin_y)
    & (bd["vertex_y"] <= ymax + margin_y)
]

polygons = []

for _, g in bd.groupby("cell_id", sort=False):
    p = g[["vertex_x", "vertex_y"]].to_numpy(dtype=float)

    if len(p) >= 3:
        polygons.append(p)


# ============================================================
# Load final production Level-0 candidate interfaces
# ============================================================

needed = [
    "super_i",
    "super_j",
    "n_boundary_edges",
    *[cfg["column"] for cfg in CHANNELS],
    "mechanics_available_local",
    "admissible",
]

tab = pq.read_table(
    LEVEL0_FILE,
    columns=needed,
).to_pandas()

# Basic identity/range guards
if tab["super_i"].min() < 0 or tab["super_j"].min() < 0:
    raise RuntimeError("Negative Level-0 supernode ID encountered.")

if tab["super_i"].max() >= N0 or tab["super_j"].max() >= N0:
    raise RuntimeError(
        "Level-0 supernode IDs exceed cells.parquet row range."
    )

# Keep edges for which BOTH cells lie in the exact Panel-A crop.
mask = (
    in_crop[tab["super_i"].to_numpy(dtype=int)]
    & in_crop[tab["super_j"].to_numpy(dtype=int)]
)

edges = tab.loc[mask].copy()

print(f"Candidate interfaces in crop: {len(edges):,}")


# ============================================================
# Attach endpoint coordinates
# ============================================================

ii = edges["super_i"].to_numpy(dtype=int)
jj = edges["super_j"].to_numpy(dtype=int)

edges["x_i"] = xy[ii, 0]
edges["y_i"] = xy[ii, 1]
edges["x_j"] = xy[jj, 0]
edges["y_j"] = xy[jj, 1]


# ============================================================
# Preserve a compact figure-source table
#
# This is intentionally small enough to ship with the future
# GitHub reproduction package without shipping the 81-GB ledger.
# ============================================================

save_cols = [
    "super_i",
    "super_j",
    "x_i",
    "y_i",
    "x_j",
    "y_j",
    "n_boundary_edges",
    "mechanics_available_local",
    "admissible",
    *[cfg["column"] for cfg in CHANNELS],
]

edges[save_cols].to_csv(
    SOURCE_CSV,
    index=False,
)

print("Saved compact source data:")
print(" ", SOURCE_CSV)


# ============================================================
# Robust plotting limits
# ============================================================

limits = {}

for cfg in CHANNELS:
    col = cfg["column"]
    label = cfg["label"]

    v = edges[col].to_numpy(dtype=float)
    v = v[np.isfinite(v)]

    if cfg["norm"] == "log":
        v = v[v > 0]

    if len(v) == 0:
        limits[col] = (1.0, 2.0)
        continue

    # Channel-specific robust display range.
    # Topology occupies a very narrow numerical interval, so its
    # nearly full empirical range is expanded across the colormap.
    qlo = cfg.get("qlo", 0.01)
    qhi = cfg.get("qhi", 0.99)

    lo, hi = np.quantile(v, [qlo, qhi])

    if cfg["norm"] == "log":
        lo = max(lo, np.min(v[v > 0]))

    if not np.isfinite(lo) or not np.isfinite(hi):
        lo = np.nanmin(v)
        hi = np.nanmax(v)

    if hi <= lo:
        hi = lo * 1.01 if lo > 0 else lo + 1.0

    limits[col] = (float(lo), float(hi))

    print(
        f"{label:24s}: "
        f"{cfg['norm']:7s} range {lo:.5g} .. {hi:.5g}"
    )


# ============================================================
# Edge segments
# ============================================================

segments = np.stack(
    [
        np.column_stack(
            [edges["x_i"].to_numpy(), edges["y_i"].to_numpy()]
        ),
        np.column_stack(
            [edges["x_j"].to_numpy(), edges["y_j"].to_numpy()]
        ),
    ],
    axis=1,
)


# ============================================================
# Plot helper
# ============================================================

def add_cell_background(ax):
    pc = PolyCollection(
        polygons,
        facecolors="white",
        edgecolors="0.83",
        linewidths=0.12,
        antialiased=True,
        zorder=0,
    )
    ax.add_collection(pc)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for s in ax.spines.values():
        s.set_visible(False)


def add_channel(ax, cfg):
    col = cfg["column"]
    title = cfg["label"]
    cmap = cfg["cmap"]

    add_cell_background(ax)

    vals = edges[col].to_numpy(dtype=float)

    finite = np.isfinite(vals)

    # Log-normalized channels require strictly positive values.
    if cfg["norm"] == "log":
        finite &= vals > 0

    # Missing/unresolved values remain explicit as neutral edges.
    if np.any(~finite):
        lc_missing = LineCollection(
            segments[~finite],
            colors="0.82",
            linewidths=0.28,
            alpha=0.55,
            zorder=1,
        )
        ax.add_collection(lc_missing)

    lo, hi = limits[col]

    if cfg["norm"] == "log":
        norm = LogNorm(
            vmin=lo,
            vmax=hi,
            clip=True,
        )

    elif cfg["norm"] == "power":
        norm = PowerNorm(
            gamma=cfg.get("gamma", 0.5),
            vmin=lo,
            vmax=hi,
            clip=True,
        )

    else:
        norm = Normalize(
            vmin=lo,
            vmax=hi,
            clip=True,
        )

    lc = LineCollection(
        segments[finite],
        array=vals[finite],
        cmap=cmap,
        norm=norm,
        linewidths=0.72,
        alpha=0.94,
        zorder=2,
    )

    ax.add_collection(lc)

    ax.set_title(
        title,
        fontsize=7.8,
        pad=2.5,
        weight="bold",
    )

    cb = plt.colorbar(
        lc,
        ax=ax,
        fraction=0.046,
        pad=0.015,
    )

    cb.ax.tick_params(
        labelsize=5.5,
        length=2,
        width=0.5,
    )

    cb.outline.set_linewidth(0.4)


# ============================================================
# Figure
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(8.3, 5.45),
)

for ax, cfg in zip(
    axes.ravel(),
    CHANNELS,
):
    add_channel(ax, cfg)

fig.subplots_adjust(
    left=0.018,
    right=0.985,
    bottom=0.025,
    top=0.955,
    wspace=0.16,
    hspace=0.12,
)


# ============================================================
# Metadata
# ============================================================

metadata = {
    "sample": SAMPLE,
    "level0_cells_total": int(N0),
    "cells_in_crop": int(len(crop_indices)),
    "candidate_interfaces_total": int(len(tab)),
    "candidate_interfaces_in_crop": int(len(edges)),
    "ledger": str(LEVEL0_FILE),
    "crop": crop,
    "channels": [
        {
            **cfg,
            "robust_vmin": limits[cfg["column"]][0],
            "robust_vmax": limits[cfg["column"]][1],
        }
        for cfg in CHANNELS
    ],
}

SOURCE_JSON.write_text(
    json.dumps(metadata, indent=2) + "\n"
)


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.02,
)

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.02,
)

plt.close(fig)

print()
print("PANEL B COMPLETE")
print("PDF:", OUT_PDF)
print("PNG:", OUT_PNG)
print("Metadata:", SOURCE_JSON)
