#!/usr/bin/env python3

import os
from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import hsv_to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


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

DATA = ROOT / "data" / SAMPLE

CELL_FILE = DATA / "cells.parquet"
BOUNDARY_FILE = DATA / "cell_boundaries.parquet"

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

LEDGER = (
    HIERARCHY_ROOT
    / "ledger"
    / SAMPLE
)

MERGE0 = LEDGER / "merge_events" / "step_000000.parquet"
LABELDIR = LEDGER / "label_checkpoints"

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

PANEL_A_CROP = SOURCEDIR / "panel_A_crop.json"

OUT_PDF = PANELDIR / "panel_C.pdf"
OUT_PNG = PANELDIR / "panel_C.png"

REGION_JSON = SOURCEDIR / "panel_C_region.json"
INTERFACE_JSON = SOURCEDIR / "panel_C_interface.json"
STATES_CSV = SOURCEDIR / "panel_C_hierarchy_states.csv"


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",

    "font.weight": "bold",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "text.parse_math": True,
})


# ============================================================
# Channel display choices
# ============================================================

CHANNELS = [
    ("molecular_effective", "Molecular", "#e83e8c"),
    ("mechanics_effective", "Mechanics", "#f28e2b"),
    ("communication_support_effective", "Communication", "#24a148"),
    ("geometry_effective", "Geometry", "#2f80ed"),
]


# ============================================================
# Helpers
# ============================================================

GOLDEN = 0.6180339887498949


def categorical_colors(labels):
    """
    Deterministic bright categorical colors from integer labels.
    Same label -> same color.
    """
    labels = np.asarray(labels, dtype=np.int64)

    h = (labels * GOLDEN) % 1.0

    sat = 0.62 + 0.22 * (
        0.5 + 0.5 * np.sin(labels * 0.713)
    )

    val = 0.88 + 0.10 * (
        0.5 + 0.5 * np.cos(labels * 0.431)
    )

    hsv = np.column_stack([h, sat, val])
    return hsv_to_rgb(hsv)


def polygon_centroid(poly):
    return np.mean(poly, axis=0)


def add_arrow(fig, ax1, ax2):
    """
    Thick figure-coordinate arrow constrained to whitespace
    between axes so it never intersects a plotted panel.
    """
    b1 = ax1.get_position()
    b2 = ax2.get_position()

    y = 0.5 * (b1.y0 + b1.y1)

    gap = b2.x0 - b1.x1
    mid = 0.5 * (b1.x1 + b2.x0)

    half = min(0.015, 0.34 * gap)

    p0 = (
        mid - half,
        y,
    )

    p1 = (
        mid + half,
        y,
    )

    if p1[0] <= p0[0]:
        return

    a = FancyArrowPatch(
        p0,
        p1,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=17,
        linewidth=2.3,
        color="0.18",
        shrinkA=0,
        shrinkB=0,
        zorder=20,
        clip_on=False,
    )

    fig.add_artist(a)


def normalized_progress(labels, n0, nf):
    nk = len(np.unique(labels))

    if n0 == nf:
        return 1.0

    return (n0 - nk) / (n0 - nf)


# ============================================================
# Read Level-0 cells
#
# Previous exact identity audit established:
# cells.parquet row position == Level-0 cell_index.
# ============================================================

cells = pq.read_table(
    CELL_FILE,
    columns=["cell_id", "x_centroid", "y_centroid"],
).to_pandas()

N0 = len(cells)

xy = cells[["x_centroid", "y_centroid"]].to_numpy(dtype=float)
cell_ids = cells["cell_id"].tolist()

print(f"Level-0 cells: {N0:,}")


# ============================================================
# Read actual cell polygons and map cell_id -> polygon
# ============================================================

bd = pq.read_table(
    BOUNDARY_FILE,
    columns=["cell_id", "vertex_x", "vertex_y"],
).to_pandas()

poly_by_id = {}

for cid, g in bd.groupby("cell_id", sort=False):
    p = g[["vertex_x", "vertex_y"]].to_numpy(dtype=float)

    if len(p) >= 3:
        poly_by_id[cid] = p


polygons = []

valid_cell_indices = []

for i, cid in enumerate(cell_ids):
    p = poly_by_id.get(cid)

    if p is not None and len(p) >= 3:
        polygons.append(p)
        valid_cell_indices.append(i)

print(f"Matched cell polygons: {len(polygons):,}")

if len(polygons) != N0:
    print(
        "WARNING:",
        N0 - len(polygons),
        "cells lacked a usable polygon."
    )


# Direct index -> polygon mapping
poly_by_index = {
    i: poly_by_id[cell_ids[i]]
    for i in range(N0)
    if cell_ids[i] in poly_by_id
}


# ============================================================
# Panel-A crop bounds
# ============================================================

if not PANEL_A_CROP.exists():
    raise SystemExit(
        "ERROR: panel_A_crop.json missing. "
        "Run Panel A first."
    )

a_crop = json.loads(PANEL_A_CROP.read_text())

Axmin = float(a_crop["xmin"])
Axmax = float(a_crop["xmax"])
Aymin = float(a_crop["ymin"])
Aymax = float(a_crop["ymax"])

Awidth = Axmax - Axmin
Aheight = Aymax - Aymin


# ============================================================
# Exact terminal labels
# ============================================================

checkpoint_files = sorted(
    LABELDIR.glob("labels_*.npz")
)

if not checkpoint_files:
    raise SystemExit("No label checkpoints found.")

checkpoints = []

for p in checkpoint_files:
    step = int(p.stem.split("_")[1])

    with np.load(p, allow_pickle=False) as z:
        labels = z["labels"].astype(np.int64)

    if len(labels) != N0:
        raise RuntimeError(
            f"Checkpoint length mismatch: {p}"
        )

    checkpoints.append(
        {
            "path": p,
            "step": step,
            "labels": labels,
            "nodes": len(np.unique(labels)),
        }
    )

final_labels = checkpoints[-1]["labels"]
Nf = checkpoints[-1]["nodes"]

print(f"Terminal nodes: {Nf:,}")


# ============================================================
# Load the actual accepted Level-0 merges
# ============================================================

merge0_cols = [
    "super_i",
    "super_j",
    "survivor_node",
    "removed_node",
    "composite_merge_cost",
    "merge_cost_threshold",
    "molecular_effective",
    "mechanics_effective",
    "communication_support_effective",
    "geometry_effective",
    "topology_effective",
]

m0 = pq.read_table(
    MERGE0,
    columns=merge0_cols,
).to_pandas()

print(f"Accepted step-0 merges: {len(m0):,}")


# ============================================================
# Select representative C region reproducibly
#
# Candidate regions are centered on actual accepted Level-0
# interfaces. We score them for:
#   - ~80 Level-0 cells
#   - substantial but non-degenerate terminal consolidation
#   - multiple terminal units
#   - no single terminal unit dominating the square
#
# The region stays inside the already-frozen Panel-A crop.
# ============================================================

# Search multiple physical window sizes around every genuine
# accepted Level-0 merger. Panel C does not need to lie inside
# Panel A's display crop; it only needs to be real tissue.
#
# Using several window sizes also lets the selection procedure
# find a readable neighborhood rather than imposing an arbitrary
# fixed square.

base_side = min(Awidth, Aheight)

side_fractions = [
    0.12,
    0.15,
    0.18,
    0.21,
    0.24,
    0.27,
    0.30,
    0.37,
    0.38,
    0.42,
]

# Full specimen centroid range is used only to avoid pathological
# windows lying almost completely outside the tissue coordinate
# domain.
Xmin_all = float(np.nanmin(xy[:, 0]))
Xmax_all = float(np.nanmax(xy[:, 0]))
Ymin_all = float(np.nanmin(xy[:, 1]))
Ymax_all = float(np.nanmax(xy[:, 1]))

candidates = []

for row_idx, row in m0.iterrows():

    i = int(row["super_i"])
    j = int(row["super_j"])

    center = 0.5 * (xy[i] + xy[j])

    cx, cy = center

    for side_fraction in side_fractions:

        side = side_fraction * base_side

        xmin = cx - side / 2
        xmax = cx + side / 2
        ymin = cy - side / 2
        ymax = cy + side / 2

        # Require the accepted interface itself to be comfortably
        # inside the displayed square.
        if not (
            xmin <= xy[i, 0] <= xmax
            and ymin <= xy[i, 1] <= ymax
            and xmin <= xy[j, 0] <= xmax
            and ymin <= xy[j, 1] <= ymax
        ):
            continue

        mask = (
            (xy[:, 0] >= xmin)
            & (xy[:, 0] <= xmax)
            & (xy[:, 1] >= ymin)
            & (xy[:, 1] <= ymax)
        )

        ids = np.flatnonzero(mask)

        n_cells = len(ids)

        # Too sparse to communicate tissue organization.
        if n_cells < 30:
            continue

        terminal = final_labels[ids]

        vals, counts = np.unique(
            terminal,
            return_counts=True,
        )

        n_terminal = len(vals)

        reduction = 1.0 - n_terminal / n_cells

        largest_fraction = counts.max() / n_cells

        # Fraction of the display window overlapping the global
        # coordinate bounding box. This is only a weak aesthetic
        # penalty against large blank margins.
        overlap_x = max(
            0.0,
            min(xmax, Xmax_all) - max(xmin, Xmin_all),
        )
        overlap_y = max(
            0.0,
            min(ymax, Ymax_all) - max(ymin, Ymin_all),
        )

        window_area = side * side

        coordinate_overlap_fraction = (
            overlap_x * overlap_y / window_area
            if window_area > 0
            else 0.0
        )

        # Desired visual characteristics:
        # - roughly 70-100 cells
        # - appreciable but not complete consolidation
        # - several terminal units
        # - no dominant terminal unit occupying the whole field
        #
        # These are display-quality criteria, not biological
        # selection criteria.
        score = (
            1.00 * abs(n_cells - 80) / 80
            + 0.70 * abs(reduction - 0.55)
            + 1.40 * max(0.0, largest_fraction - 0.35)
            + 0.75 * max(0.0, 8 - n_terminal) / 8
            + 0.30 * max(
                0.0,
                0.85 - coordinate_overlap_fraction,
            )
        )

        candidates.append(
            {
                "score": float(score),
                "row_idx": int(row_idx),
                "super_i": i,
                "super_j": j,
                "side_fraction": float(side_fraction),
                "side": float(side),
                "xmin": float(xmin),
                "xmax": float(xmax),
                "ymin": float(ymin),
                "ymax": float(ymax),
                "n_cells": int(n_cells),
                "n_terminal": int(n_terminal),
                "reduction": float(reduction),
                "largest_terminal_fraction": float(
                    largest_fraction
                ),
                "coordinate_overlap_fraction": float(
                    coordinate_overlap_fraction
                ),
                "ids": ids,
            }
        )


if not candidates:
    raise RuntimeError(
        "No usable C-region candidate found."
    )

best = min(
    candidates,
    key=lambda x: x["score"],
)

region_ids = best.pop("ids")

Cxmin = best["xmin"]
Cxmax = best["xmax"]
Cymin = best["ymin"]
Cymax = best["ymax"]

interface_row = m0.loc[best["row_idx"]]

ii = int(interface_row["super_i"])
jj = int(interface_row["super_j"])

print()
print("SELECTED PANEL-C REGION")
print(json.dumps(best, indent=2))

print()
print(
    "Representative accepted interface:",
    ii,
    jj,
)


# ============================================================
# Save region audit
# ============================================================

region_metadata = {
    **best,
    "sample": SAMPLE,
    "selection_rule": (
        "best-scoring square centered on an accepted "
        "Level-0 interface; target approximately 80 cells, "
        "substantial nondegenerate terminal consolidation"
    ),
}

REGION_JSON.write_text(
    json.dumps(region_metadata, indent=2) + "\n"
)


# ============================================================
# Save interface values
# ============================================================

interface_metadata = {
    "sample": SAMPLE,
    "microstep": 0,
    "super_i": ii,
    "super_j": jj,
    "survivor_node": int(
        interface_row["survivor_node"]
    ),
    "removed_node": int(
        interface_row["removed_node"]
    ),
    "composite_merge_cost": float(
        interface_row["composite_merge_cost"]
    ),
    "merge_cost_threshold": float(
        interface_row["merge_cost_threshold"]
    ),
}

for col, label, color in CHANNELS:
    interface_metadata[col] = float(
        interface_row[col]
    )

INTERFACE_JSON.write_text(
    json.dumps(interface_metadata, indent=2) + "\n"
)


# ============================================================
# Select hierarchy checkpoints by normalized node-reduction
# progress, not raw microstep.
#
# u = (N0 - Nk)/(N0 - Nf)
# ============================================================

for q in checkpoints:
    q["u"] = (
        (N0 - q["nodes"])
        / (N0 - Nf)
    )

targets = [0.0, 1/3, 2/3, 1.0]

selected_states = []

used_steps = set()

for target in targets:

    options = [
        q for q in checkpoints
        if q["step"] not in used_steps
    ]

    q = min(
        options,
        key=lambda z: abs(z["u"] - target),
    )

    used_steps.add(q["step"])
    selected_states.append(q)


state_rows = []

for q in selected_states:

    local_labels = q["labels"][region_ids]

    state_rows.append({
        "target_u": float(
            targets[len(state_rows)]
        ),
        "checkpoint_step": int(q["step"]),
        "global_nodes": int(q["nodes"]),
        "global_u": float(q["u"]),
        "region_cells": int(len(region_ids)),
        "region_active_labels": int(
            len(np.unique(local_labels))
        ),
    })


pd.DataFrame(state_rows).to_csv(
    STATES_CSV,
    index=False,
)

print()
print("SELECTED STATES")
print(
    pd.DataFrame(state_rows).to_string(
        index=False
    )
)


# ============================================================
# Region polygons
# ============================================================

region_poly_ids = [
    i for i in region_ids
    if i in poly_by_index
]

region_polygons = [
    poly_by_index[i]
    for i in region_poly_ids
]


# ============================================================
# Plot utility
# ============================================================

def plot_partition(
    ax,
    labels,
    title=None,
    internal_edges=True,
):
    local_labels = labels[region_poly_ids]

    colors = categorical_colors(local_labels)

    collection = PolyCollection(
        region_polygons,
        facecolors=colors,
        edgecolors="white",
        linewidths=0.30 if internal_edges else 0.15,
        antialiased=True,
    )

    ax.add_collection(collection)

    ax.set_xlim(Cxmin, Cxmax)
    ax.set_ylim(Cymin, Cymax)

    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_edgecolor("0.60")

    if title:
        ax.set_title(
            title,
            fontsize=8.0,
            weight="bold",
            pad=3,
        )


# ============================================================
# C2 boundary zoom helper
# ============================================================

pi = poly_by_index[ii]
pj = poly_by_index[jj]

all_pair = np.vstack([pi, pj])

zx0 = all_pair[:, 0].min()
zx1 = all_pair[:, 0].max()
zy0 = all_pair[:, 1].min()
zy1 = all_pair[:, 1].max()

zdx = zx1 - zx0
zdy = zy1 - zy0

zpad = 0.25 * max(zdx, zdy)

zx0 -= zpad
zx1 += zpad
zy0 -= zpad
zy1 += zpad


# ============================================================
# Approximate the actual shared-interface locus
#
# We do not invent a synthetic line. We identify boundary
# vertices from each measured polygon that are nearest the
# opposing polygon and highlight the closest portion.
# ============================================================

D = np.linalg.norm(
    pi[:, None, :] - pj[None, :, :],
    axis=2,
)

min_d = np.nanmin(D)

# tolerance scaled to polygon size
scale = max(
    np.ptp(all_pair[:, 0]),
    np.ptp(all_pair[:, 1]),
)

tol = max(
    min_d * 2.5,
    scale * 0.04,
)

di = np.min(D, axis=1)
dj = np.min(D, axis=0)

shared_i = pi[di <= tol]
shared_j = pj[dj <= tol]


# ============================================================
# Build Figure C
# ============================================================

fig = plt.figure(
    figsize=(16.2, 4.45),
    facecolor="white",
)

outer = fig.add_gridspec(
    1,
    3,
    width_ratios=[0.87, 1.24, 2.62],
    left=0.025,
    right=0.985,
    bottom=0.08,
    top=0.90,
    wspace=0.12,
)

ax1 = fig.add_subplot(outer[0, 0])
ax2 = fig.add_subplot(outer[0, 1])

right = outer[0, 2].subgridspec(
    1,
    4,
    wspace=0.30,
)

axs3 = [
    fig.add_subplot(right[0, k])
    for k in range(4)
]


# ============================================================
# C1 — real Level-0 region
# ============================================================

level0 = np.arange(N0, dtype=np.int64)

plot_partition(
    ax1,
    level0,
    title="71 measured cells",
)

# Highlight chosen accepted cells.
for idx, color in [(ii, "black"), (jj, "black")]:
    p = poly_by_index[idx]

    closed = np.vstack([p, p[0]])

    ax1.plot(
        closed[:, 0],
        closed[:, 1],
        color=color,
        linewidth=1.4,
        zorder=10,
    )



# ============================================================
# C2 — five views of the same real interface
#
# Cells are deliberately neutral gray here so that color means
# channel identity, not cell identity.
#
# Each channel receives:
#   1. the same two measured cell polygons,
#   2. one colored interface centerline,
#   3. a thin colored centroid-to-centroid relation segment,
#   4. the actual raw channel value,
#   5. a one-color bar showing its empirical percentile among
#      all Level-0 candidate interfaces for that channel.
#
# Percentile scaling is used only for display. Raw values remain
# printed and are not made comparable across channels.
# ============================================================

from matplotlib.patches import Rectangle


# ------------------------------------------------------------
# Load Level-0 candidate population for within-channel scaling
# ------------------------------------------------------------

candidate0_file = (
    LEDGER
    / "candidate_boundaries"
    / "step_000000.parquet"
)

candidate_cols = [
    c[0] for c in CHANNELS
]

candidate0 = pq.read_table(
    candidate0_file,
    columns=candidate_cols,
).to_pandas()


def empirical_percentile(values, x):
    """
    Fraction of finite Level-0 candidate values <= x.
    Used only as a within-channel display coordinate.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    if len(v) == 0 or not np.isfinite(x):
        return np.nan

    return float(np.mean(v <= x))


channel_display = []

for col, label, color in CHANNELS:

    value = float(interface_row[col])

    pct = empirical_percentile(
        candidate0[col].to_numpy(),
        value,
    )

    channel_display.append(
        {
            "column": col,
            "label": label,
            "color": color,
            "value": value,
            "percentile": pct,
        }
    )


# ------------------------------------------------------------
# Construct ONE display centerline for the shared interface.
#
# The two measured cell polygons may contain two closely spaced
# contours. Those remain visible as black cell outlines.
#
# The colored SUTRA relation is shown only once: as the midpoint
# locus between densely sampled nearest portions of the two
# measured contours. This avoids the misleading appearance of
# two separate colored interaction boundaries.
# ------------------------------------------------------------

def resample_polygon(poly, n_per_edge=14):

    poly = np.asarray(poly, dtype=float)

    out = []

    n = len(poly)

    for k in range(n):

        a = poly[k]
        b = poly[(k + 1) % n]

        t = np.linspace(
            0.0,
            1.0,
            n_per_edge,
            endpoint=False,
        )[:, None]

        out.append(
            a[None, :] * (1.0 - t)
            + b[None, :] * t
        )

    return np.vstack(out)


def shared_interface_centerline(poly_a, poly_b):

    A = resample_polygon(poly_a)
    B = resample_polygon(poly_b)

    D = np.linalg.norm(
        A[:, None, :] - B[None, :, :],
        axis=2,
    )

    nearest_b = np.argmin(D, axis=1)
    nearest_d = D[
        np.arange(len(A)),
        nearest_b,
    ]

    d0 = float(np.nanmin(nearest_d))

    # Characteristic polygon-edge scale.
    pair = np.vstack([poly_a, poly_b])

    span = max(
        float(np.ptp(pair[:, 0])),
        float(np.ptp(pair[:, 1])),
    )

    # Select only the genuinely neighboring contour portion.
    tol = max(
        d0 + 0.055 * span,
        2.25 * d0 + 1e-12,
    )

    keep = nearest_d <= tol

    if np.count_nonzero(keep) < 3:

        order = np.argsort(nearest_d)

        keep_idx = order[
            :min(10, len(order))
        ]

        mids = 0.5 * (
            A[keep_idx]
            + B[nearest_b[keep_idx]]
        )

    else:

        mids = 0.5 * (
            A[keep]
            + B[nearest_b[keep]]
        )

    center = np.mean(
        mids,
        axis=0,
    )

    Z = mids - center

    # Principal direction of the interface.
    _, _, vh = np.linalg.svd(
        Z,
        full_matrices=False,
    )

    direction = vh[0]

    coord = Z @ direction

    # Trim extreme sampled points slightly so the display line
    # remains on the actual shared portion.
    qlo, qhi = np.quantile(
        coord,
        [0.05, 0.95],
    )

    p0 = center + qlo * direction
    p1 = center + qhi * direction

    return np.vstack([p0, p1])


interface_line = shared_interface_centerline(
    pi,
    pj,
)

centroid_i = polygon_centroid(pi)
centroid_j = polygon_centroid(pj)


# ------------------------------------------------------------
# C2 parent axis
# ------------------------------------------------------------

ax2.clear()
ax2.set_axis_off()

ax2.set_title("")


# ------------------------------------------------------------
# Five miniature pair panels + decision panel
# ------------------------------------------------------------

mini_positions = [
    # x, y, width, height
    (0.015, 0.565, 0.285, 0.315),
    (0.355, 0.565, 0.285, 0.315),
    (0.695, 0.565, 0.285, 0.315),

    (0.015, 0.155, 0.285, 0.315),
    (0.355, 0.155, 0.285, 0.315),
]

mini_axes = []


def draw_channel_pair(
    ax,
    item,
):

    color = item["color"]

    # Rounded white card behind each real cell pair.
    card = FancyBboxPatch(
        (-0.03, -0.18),
        1.06,
        1.31,
        boxstyle="round,pad=0.018,rounding_size=0.04",
        transform=ax.transAxes,
        facecolor=(0.985, 0.985, 0.985),
        edgecolor=(0.82, 0.82, 0.82),
        linewidth=0.7,
        zorder=0,
        clip_on=False,
    )
    ax.add_patch(card)
    value = item["value"]
    pct = item["percentile"]

    # Neutral cells: color now exclusively denotes evidence
    # channel identity.
    pc = PolyCollection(
        [pi, pj],
        facecolors=[
            (0.86, 0.86, 0.86),
            (0.74, 0.74, 0.74),
        ],
        edgecolors="0.10",
        linewidths=1.15,
        antialiased=True,
        zorder=2,
    )

    ax.add_collection(pc)

    # Thin pair-relation segment between cell centroids.
    ax.plot(
        [centroid_i[0], centroid_j[0]],
        [centroid_i[1], centroid_j[1]],
        color=color,
        linewidth=1.7,
        alpha=0.46,
        zorder=4,
    )

    ax.scatter(
        [centroid_i[0], centroid_j[0]],
        [centroid_i[1], centroid_j[1]],
        s=12,
        facecolor=color,
        edgecolor="white",
        linewidth=0.35,
        zorder=5,
    )

    # ONE colored interface line.
    ax.plot(
        interface_line[:, 0],
        interface_line[:, 1],
        color=color,
        linewidth=4.3,
        solid_capstyle="round",
        zorder=7,
    )

    ax.set_xlim(zx0, zx1)
    ax.set_ylim(zy0, zy1)
    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Channel title + raw value.
    ax.text(
        0.01,
        1.055,
        item["label"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7.0,
        weight="bold",
        color=color,
    )

    ax.text(
        0.99,
        1.055,
        f'{value:.3g}',
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.9,
        weight="bold",
        color="0.08",
    )

    bx = 0.06
    by = -0.115
    bw = 0.88
    bh = 0.050

    # Topology is intentionally not percentile-scaled.
    #
    # The Level-0 numerical range is extremely narrow, so a
    # percentile bar would exaggerate visually tiny variation.
    # SUTRA also does not use a standalone topology hard gate.
    if item["label"] == "Topology":

        ax.add_patch(
            FancyBboxPatch(
                (bx, by),
                bw,
                bh,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=ax.transAxes,
                facecolor=(0.97, 0.95, 1.00),
                edgecolor=color,
                linewidth=1.0,
                clip_on=False,
            )
        )

        ax.text(
            bx + bw / 2,
            by + bh / 2,
            "topology evidence",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.8,
            weight="bold",
            color=color,
        )

    else:

        # Continuous channels:
        # bar length = within-channel empirical percentile.
        ax.add_patch(
            Rectangle(
                (bx, by),
                bw,
                bh,
                transform=ax.transAxes,
                facecolor="white",
                edgecolor="0.65",
                linewidth=0.50,
                clip_on=False,
            )
        )

        if np.isfinite(pct):

            ax.add_patch(
                Rectangle(
                    (bx, by),
                    bw * pct,
                    bh,
                    transform=ax.transAxes,
                    facecolor=color,
                    edgecolor="none",
                    alpha=0.90,
                    clip_on=False,
                )
            )

            ptxt = f"P{100*pct:.0f}"

        else:
            ptxt = "NA"

        ax.text(
            bx + bw,
            by - 0.018,
            ptxt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=5.8,
            color="0.35",
        )


for item, pos in zip(
    channel_display,
    mini_positions,
):

    a = ax2.inset_axes(pos)

    draw_channel_pair(
        a,
        item,
    )

    mini_axes.append(a)


# ------------------------------------------------------------
# Compact decision box
# ------------------------------------------------------------

decision = ax2.inset_axes(
    [0.685, 0.105, 0.315, 0.405]
)

decision.set_axis_off()

decision.add_patch(
    FancyBboxPatch(
        (0.02, 0.02),
        0.96,
        0.96,
        boxstyle="round,pad=0.02,rounding_size=0.035",
        transform=decision.transAxes,
        facecolor=(0.985, 0.985, 0.985),
        edgecolor="0.45",
        linewidth=0.8,
    )
)

cost = float(
    interface_row["composite_merge_cost"]
)

tau = float(
    interface_row["merge_cost_threshold"]
)

decision.text(
    0.50,
    0.76,
    r"$\mathbf{e}_{ij}=(M_{ij},K_{ij},C_{ij},G_{ij},B_{ij})$",
    transform=decision.transAxes,
    ha="center",
    va="center",
    fontsize=7.0,
)

decision.text(
    0.50,
    0.56,
    r"$\downarrow$",
    transform=decision.transAxes,
    ha="center",
    va="center",
    fontsize=10,
)

decision.text(
    0.50,
    0.34,
    rf"$C_{{ij}}^{{\mathrm{{merge}}}}={cost:.3f}$",
    transform=decision.transAxes,
    ha="center",
    va="center",
    fontsize=7.4,
)

decision.text(
    0.50,
    0.22,
    rf"$<\;\tau={tau:.3f}$",
    transform=decision.transAxes,
    ha="center",
    va="center",
    fontsize=7.4,
)


decision.text(
    0.50,
    0.075,
    "accepted",
    transform=decision.transAxes,
    ha="center",
    va="center",
    fontsize=6.8,
    weight="bold",
    color="#14833b",
)


# Explicit note about display scaling.

# ============================================================
# C3 — exact hierarchy states, same x/y coordinates
# ============================================================

for k, (ax, q) in enumerate(
    zip(axs3, selected_states)
):

    plot_partition(
        ax,
        q["labels"],
    )

    if k == 0:
        label = "Level 0"
    elif k == len(axs3) - 1:
        label = "Terminal"
    else:
        label = f"u = {q['u']:.2f}"

    ax.set_title(
        label,
        fontsize=8.0,
        weight="bold",
        pad=3,
    )

    local_n = len(
        np.unique(q["labels"][region_ids])
    )

    ax.text(
        0.50,
        -0.055,
        f"{local_n} units",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        weight="bold",
    )


# ============================================================
# Shrink hierarchy maps slightly around their centers.
# This creates deliberate arrow corridors in white space.
# ============================================================

for ax in axs3:

    b = ax.get_position()

    scale = 0.80

    nw = b.width * scale
    nh = b.height * scale

    cx = 0.5 * (b.x0 + b.x1)
    cy = 0.5 * (b.y0 + b.y1)

    ax.set_position([
        cx - nw / 2,
        cy - nh / 2,
        nw,
        nh,
    ])


# ============================================================
# Arrows
# ============================================================

add_arrow(fig, ax1, ax2)
add_arrow(fig, ax2, axs3[0])

for a, b in zip(axs3[:-1], axs3[1:]):

    ba = a.get_position()
    bb = b.get_position()

    y = 0.5 * (ba.y0 + ba.y1)

    # Keep arrow fully inside whitespace.
    gap = bb.x0 - ba.x1
    mid = 0.5 * (ba.x1 + bb.x0)

    # Short compact arrow centered in the whitespace.
    half = min(0.012, 0.32 * gap)

    x0 = mid - half
    x1 = mid + half

    if x1 > x0:

        arrow = FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=16,
            linewidth=2.2,
            color="0.18",
            shrinkA=0,
            shrinkB=0,
            zorder=20,
            clip_on=False,
        )

        fig.add_artist(arrow)


# ============================================================
# Save
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.03,
)

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.03,
)

plt.close(fig)

print()
print("PANEL C COMPLETE")
print("PDF:", OUT_PDF)
print("PNG:", OUT_PNG)
print("Region:", REGION_JSON)
print("Interface:", INTERFACE_JSON)
print("States:", STATES_CSV)
