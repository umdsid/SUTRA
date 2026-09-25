#!/usr/bin/env python3

import os
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import FancyBboxPatch


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

CANDIDATE_FILE = (
    LEDGER
    / "candidate_boundaries"
    / "step_000000.parquet"
)

MERGE_FILE = (
    LEDGER
    / "merge_events"
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

OUT_PDF = PANELDIR / "panel_E.pdf"
OUT_PNG = PANELDIR / "panel_E.png"

PAIR_JSON = SOURCEDIR / "panel_E_matched_interfaces.json"
EVIDENCE_CSV = SOURCEDIR / "panel_E_matched_interface_evidence.csv"


# ============================================================
# Style
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "font.weight": "normal",
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


CHANNELS = [
    (
        "molecular_effective",
        "Molecular",
        "#d81b60",
    ),
    (
        "mechanics_effective",
        "Mechanics",
        "#ef8a17",
    ),
    (
        "communication_support_effective",
        "Communication",
        "#1a9850",
    ),
    (
        "geometry_effective",
        "Geometry",
        "#2878d0",
    ),
]


# ============================================================
# Read Level-0 cells / polygons
# ============================================================

cells = pq.read_table(
    CELL_FILE,
    columns=[
        "cell_id",
        "x_centroid",
        "y_centroid",
    ],
).to_pandas()

N0 = len(cells)

cell_ids = cells["cell_id"].tolist()

xy = cells[
    ["x_centroid", "y_centroid"]
].to_numpy(dtype=float)


bd = pq.read_table(
    BOUNDARY_FILE,
    columns=[
        "cell_id",
        "vertex_x",
        "vertex_y",
    ],
).to_pandas()

poly_by_id = {}

for cid, g in bd.groupby(
    "cell_id",
    sort=False,
):
    p = g[
        ["vertex_x", "vertex_y"]
    ].to_numpy(dtype=float)

    if len(p) >= 3:
        poly_by_id[cid] = p


poly_by_index = {
    i: poly_by_id[cell_ids[i]]
    for i in range(N0)
    if cell_ids[i] in poly_by_id
}


# ============================================================
# Read candidate interfaces
# ============================================================

candidate_cols = [
    "super_i",
    "super_j",
    "molecular_effective",
    "mechanics_effective",
    "communication_support_effective",
    "geometry_effective",
    "topology_effective",
    "composite_merge_cost",
    "merge_cost_threshold",
    "admissible",
    "merge_allowed",
]

cand = pq.read_table(
    CANDIDATE_FILE,
    columns=candidate_cols,
).to_pandas()


merge = pq.read_table(
    MERGE_FILE,
    columns=[
        "super_i",
        "super_j",
        "survivor_node",
        "removed_node",
    ],
).to_pandas()


def canonical_pair(i, j):
    i = int(i)
    j = int(j)
    return (
        min(i, j),
        max(i, j),
    )


actual_merge_pairs = {
    canonical_pair(i, j)
    for i, j in zip(
        merge["super_i"],
        merge["super_j"],
    )
}


cand["pair"] = [
    canonical_pair(i, j)
    for i, j in zip(
        cand["super_i"],
        cand["super_j"],
    )
]

cand["actually_merged"] = cand[
    "pair"
].isin(actual_merge_pairs)


# ============================================================
# Define retained candidates conservatively
#
# We specifically want an interface that remained physically
# present but whose composite cost exceeded the current
# threshold.
# ============================================================

# Compare the full Level-0 decision populations rather than
# only the small numerical batch executed at microstep 0.
accepted = cand[
    cand["merge_allowed"].astype(bool)
].copy()

retained = cand[
    ~cand["merge_allowed"].astype(bool)
].copy()


for col, _, _ in CHANNELS:
    accepted = accepted[
        np.isfinite(accepted[col])
    ]
    retained = retained[
        np.isfinite(retained[col])
    ]


print(
    "Level-0 admissible candidates:",
    len(accepted),
)

print(
    "Level-0 retained candidates:",
    len(retained),
)


if len(accepted) == 0:
    raise RuntimeError(
        "No accepted Level-0 merger found."
    )

if len(retained) == 0:
    raise RuntimeError(
        "No retained above-threshold Level-0 candidate found."
    )


# ============================================================
# Empirical percentile coordinates for display / matching
# ============================================================

percentiles = {}

for col, _, _ in CHANNELS:

    vals = cand[col].to_numpy(
        dtype=float
    )

    vals = vals[
        np.isfinite(vals)
    ]

    vals = np.sort(vals)

    percentiles[col] = vals


def empirical_percentile(
    col,
    value,
):

    vals = percentiles[col]

    if (
        len(vals) == 0
        or not np.isfinite(value)
    ):
        return np.nan

    return float(
        np.searchsorted(
            vals,
            value,
            side="right",
        )
        / len(vals)
    )


# ============================================================
# Match an actual accepted interface to a retained interface.
#
# Primary criterion:
#     almost identical molecular percentile.
#
# Secondary criterion:
#     clear difference in other effective evidence and in
#     composite cost margin.
#
# This prevents hand-picking a pretty example.
# ============================================================

rows = []

for ai, ar in accepted.iterrows():

    ma = float(
        ar["molecular_effective"]
    )

    pma = empirical_percentile(
        "molecular_effective",
        ma,
    )

    # First find retained candidates closest in molecular value.
    diff = np.abs(
        retained[
            "molecular_effective"
        ].to_numpy(dtype=float)
        - ma
    )

    # Only inspect closest 200 molecular matches.
    order = np.argsort(diff)[
        :min(200, len(diff))
    ]

    for pos in order:

        rr = retained.iloc[pos]

        mr = float(
            rr["molecular_effective"]
        )

        pmr = empirical_percentile(
            "molecular_effective",
            mr,
        )

        molecular_pct_delta = abs(
            pma - pmr
        )

        # Other-channel percentile separation.
        other_deltas = []

        for col, _, _ in CHANNELS[1:]:

            pa = empirical_percentile(
                col,
                float(ar[col]),
            )

            pr = empirical_percentile(
                col,
                float(rr[col]),
            )

            if np.isfinite(pa) and np.isfinite(pr):
                other_deltas.append(
                    abs(pa - pr)
                )

        mean_other_delta = (
            float(np.mean(other_deltas))
            if other_deltas
            else 0.0
        )

        accepted_margin = (
            float(
                ar["merge_cost_threshold"]
            )
            - float(
                ar["composite_merge_cost"]
            )
        )

        retained_margin = (
            float(
                rr["composite_merge_cost"]
            )
            - float(
                rr["merge_cost_threshold"]
            )
        )

        # Main objective is molecular matching.
        # Other divergence breaks near-ties.
        score = (
            8.0 * molecular_pct_delta
            - 1.5 * mean_other_delta
            - 0.15 * max(
                accepted_margin,
                0.0,
            )
            - 0.15 * max(
                retained_margin,
                0.0,
            )
        )

        rows.append(
            {
                "score": score,
                "molecular_pct_delta":
                    molecular_pct_delta,
                "mean_other_pct_delta":
                    mean_other_delta,
                "accepted_index": ai,
                "retained_index":
                    rr.name,
            }
        )


ranked = pd.DataFrame(rows)

# Prefer a very tight molecular match when available.
tight = ranked[
    ranked["molecular_pct_delta"]
    <= 0.005
]

if len(tight):
    chosen = tight.sort_values(
        "score"
    ).iloc[0]
else:
    chosen = ranked.sort_values(
        "score"
    ).iloc[0]


a = accepted.loc[
    chosen["accepted_index"]
]

r = retained.loc[
    chosen["retained_index"]
]


print()
print("SELECTED MATCH")
print(
    "molecular percentile difference:",
    chosen["molecular_pct_delta"],
)

print(
    "mean other-channel percentile difference:",
    chosen["mean_other_pct_delta"],
)

print(
    "admissible pair:",
    a["pair"],
)

print(
    "retained pair:",
    r["pair"],
)


# ============================================================
# Geometry helpers
# ============================================================

def polygon_centroid(poly):
    return np.mean(
        poly,
        axis=0,
    )


def resample_polygon(
    poly,
    n_per_edge=18,
):

    poly = np.asarray(
        poly,
        dtype=float,
    )

    out = []

    for k in range(len(poly)):

        p0 = poly[k]
        p1 = poly[
            (k + 1) % len(poly)
        ]

        t = np.linspace(
            0.0,
            1.0,
            n_per_edge,
            endpoint=False,
        )[:, None]

        out.append(
            p0[None, :] * (1 - t)
            + p1[None, :] * t
        )

    return np.vstack(out)


def shared_interface_centerline(
    poly_a,
    poly_b,
):

    A = resample_polygon(poly_a)
    B = resample_polygon(poly_b)

    D = np.linalg.norm(
        A[:, None, :]
        - B[None, :, :],
        axis=2,
    )

    nearest = np.argmin(
        D,
        axis=1,
    )

    d = D[
        np.arange(len(A)),
        nearest,
    ]

    d0 = float(
        np.min(d)
    )

    pair = np.vstack(
        [poly_a, poly_b]
    )

    span = max(
        np.ptp(pair[:, 0]),
        np.ptp(pair[:, 1]),
    )

    tol = max(
        d0 + 0.055 * span,
        2.25 * d0 + 1e-12,
    )

    keep = d <= tol

    if np.sum(keep) < 3:

        keep_idx = np.argsort(d)[
            :min(12, len(d))
        ]

        mids = 0.5 * (
            A[keep_idx]
            + B[nearest[keep_idx]]
        )

    else:

        mids = 0.5 * (
            A[keep]
            + B[nearest[keep]]
        )

    center = mids.mean(axis=0)

    Z = mids - center

    _, _, vh = np.linalg.svd(
        Z,
        full_matrices=False,
    )

    direction = vh[0]

    q = Z @ direction

    lo, hi = np.quantile(
        q,
        [0.04, 0.96],
    )

    return np.vstack([
        center + lo * direction,
        center + hi * direction,
    ])


def get_pair_geometry(row):

    i = int(row["super_i"])
    j = int(row["super_j"])

    pi = poly_by_index[i]
    pj = poly_by_index[j]

    c1 = polygon_centroid(pi)
    c2 = polygon_centroid(pj)

    line = shared_interface_centerline(
        pi,
        pj,
    )

    allp = np.vstack(
        [pi, pj]
    )

    span = max(
        np.ptp(allp[:, 0]),
        np.ptp(allp[:, 1]),
    )

    pad = 0.18 * span

    limits = (
        allp[:, 0].min() - pad,
        allp[:, 0].max() + pad,
        allp[:, 1].min() - pad,
        allp[:, 1].max() + pad,
    )

    return (
        i,
        j,
        pi,
        pj,
        c1,
        c2,
        line,
        limits,
    )


GA = get_pair_geometry(a)
GR = get_pair_geometry(r)


# ============================================================
# Save source data
# ============================================================

selection = {
    "sample": SAMPLE,
    "selection_rule": (
        "actual accepted step-0 merger matched to a retained "
        "above-threshold Level-0 candidate primarily by "
        "molecular_effective percentile; non-molecular evidence "
        "used only to break close matches"
    ),
    "molecular_percentile_difference":
        float(
            chosen[
                "molecular_pct_delta"
            ]
        ),
    "mean_other_channel_percentile_difference":
        float(
            chosen[
                "mean_other_pct_delta"
            ]
        ),
    "admissible": {
        "super_i":
            int(a["super_i"]),
        "super_j":
            int(a["super_j"]),
        "composite_merge_cost":
            float(
                a[
                    "composite_merge_cost"
                ]
            ),
        "threshold":
            float(
                a[
                    "merge_cost_threshold"
                ]
            ),
    },
    "retained": {
        "super_i":
            int(r["super_i"]),
        "super_j":
            int(r["super_j"]),
        "composite_merge_cost":
            float(
                r[
                    "composite_merge_cost"
                ]
            ),
        "threshold":
            float(
                r[
                    "merge_cost_threshold"
                ]
            ),
    },
}

PAIR_JSON.write_text(
    json.dumps(
        selection,
        indent=2,
    )
    + "\n"
)


evidence_rows = []

for col, label, color in CHANNELS:

    av = float(a[col])
    rv = float(r[col])

    evidence_rows.append({
        "channel": label,
        "column": col,
        "accepted_raw": av,
        "retained_raw": rv,
        "accepted_percentile":
            empirical_percentile(
                col,
                av,
            ),
        "retained_percentile":
            empirical_percentile(
                col,
                rv,
            ),
    })


pd.DataFrame(
    evidence_rows
).to_csv(
    EVIDENCE_CSV,
    index=False,
)


# ============================================================
# Plot helpers
# ============================================================

ACCEPT_COLOR = "#16853d"
RETAIN_COLOR = "#b33b3b"


def draw_real_pair(
    ax,
    geom,
    decision_color,
):

    (
        i,
        j,
        pi,
        pj,
        c1,
        c2,
        line,
        limits,
    ) = geom

    pc = PolyCollection(
        [pi, pj],
        facecolors=[
            (0.90, 0.90, 0.90),
            (0.76, 0.76, 0.76),
        ],
        edgecolors="0.10",
        linewidths=1.4,
        zorder=2,
    )

    ax.add_collection(pc)

    # Physical relation between centers.
    ax.plot(
        [c1[0], c2[0]],
        [c1[1], c2[1]],
        color=decision_color,
        linewidth=1.9,
        alpha=0.42,
        zorder=4,
    )

    ax.scatter(
        [c1[0], c2[0]],
        [c1[1], c2[1]],
        s=21,
        facecolor=decision_color,
        edgecolor="white",
        linewidth=0.6,
        zorder=5,
    )

    # Single shared-interface representation.
    ax.plot(
        line[:, 0],
        line[:, 1],
        color=decision_color,
        linewidth=5.0,
        solid_capstyle="round",
        zorder=6,
    )

    xmin, xmax, ymin, ymax = limits

    ax.set_xlim(
        xmin,
        xmax,
    )
    ax.set_ylim(
        ymin,
        ymax,
    )

    ax.set_aspect("equal")

    ax.set_xticks([])
    ax.set_yticks([])

    for spine in ax.spines.values():
        spine.set_visible(False)


# ============================================================
# Figure
# ============================================================

fig = plt.figure(
    figsize=(9.2, 3.45),
    facecolor="white",
)

gs = fig.add_gridspec(
    1,
    3,
    width_ratios=[0.68, 1.85, 0.68],
    left=0.035,
    right=0.975,
    bottom=0.20,
    top=0.95,
    wspace=0.24,
)

axA = fig.add_subplot(gs[0, 0])
axM = fig.add_subplot(gs[0, 1])
axR = fig.add_subplot(gs[0, 2])


# ------------------------------------------------------------
# Real matched interfaces
# ------------------------------------------------------------

draw_real_pair(
    axA,
    GA,
    ACCEPT_COLOR,
)

draw_real_pair(
    axR,
    GR,
    RETAIN_COLOR,
)

axA.set_title(
    "Admissible",
    fontsize=10.5,
    weight="bold",
    color=ACCEPT_COLOR,
    pad=5,
)

axR.set_title(
    "Retained",
    fontsize=10.5,
    weight="bold",
    color=RETAIN_COLOR,
    pad=5,
)

acost = float(
    a["composite_merge_cost"]
)

atau = float(
    a["merge_cost_threshold"]
)

rcost = float(
    r["composite_merge_cost"]
)

rtau = float(
    r["merge_cost_threshold"]
)

axA.text(
    0.5,
    -0.025,
    f"cost {acost:.3f} < threshold {atau:.3f}",
    transform=axA.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
    weight="bold",
    color=ACCEPT_COLOR,
)

axR.text(
    0.5,
    -0.025,
    f"cost {rcost:.3f} > threshold {rtau:.3f}",
    transform=axR.transAxes,
    ha="center",
    va="top",
    fontsize=7.5,
    weight="bold",
    color=RETAIN_COLOR,
)


# ------------------------------------------------------------
# Continuous evidence only
# ------------------------------------------------------------

axM.set_xlim(-3, 103)
axM.set_ylim(-0.55, 3.55)

axM.set_yticks(range(4))
axM.set_yticklabels(
    [
        "Molecular",
        "Mechanics",
        "Communication",
        "Geometry",
    ],
    fontsize=7.5,
    weight="bold",
)

axM.invert_yaxis()

axM.set_xticks(
    [0, 25, 50, 75, 100]
)

axM.tick_params(
    axis="x",
    labelsize=7.0,
)

axM.set_xlabel(
    "Within-channel Level-0 percentile",
    fontsize=8.0,
    weight="bold",
)

axM.grid(
    axis="x",
    linewidth=0.6,
    alpha=0.14,
)

for name in [
    "top",
    "right",
    "left",
]:
    axM.spines[name].set_visible(False)

axM.tick_params(
    axis="y",
    length=0,
    pad=3,
)

# Highlight the matching channel.
axM.axhspan(
    -0.36,
    0.36,
    facecolor="#f2f2f2",
    edgecolor="none",
    zorder=0,
)


dfE = pd.DataFrame(
    evidence_rows
)

# Remove topology entirely from E.
dfE = dfE[
    dfE["channel"] != "Topology"
].reset_index(drop=True)


for y, row in dfE.iterrows():

    channel_color = CHANNELS[y][2]

    pa = (
        100.0
        * float(
            row["accepted_percentile"]
        )
    )

    pr = (
        100.0
        * float(
            row["retained_percentile"]
        )
    )

    # Molecular points are intentionally almost coincident.
    # Preserve their exact x positions and separate them only
    # vertically so both observations remain visible.
    if row["channel"] == "Molecular":
        ya = y + 0.085
        yr = y - 0.085
    else:
        ya = y
        yr = y

    axM.plot(
        [pa, pr],
        [ya, yr],
        color=channel_color,
        linewidth=2.5,
        alpha=0.34,
        solid_capstyle="round",
        zorder=1,
    )

    axM.scatter(
        pa,
        ya,
        s=72,
        facecolor=ACCEPT_COLOR,
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )

    axM.scatter(
        pr,
        yr,
        s=72,
        facecolor="white",
        edgecolor=RETAIN_COLOR,
        linewidth=1.8,
        zorder=5,
    )

    axM.annotate(
        f"P{pa:.1f}",
        (pa, ya),
        xytext=(0, 7 if row["channel"] == "Molecular" else -11),
        textcoords="offset points",
        ha="center",
        va="bottom" if row["channel"] == "Molecular" else "top",
        fontsize=6.2,
        weight="bold",
        color=ACCEPT_COLOR,
    )

    axM.annotate(
        f"P{pr:.1f}",
        (pr, yr),
        xytext=(0, -7 if row["channel"] == "Molecular" else 9),
        textcoords="offset points",
        ha="center",
        va="top" if row["channel"] == "Molecular" else "bottom",
        fontsize=6.2,
        weight="bold",
        color=RETAIN_COLOR,
    )


mol_delta = (
    100.0
    * float(
        chosen[
            "molecular_pct_delta"
        ]
    )
)




# ------------------------------------------------------------
# Small legend
# ------------------------------------------------------------

axM.scatter(
    [],
    [],
    s=60,
    facecolor=ACCEPT_COLOR,
    edgecolor="white",
    label="admissible",
)

axM.scatter(
    [],
    [],
    s=60,
    facecolor="white",
    edgecolor=RETAIN_COLOR,
    linewidth=1.7,
    label="retained",
)

leg = axM.legend(
    loc="upper center",
    bbox_to_anchor=(
        0.5,
        -0.20,
    ),
    ncol=2,
    frameon=False,
    fontsize=6.8,
)

for t in leg.get_texts():
    t.set_weight("bold")




# ============================================================
# Save
# ============================================================

fig.savefig(
    OUT_PDF,
    bbox_inches="tight",
    pad_inches=0.035,
)

fig.savefig(
    OUT_PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.035,
)

plt.close(fig)

print()
print("PANEL D COMPLETE")
print("PDF:", OUT_PDF)
print("PNG:", OUT_PNG)
print("Selection:", PAIR_JSON)
print("Evidence:", EVIDENCE_CSV)
