#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection, LineCollection
from matplotlib.colors import hsv_to_rgb


# ============================================================
# Configuration
# ============================================================

ROOT = Path.home() / "Desktop" / "SUTRA"
SAMPLE = "healthy_reference"

N0 = 24406
NF = 10681
FINAL_REMOVED = N0 - NF

LEDGER = (
    ROOT
    / "results"
    / "hierarchy_v0911_specimen_local_contextual_flow"
    / "ledger"
    / SAMPLE
)

LABEL_DIR = LEDGER / "label_checkpoints"
MERGE_DIR = LEDGER / "merge_events"
CAND_DIR = LEDGER / "candidate_boundaries"

LEVEL0 = (
    ROOT
    / "results"
    / "hierarchy_level0_v070"
    / SAMPLE
)

CELLS_L0 = LEVEL0 / "cells.parquet"

CELL_FILE = (
    ROOT
    / "data"
    / SAMPLE
    / "cells.parquet"
)

BOUNDARY_FILE = (
    ROOT
    / "data"
    / SAMPLE
    / "cell_boundaries.parquet"
)

FIGROOT = ROOT / "figures" / "fig2_crossorgan" / "brain"
PANELDIR = FIGROOT / "panels"
SOURCEDIR = FIGROOT / "source_data"

PANELDIR.mkdir(parents=True, exist_ok=True)
SOURCEDIR.mkdir(parents=True, exist_ok=True)

OLD_FIG2 = ROOT / "figures" / "fig2"
EXPRESSION_PANEL = OLD_FIG2 / "panels" / "panel_E.png"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Utilities
# ============================================================

def retry_read_parquet(path: Path, columns=None, attempts: int = 8):

    last = None

    for attempt in range(1, attempts + 1):

        try:

            return pq.read_table(
                path,
                columns=columns,
            )

        except (TimeoutError, OSError) as exc:

            last = exc

            if attempt == attempts:
                break

            wait_s = min(
                3 * attempt,
                20,
            )

            print(
                f"[I/O retry {attempt}/{attempts}] "
                f"{path.name}; sleep {wait_s}s"
            )

            time.sleep(wait_s)

    raise RuntimeError(
        f"Could not read {path}"
    ) from last


def step_from_name(path: Path) -> int:

    m = re.search(
        r"(\d+)",
        path.stem,
    )

    if not m:
        raise ValueError(path)

    return int(m.group(1))


def hierarchy_u(nodes: float) -> float:

    return float(
        np.clip(
            (N0 - nodes)
            / max(FINAL_REMOVED, 1),
            0,
            1,
        )
    )


def deterministic_color(label: int):

    phi = 0.6180339887498949

    h = (int(label) * phi) % 1.0

    return hsv_to_rgb([
        h,
        0.72,
        0.92,
    ])


def save_panel(fig, letter: str):

    png = PANELDIR / f"panel_{letter}.png"
    pdf = PANELDIR / f"panel_{letter}.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        pad_inches=0.025,
        facecolor="white",
    )

    plt.close(fig)

    print(f"panel {letter}: {png}")


# ============================================================
# Exact production trajectory
# ============================================================

def merge_trajectory():

    files = sorted(
        MERGE_DIR.glob("step_*.parquet"),
        key=step_from_name,
    )

    if not files:
        raise RuntimeError("No merge-event files")

    rows = []

    nodes = N0
    cumulative = 0

    rows.append({
        "step": 0,
        "nodes": nodes,
        "merges_this_step": 0,
        "cumulative_merges": 0,
        "u": 0.0,
        "mean_mass": 1.0,
    })

    for p in files:

        step = step_from_name(p)

        nm = int(
            pq.ParquetFile(p)
            .metadata
            .num_rows
        )

        cumulative += nm
        nodes -= nm

        rows.append({
            "step": step + 1,
            "nodes": nodes,
            "merges_this_step": nm,
            "cumulative_merges": cumulative,
            "u": hierarchy_u(nodes),
            "mean_mass": N0 / nodes,
        })

    df = pd.DataFrame(rows)

    if int(df.iloc[-1].nodes) != NF:
        raise RuntimeError(
            f"Final nodes {int(df.iloc[-1].nodes)} != {NF}"
        )

    df.to_csv(
        SOURCEDIR / "hierarchy_trajectory.csv",
        index=False,
    )

    return df


def nodes_at_step(traj, step):

    x = traj[
        traj.step <= step
    ]

    if len(x) == 0:
        return N0

    return int(
        x.iloc[-1].nodes
    )


# ============================================================
# Label checkpoints
# ============================================================

def load_label_checkpoints():

    files = sorted(
        LABEL_DIR.glob("labels_*.npz"),
        key=step_from_name,
    )

    arrays = {}
    rows = []

    for p in files:

        step = step_from_name(p)

        with np.load(
            p,
            allow_pickle=False,
        ) as z:

            labels = np.asarray(
                z["labels"],
                dtype=np.int64,
            )

        if len(labels) != N0:
            raise RuntimeError(
                f"{p}: labels length {len(labels)}"
            )

        nodes = int(
            np.unique(labels).size
        )

        arrays[step] = labels

        rows.append({
            "step": step,
            "nodes": nodes,
            "u": hierarchy_u(nodes),
            "mean_mass": N0 / nodes,
        })

    meta = (
        pd.DataFrame(rows)
        .sort_values("u")
        .reset_index(drop=True)
    )

    meta.to_csv(
        SOURCEDIR / "label_checkpoint_summary.csv",
        index=False,
    )

    return meta, arrays


def choose_checkpoint_states(
    meta,
    targets=(0.0, 0.50, 0.85, 1.0),
):

    rows = []

    used = set()

    for target in targets:

        order = np.argsort(
            np.abs(
                meta["u"].to_numpy()
                - target
            )
        )

        for idx in order:

            step = int(
                meta.iloc[idx].step
            )

            if step not in used:
                used.add(step)
                rows.append(
                    meta.iloc[idx]
                )
                break

    return (
        pd.DataFrame(rows)
        .reset_index(drop=True)
    )


# ============================================================
# Polygon data
# ============================================================

def resolve_xy_columns(df):

    candidates = [
        ("x", "y"),
        ("x_centroid", "y_centroid"),
        ("centroid_x", "centroid_y"),
    ]

    for xcol, ycol in candidates:
        if (
            xcol in df.columns
            and ycol in df.columns
        ):
            return xcol, ycol

    raise RuntimeError(
        f"Could not resolve cell coordinates: {list(df.columns)}"
    )


def load_cells_and_polygons():

    # --------------------------------------------------------
    # Hierarchy-aligned Level-0 table.
    #
    # This is the authoritative order used by label arrays and
    # expression matrix columns.
    # --------------------------------------------------------

    cells = pd.read_parquet(
        CELLS_L0
    )

    cells = (
        cells
        .sort_values("cell_index")
        .reset_index(drop=True)
    )

    assert len(cells) == N0

    assert np.array_equal(
        cells["cell_index"].to_numpy(),
        np.arange(N0),
    )

    assert np.array_equal(
        cells["matrix_column"].to_numpy(),
        np.arange(N0),
    )

    level0_ids = (
        cells["cell_id"]
        .astype(str)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Raw Xenium cell table.
    #
    # This is the table that is natively keyed to
    # cell_boundaries.parquet and was used successfully by the
    # original Fig. 2 polygon renderer.
    # --------------------------------------------------------

    raw_schema = (
        pq.ParquetFile(CELL_FILE)
        .schema_arrow
        .names
    )

    raw_cols = ["cell_id"]

    if (
        "x_centroid" in raw_schema
        and "y_centroid" in raw_schema
    ):
        raw_cols += [
            "x_centroid",
            "y_centroid",
        ]

    elif (
        "x" in raw_schema
        and "y" in raw_schema
    ):
        raw_cols += [
            "x",
            "y",
        ]

    raw_cells = retry_read_parquet(
        CELL_FILE,
        raw_cols,
    ).to_pandas()

    raw_cells["__cell_id_str"] = (
        raw_cells["cell_id"]
        .astype(str)
    )

    # --------------------------------------------------------
    # Raw cell boundaries.
    # --------------------------------------------------------

    boundaries = retry_read_parquet(
        BOUNDARY_FILE,
        [
            "cell_id",
            "vertex_x",
            "vertex_y",
        ],
    ).to_pandas()

    boundaries["__cell_id_str"] = (
        boundaries["cell_id"]
        .astype(str)
    )

    by_id = {}

    for cid, g in boundaries.groupby(
        "__cell_id_str",
        sort=False,
    ):

        poly = g[
            [
                "vertex_x",
                "vertex_y",
            ]
        ].to_numpy(
            dtype=float
        )

        if len(poly) >= 3:
            by_id[str(cid)] = poly

    print()
    print("POLYGON MAPPING")
    print(
        "raw cells:",
        len(raw_cells),
    )
    print(
        "raw boundary IDs:",
        len(by_id),
    )
    print(
        "Level-0 cells:",
        len(cells),
    )

    # --------------------------------------------------------
    # First try the scientifically preferred mapping:
    # exact cell ID/barcode correspondence.
    # --------------------------------------------------------

    polygons = [
        by_id.get(cid)
        for cid in level0_ids
    ]

    matched = sum(
        p is not None
        for p in polygons
    )

    print(
        "exact Level-0 ID -> boundary matches:",
        matched,
        "/",
        N0,
    )

    # --------------------------------------------------------
    # If direct IDs differ because raw Xenium cells use another
    # native identifier, use the raw-cell row correspondence
    # only after verifying that raw cell order itself matches
    # the Level-0 barcode order.
    # --------------------------------------------------------

    if matched != N0:

        raw_ids = (
            raw_cells["__cell_id_str"]
            .to_numpy()
        )

        if (
            len(raw_ids) == N0
            and np.array_equal(
                raw_ids,
                level0_ids,
            )
        ):

            print(
                "raw cell order == Level-0 order: EXACT"
            )

            polygons = [
                by_id.get(cid)
                for cid in raw_ids
            ]

            matched = sum(
                p is not None
                for p in polygons
            )

        else:

            # ------------------------------------------------
            # One more common Xenium case:
            # boundary IDs correspond to raw-cell IDs, while
            # raw cells carry a separate barcode-like column.
            # Look for a column that exactly equals the Level-0
            # barcode vector. No fuzzy matching is allowed.
            # ------------------------------------------------

            raw_full = pd.read_parquet(
                CELL_FILE
            )

            bridge_col = None

            for col in raw_full.columns:

                try:
                    v = (
                        raw_full[col]
                        .astype(str)
                        .to_numpy()
                    )
                except Exception:
                    continue

                if (
                    len(v) == N0
                    and np.array_equal(
                        v,
                        level0_ids,
                    )
                ):
                    bridge_col = col
                    break

            if bridge_col is None:

                print()
                print(
                    "Raw cell columns:",
                    list(raw_full.columns),
                )

                print(
                    "Raw cell_id head:",
                    raw_ids[:10].tolist(),
                )

                print(
                    "Level-0 cell_id head:",
                    level0_ids[:10].tolist(),
                )

                print(
                    "Boundary ID head:",
                    list(by_id.keys())[:10],
                )

                raise RuntimeError(
                    "Could not establish an exact mapping "
                    "between Level-0 cells and raw boundary IDs. "
                    "No positional or fuzzy fallback was used."
                )

            print(
                "Exact Level-0 barcode bridge column:",
                bridge_col,
            )

            # raw_full row i now corresponds exactly to
            # Level-0 row i; raw cell_id supplies boundary key.
            raw_boundary_ids = (
                raw_full["cell_id"]
                .astype(str)
                .to_numpy()
            )

            polygons = [
                by_id.get(cid)
                for cid in raw_boundary_ids
            ]

            matched = sum(
                p is not None
                for p in polygons
            )

    valid = np.asarray(
        [
            i
            for i, poly in enumerate(polygons)
            if poly is not None
        ],
        dtype=int,
    )

    print(
        "final polygon matches:",
        len(valid),
        "/",
        N0,
    )

    # A few genuinely missing/degenerate polygons would be
    # tolerable, but a large mapping loss is not.
    if len(valid) < 0.98 * N0:

        raise RuntimeError(
            f"Only {len(valid)}/{N0} Level-0 cells "
            "received polygons; refusing to render."
        )

    return (
        cells,
        polygons,
        valid,
    )


# ============================================================
# Candidate files
# ============================================================

def candidate_files():

    files = sorted(
        CAND_DIR.glob("step_*.parquet"),
        key=step_from_name,
    )

    if not files:
        raise RuntimeError(
            "No candidate-boundary files"
        )

    return files


def select_candidate_files(files, n=60):

    if len(files) <= n:
        return files

    idx = np.unique(
        np.round(
            np.linspace(
                0,
                len(files) - 1,
                n,
            )
        ).astype(int)
    )

    return [files[i] for i in idx]


def candidate_state_meta(traj):

    rows = []

    for p in candidate_files():

        step = step_from_name(p)
        nodes = nodes_at_step(traj, step)

        rows.append({
            "path": str(p),
            "step": step,
            "nodes": nodes,
            "u": hierarchy_u(nodes),
        })

    return pd.DataFrame(rows)


def nearest_candidate_file(
    cand_meta,
    target_u,
):

    i = int(
        np.argmin(
            np.abs(
                cand_meta["u"].to_numpy()
                - target_u
            )
        )
    )

    return cand_meta.iloc[i]


# ============================================================
# Candidate endpoint resolver
# ============================================================

PAIR_ALIASES = [
    ("super_i", "super_j"),
    ("node_u", "node_v"),
    ("node_i", "node_j"),
    ("source_node", "target_node"),
    ("source", "target"),
    ("src", "dst"),
    ("left_node", "right_node"),
    ("component_u", "component_v"),
    ("component_a", "component_b"),
    ("label_u", "label_v"),
    ("label_a", "label_b"),
    ("supernode_u", "supernode_v"),
    ("supernode_a", "supernode_b"),
    ("u_node", "v_node"),
    ("a", "b"),
]


def resolve_endpoint_columns(path):

    names = (
        pq.ParquetFile(path)
        .schema_arrow
        .names
    )

    lower = {
        n.lower(): n
        for n in names
    }

    for a, b in PAIR_ALIASES:

        if (
            a.lower() in lower
            and b.lower() in lower
        ):

            return (
                lower[a.lower()],
                lower[b.lower()],
            )

    # Last-resort semantic search.
    integer_like = []

    schema = (
        pq.ParquetFile(path)
        .schema_arrow
    )

    for field in schema:

        s = str(field.type).lower()

        if (
            "int" in s
            and any(
                token in field.name.lower()
                for token in [
                    "node",
                    "component",
                    "label",
                    "supernode",
                ]
            )
        ):
            integer_like.append(
                field.name
            )

    if len(integer_like) == 2:
        return (
            integer_like[0],
            integer_like[1],
        )

    raise RuntimeError(
        "Could not resolve candidate endpoint columns.\n"
        f"Candidate schema columns:\n{names}"
    )


# ============================================================
# Unit centroids from Level-0 labels
# ============================================================

def unit_centroids(labels, x, y):

    lab, inv = np.unique(
        labels,
        return_inverse=True,
    )

    n = len(lab)

    count = np.bincount(
        inv,
        minlength=n,
    ).astype(float)

    sx = np.bincount(
        inv,
        weights=x,
        minlength=n,
    )

    sy = np.bincount(
        inv,
        weights=y,
        minlength=n,
    )

    cx = sx / count
    cy = sy / count

    return {
        int(k): (
            float(a),
            float(b),
        )
        for k, a, b
        in zip(lab, cx, cy)
    }


# ============================================================
# Algorithmic crop selection
# ============================================================

def choose_crop(
    cells,
    initial_labels=None,
    terminal_labels=None,
):
    """
    Select a square field of view from Level-0 geometry only.

    No hierarchy outcome, terminal labels, admissibility, merger
    count, or later-state information enters the selection.

    Objective:
      1. high spatial fill across the square;
      2. readable Level-0 cell count;
      3. avoid marginal/peripheral slivers.

    This makes the displayed field a geometry-selected field of
    view rather than an outcome-selected example.
    """

    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)

    xmin0, xmax0 = x.min(), x.max()
    ymin0, ymax0 = y.min(), y.max()

    xr = xmax0 - xmin0
    yr = ymax0 - ymin0

    short_extent = min(xr, yr)

    # Scale desired visual complexity with specimen size.
    # Healthy brain -> ~470 target cells.
    # Kidney -> capped near 900.
    target_cells = int(
        np.clip(
            3.0 * np.sqrt(len(cells)),
            400,
            900,
        )
    )

    # Search a range of genuinely square windows.
    side_fracs = np.linspace(
        0.11,
        0.24,
        8,
    )

    best = None

    for frac in side_fracs:

        side = float(
            frac * short_extent
        )

        # Candidate centers stay inside specimen coordinate bounds.
        centers_x = np.linspace(
            xmin0 + side / 2,
            xmax0 - side / 2,
            17,
        )

        centers_y = np.linspace(
            ymin0 + side / 2,
            ymax0 - side / 2,
            17,
        )

        for cx in centers_x:
            for cy in centers_y:

                xmin = cx - side / 2
                xmax = cx + side / 2
                ymin = cy - side / 2
                ymax = cy + side / 2

                mask = (
                    (x >= xmin)
                    & (x <= xmax)
                    & (y >= ymin)
                    & (y <= ymax)
                )

                ids = np.flatnonzero(mask)
                n_cells = len(ids)

                if n_cells < 180:
                    continue

                # --------------------------------------------
                # Spatial occupancy:
                # divide the square into 8x8 bins and ask what
                # fraction actually contains tissue.
                # --------------------------------------------

                xx = x[ids]
                yy = y[ids]

                H, _, _ = np.histogram2d(
                    xx,
                    yy,
                    bins=8,
                    range=[
                        [xmin, xmax],
                        [ymin, ymax],
                    ],
                )

                occupied_fraction = float(
                    np.mean(H > 0)
                )

                # Also punish a crop whose population is
                # concentrated into only part of the square.
                row_fill = float(
                    np.mean(
                        np.sum(H, axis=1) > 0
                    )
                )

                col_fill = float(
                    np.mean(
                        np.sum(H, axis=0) > 0
                    )
                )

                edge_fill = min(
                    row_fill,
                    col_fill,
                )

                # Cell-count readability.
                count_score = math.exp(
                    -(
                        (n_cells - target_cells)
                        / max(
                            0.55 * target_cells,
                            1,
                        )
                    ) ** 2
                )

                # Geometry-only score.
                score = (
                    0.62 * occupied_fraction
                    + 0.23 * edge_fill
                    + 0.15 * count_score
                )

                rec = {
                    "cx": float(cx),
                    "cy": float(cy),
                    "xmin": float(xmin),
                    "xmax": float(xmax),
                    "ymin": float(ymin),
                    "ymax": float(ymax),
                    "side_length": float(side),
                    "n_level0_cells": int(n_cells),
                    "target_cells": int(target_cells),
                    "grid_occupancy_fraction":
                        occupied_fraction,
                    "row_fill_fraction":
                        row_fill,
                    "column_fill_fraction":
                        col_fill,
                    "selection_score":
                        float(score),
                    "selection_basis":
                        (
                            "Level-0 geometry only: square "
                            "occupancy, axis fill, and readable "
                            "cell count"
                        ),
                }

                if (
                    best is None
                    or rec["selection_score"]
                    > best["selection_score"]
                ):
                    best = rec

    if best is None:
        raise RuntimeError(
            "Could not identify a filled square field of view."
        )

    # Descriptive outcome statistics are computed only AFTER
    # selection and do not contribute to the selection score.
    if (
        initial_labels is not None
        and terminal_labels is not None
    ):

        mask = (
            (x >= best["xmin"])
            & (x <= best["xmax"])
            & (y >= best["ymin"])
            & (y <= best["ymax"])
        )

        ids = np.flatnonzero(mask)

        init_n = int(
            np.unique(
                initial_labels[ids]
            ).size
        )

        term_n = int(
            np.unique(
                terminal_labels[ids]
            ).size
        )

        best[
            "initial_units_local_posthoc"
        ] = init_n

        best[
            "terminal_units_local_posthoc"
        ] = term_n

        best[
            "local_reduction_posthoc"
        ] = float(
            1
            - term_n
            / max(init_n, 1)
        )

    with open(
        SOURCEDIR / "local_crop_manifest.json",
        "w",
    ) as f:

        json.dump(
            best,
            f,
            indent=2,
        )

    print()
    print("SELECTED SQUARE LOCAL CROP")
    print(json.dumps(best, indent=2))

    return best


def crop_mask(cells, crop):

    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)

    return (
        (x >= crop["xmin"])
        & (x <= crop["xmax"])
        & (y >= crop["ymin"])
        & (y <= crop["ymax"])
    )


# ============================================================
# A — whole-tissue overview
# ============================================================

def panel_A(
    cells,
    polygons,
    valid,
    meta,
    arrays,
    crop,
):

    picked = choose_checkpoint_states(
        meta,
        targets=(
            0.0,
            0.34,
            0.67,
            1.0,
        ),
    )

    picked.to_csv(
        SOURCEDIR / "panel_A_states.csv",
        index=False,
    )

    all_xy = np.vstack([
        polygons[i]
        for i in valid
    ])

    xmin, xmax = (
        all_xy[:, 0].min(),
        all_xy[:, 0].max(),
    )

    ymin, ymax = (
        all_xy[:, 1].min(),
        all_xy[:, 1].max(),
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(13.2, 3.0),
    )

    for ax, row in zip(
        axes,
        picked.itertuples(),
    ):

        labels = arrays[int(row.step)]

        colors = np.asarray([
            deterministic_color(
                labels[i]
            )
            for i in valid
        ])

        pc = PolyCollection(
            [polygons[i] for i in valid],
            facecolors=colors,
            edgecolors="white",
            linewidths=0.10,
        )

        ax.add_collection(pc)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # Exact geometry-selected field used in B/C.
        # Show it only on the first whole-tissue map.
        if ax is axes[0]:

            rect = plt.Rectangle(
                (
                    crop["xmin"],
                    crop["ymin"],
                ),
                crop["xmax"] - crop["xmin"],
                crop["ymax"] - crop["ymin"],
                fill=False,
                edgecolor="black",
                linewidth=2.0,
                zorder=20,
            )

            ax.add_patch(rect)

        if row.u < 0.01:
            title = "Level 0"
        elif row.u > 0.99:
            title = "Terminal"
        else:
            title = f"u = {row.u:.2f}"

        ax.text(
            0.5,
            1.01,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

        ax.text(
            0.5,
            -0.01,
            f"{int(row.nodes):,} units",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
            weight="bold",
        )

    fig.subplots_adjust(
        left=.005,
        right=.995,
        top=.91,
        bottom=.08,
        wspace=.018,
    )

    save_panel(fig, "A")


# ============================================================
# B — real local hierarchy
# ============================================================

def panel_B(
    cells,
    polygons,
    meta,
    arrays,
    crop,
):

    picked = choose_checkpoint_states(
        meta,
        targets=(
            0.0,
            0.50,
            0.85,
            1.0,
        ),
    )

    picked.to_csv(
        SOURCEDIR / "panel_B_states.csv",
        index=False,
    )

    mask = crop_mask(
        cells,
        crop,
    )

    idx = np.flatnonzero(mask)

    idx = np.asarray([
        i
        for i in idx
        if polygons[i] is not None
    ], dtype=int)

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12.8, 3.0),
    )

    for ax, row in zip(
        axes,
        picked.itertuples(),
    ):

        labels = arrays[int(row.step)]

        colors = np.asarray([
            deterministic_color(
                labels[i]
            )
            for i in idx
        ])

        pc = PolyCollection(
            [polygons[i] for i in idx],
            facecolors=colors,
            edgecolors="white",
            linewidths=0.28,
        )

        ax.add_collection(pc)

        ax.set_xlim(
            crop["xmin"],
            crop["xmax"],
        )

        ax.set_ylim(
            crop["ymin"],
            crop["ymax"],
        )

        ax.set_aspect("equal")
        ax.set_axis_off()

        local_units = np.unique(
            labels[idx]
        ).size

        if row.u < .01:
            title = "Level 0"
        elif row.u > .99:
            title = "Terminal"
        else:
            title = f"u = {row.u:.2f}"

        ax.text(
            .5,
            1.01,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

        ax.text(
            .5,
            -.01,
            f"{local_units:,} local units",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=7,
        )

    fig.subplots_adjust(
        left=.005,
        right=.995,
        top=.90,
        bottom=.10,
        wspace=.025,
    )

    save_panel(fig, "B")


# ============================================================
# C — candidate versus admissible interfaces in same region
# ============================================================

def panel_C(
    traj,
    cells,
    meta,
    arrays,
    crop,
):

    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)

    picked = choose_checkpoint_states(
        meta,
        targets=(
            0.0,
            0.50,
            0.85,
            1.0,
        ),
    )

    cm = candidate_state_meta(
        traj
    )

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(12.8, 3.0),
    )

    source_rows = []

    for ax, state in zip(
        axes,
        picked.itertuples(),
    ):

        cand = nearest_candidate_file(
            cm,
            float(state.u),
        )

        path = Path(
            cand.path
        )

        a_col, b_col = (
            resolve_endpoint_columns(path)
        )

        schema_names = (
            pq.ParquetFile(path)
            .schema_arrow
            .names
        )

        adm_col = (
            "admissible"
            if "admissible" in schema_names
            else None
        )

        if adm_col is None:
            raise RuntimeError(
                f"{path}: no admissible column"
            )

        tab = retry_read_parquet(
            path,
            [
                a_col,
                b_col,
                adm_col,
            ],
        ).to_pandas()

        labels = arrays[
            int(state.step)
        ]

        cent = unit_centroids(
            labels,
            x,
            y,
        )

        neutral_segments = []
        admissible_segments = []

        for r in tab.itertuples(
            index=False
        ):

            a = int(
                getattr(r, a_col)
            )

            b = int(
                getattr(r, b_col)
            )

            if (
                a not in cent
                or b not in cent
            ):
                continue

            xa, ya = cent[a]
            xb, yb = cent[b]

            mx = .5 * (xa + xb)
            my = .5 * (ya + yb)

            if not (
                crop["xmin"]
                <= mx
                <= crop["xmax"]
                and crop["ymin"]
                <= my
                <= crop["ymax"]
            ):
                continue

            seg = [
                (xa, ya),
                (xb, yb),
            ]

            is_adm = bool(
                getattr(
                    r,
                    adm_col,
                )
            )

            neutral_segments.append(
                seg
            )

            if is_adm:
                admissible_segments.append(
                    seg
                )

        # Physical candidates.
        lc0 = LineCollection(
            neutral_segments,
            colors="0.72",
            linewidths=0.55,
            alpha=.65,
            zorder=1,
        )

        ax.add_collection(lc0)

        # Contextually admissible subset.
        lc1 = LineCollection(
            admissible_segments,
            colors="#d95f02",
            linewidths=1.35,
            alpha=.95,
            zorder=3,
        )

        ax.add_collection(lc1)

        # Current unit centroids.
        local_cent = [
            (a, b)
            for a, b in cent.values()
            if (
                crop["xmin"] <= a <= crop["xmax"]
                and crop["ymin"] <= b <= crop["ymax"]
            )
        ]

        if local_cent:

            pts = np.asarray(
                local_cent
            )

            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=3,
                color="0.20",
                linewidths=0,
                zorder=4,
            )

        ax.set_xlim(
            crop["xmin"],
            crop["xmax"],
        )

        ax.set_ylim(
            crop["ymin"],
            crop["ymax"],
        )

        ax.set_aspect("equal")
        ax.set_axis_off()

        if state.u < .01:
            title = "Level 0"
        elif state.u > .99:
            title = "Terminal"
        else:
            title = f"u = {state.u:.2f}"

        ax.text(
            .5,
            1.01,
            title,
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
            weight="bold",
        )

        ax.text(
            .5,
            -.01,
            (
                f"{len(neutral_segments):,} candidate  |  "
                f"{len(admissible_segments):,} admissible"
            ),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6.5,
        )

        source_rows.append({
            "label_checkpoint_step":
                int(state.step),
            "label_checkpoint_u":
                float(state.u),
            "candidate_step":
                int(cand.step),
            "candidate_u":
                float(cand.u),
            "endpoint_a_column":
                a_col,
            "endpoint_b_column":
                b_col,
            "local_candidate_interfaces":
                len(neutral_segments),
            "local_admissible_interfaces":
                len(admissible_segments),
        })

    pd.DataFrame(
        source_rows
    ).to_csv(
        SOURCEDIR / "panel_C_interface_states.csv",
        index=False,
    )

    # Minimal legend on first panel only.
    axes[0].plot(
        [],
        [],
        color="0.72",
        lw=1.2,
        label="Candidate",
    )

    axes[0].plot(
        [],
        [],
        color="#d95f02",
        lw=1.8,
        label="Admissible",
    )

    axes[0].legend(
        frameon=False,
        fontsize=6.2,
        loc="upper left",
    )

    fig.subplots_adjust(
        left=.005,
        right=.995,
        top=.90,
        bottom=.10,
        wspace=.025,
    )

    save_panel(fig, "C")


# ============================================================
# D — candidate cost landscape versus threshold
# ============================================================

def panel_D(traj):

    files = select_candidate_files(
        candidate_files(),
        n=50,
    )

    rows = []

    for k, p in enumerate(files):

        step = step_from_name(p)

        tab = retry_read_parquet(
            p,
            [
                "composite_merge_cost",
                "merge_cost_threshold",
            ],
        )

        cost = (
            tab["composite_merge_cost"]
            .to_numpy(
                zero_copy_only=False
            )
            .astype(float)
        )

        tau = (
            tab["merge_cost_threshold"]
            .to_numpy(
                zero_copy_only=False
            )
            .astype(float)
        )

        cost = cost[
            np.isfinite(cost)
        ]

        tau = tau[
            np.isfinite(tau)
        ]

        if not len(cost):
            continue

        nodes = nodes_at_step(
            traj,
            step,
        )

        rows.append({
            "step": step,
            "u": hierarchy_u(nodes),
            "nodes": nodes,
            "q10": np.quantile(cost, .10),
            "q25": np.quantile(cost, .25),
            "median": np.quantile(cost, .50),
            "q75": np.quantile(cost, .75),
            "q90": np.quantile(cost, .90),
            "threshold":
                np.median(tau)
                if len(tau)
                else np.nan,
            "candidate_count":
                len(cost),
        })

        print(
            f"D {k+1:2d}/{len(files)} "
            f"step={step}"
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("u")
    )

    df.to_csv(
        SOURCEDIR / "panel_D_cost_landscape.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(5.2, 3.6)
    )

    ax.fill_between(
        df.u,
        df.q10,
        df.q90,
        color="0.75",
        alpha=.28,
        linewidth=0,
        label="10–90%",
    )

    ax.fill_between(
        df.u,
        df.q25,
        df.q75,
        color="0.45",
        alpha=.28,
        linewidth=0,
        label="IQR",
    )

    ax.plot(
        df.u,
        df["median"],
        color="0.20",
        lw=1.8,
        label="Candidate median",
    )

    ax.plot(
        df.u,
        df.threshold,
        color="#d95f02",
        lw=2.2,
        ls="--",
        label=r"Threshold $\tau(u)$",
    )

    positive = np.concatenate([
        df.q10.to_numpy(),
        df.q90.to_numpy(),
        df.threshold.to_numpy(),
    ])

    positive = positive[
        np.isfinite(positive)
        & (positive > 0)
    ]

    if (
        len(positive)
        and positive.max()
        / positive.min()
        > 30
    ):
        ax.set_yscale("log")

    ax.set_xlim(0, 1)

    ax.set_xlabel(
        "Normalized hierarchy progress, u"
    )

    ax.set_ylabel(
        "Composite merger cost"
    )

    ax.grid(
        alpha=.14,
        linewidth=.6,
        which="both",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False,
        fontsize=6.4,
        ncol=2,
    )

    fig.tight_layout(pad=.7)

    save_panel(fig, "D")


# ============================================================
# E — use frozen expression panel
# ============================================================

def panel_E():

    if not EXPRESSION_PANEL.exists():
        raise RuntimeError(
            "Missing molecular panel. Run:\n"
            "scripts/figures/fig2_panel_E_expression.py"
        )

    img = plt.imread(
        EXPRESSION_PANEL
    )

    fig, ax = plt.subplots(
        figsize=(7.7, 3.4)
    )

    ax.imshow(img)
    ax.set_axis_off()

    fig.subplots_adjust(
        left=0,
        right=1,
        top=1,
        bottom=0,
    )

    save_panel(fig, "E")


# ============================================================
# F — candidates remain while admissibility goes to zero
# ============================================================

def panel_F(traj):

    files = select_candidate_files(
        candidate_files(),
        n=75,
    )

    rows = []

    for k, p in enumerate(files):

        step = step_from_name(p)

        pf = pq.ParquetFile(p)

        n_candidates = int(
            pf.metadata.num_rows
        )

        tab = retry_read_parquet(
            p,
            ["admissible"],
        )

        adm = (
            tab["admissible"]
            .to_numpy(
                zero_copy_only=False
            )
            .astype(bool)
        )

        n_adm = int(
            np.sum(adm)
        )

        nodes = nodes_at_step(
            traj,
            step,
        )

        rows.append({
            "step": step,
            "u": hierarchy_u(nodes),
            "nodes": nodes,
            "candidate_interfaces":
                n_candidates,
            "admissible_interfaces":
                n_adm,
            "admissible_fraction":
                (
                    n_adm / n_candidates
                    if n_candidates
                    else np.nan
                ),
        })

        print(
            f"F {k+1:2d}/{len(files)} "
            f"step={step} "
            f"cand={n_candidates} "
            f"adm={n_adm}"
        )

    df = (
        pd.DataFrame(rows)
        .sort_values("u")
    )

    df.to_csv(
        SOURCEDIR / "panel_F_interface_exhaustion.csv",
        index=False,
    )

    fig, ax = plt.subplots(
        figsize=(5.2, 3.6)
    )

    l1 = ax.plot(
        df.u,
        df.candidate_interfaces,
        lw=2.2,
        color="0.20",
        label="Candidate interfaces",
    )

    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)

    ax.set_xlabel(
        "Normalized hierarchy progress, u"
    )

    ax.set_ylabel(
        "Candidate interface count"
    )

    ax.grid(
        alpha=.14,
        linewidth=.6,
    )

    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()

    l2 = ax2.plot(
        df.u,
        100 * df.admissible_fraction,
        lw=2.3,
        color="#d95f02",
        label="Admissible fraction",
    )

    ax2.set_ylim(
        0,
        max(
            40,
            100
            * np.nanmax(
                df.admissible_fraction
            )
            * 1.12,
        ),
    )

    ax2.set_ylabel(
        "Admissible candidate interfaces (%)"
    )

    ax2.spines["top"].set_visible(False)

    handles = l1 + l2

    ax.legend(
        handles,
        [
            h.get_label()
            for h in handles
        ],
        frameon=False,
        fontsize=6.5,
        loc="upper right",
    )

    last = df.iloc[-1]

    if int(
        last.admissible_interfaces
    ) == 0:

        ax.scatter(
            [last.u],
            [last.candidate_interfaces],
            color="0.20",
            s=28,
            zorder=5,
        )

        ax2.scatter(
            [last.u],
            [0],
            color="#d95f02",
            s=28,
            zorder=5,
        )

        ax2.annotate(
            (
                f"{int(last.candidate_interfaces):,} "
                "physical candidates remain\n"
                r"$f_{\rm adm}=0$"
            ),
            xy=(
                last.u,
                0,
            ),
            xytext=(-10, 27),
            textcoords="offset points",
            ha="right",
            fontsize=6.6,
            weight="bold",
        )

    fig.tight_layout(pad=.7)

    save_panel(fig, "F")


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--panel",
        default="ALL",
        choices=[
            "ALL",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
        ],
    )

    args = parser.parse_args()

    traj = merge_trajectory()

    meta, arrays = (
        load_label_checkpoints()
    )

    cells, polygons, valid = (
        load_cells_and_polygons()
    )

    initial = meta.iloc[
        np.argmin(
            np.abs(meta.u.to_numpy())
        )
    ]

    terminal = meta.iloc[
        np.argmax(meta.u.to_numpy())
    ]

    crop = choose_crop(
        cells,
        arrays[int(initial.step)],
        arrays[int(terminal.step)],
    )

    jobs = (
        list("ABCDEF")
        if args.panel == "ALL"
        else [args.panel]
    )

    for letter in jobs:

        print()
        print("=" * 72)
        print(
            f"MAIN FIGURE 2 PANEL {letter}"
        )
        print("=" * 72)

        if letter == "A":

            panel_A(
                cells,
                polygons,
                valid,
                meta,
                arrays,
                crop,
            )

        elif letter == "B":

            panel_B(
                cells,
                polygons,
                meta,
                arrays,
                crop,
            )

        elif letter == "C":

            panel_C(
                traj,
                cells,
                meta,
                arrays,
                crop,
            )

        elif letter == "D":

            panel_D(
                traj
            )

        elif letter == "E":

            panel_E()

        elif letter == "F":

            panel_F(
                traj
            )


if __name__ == "__main__":
    main()
