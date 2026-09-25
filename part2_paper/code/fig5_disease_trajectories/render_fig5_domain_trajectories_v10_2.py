#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.interpolate import splprep, splev
from scipy.spatial import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch


ROOT = Path.home() / "Desktop" / "SUTRA"

V7 = ROOT / "results/Fig5_Disease_Trajectories/structural_trajectory_v7"
V9 = ROOT / "results/Fig5_Disease_Trajectories/relational_flow_v9"
V10 = ROOT / "results/Fig5_Disease_Trajectories/domain_trajectories_v10"
V101 = ROOT / "results/Fig5_Disease_Trajectories/domain_trajectories_v10_1"

OUT = ROOT / "results/Fig5_Disease_Trajectories/domain_trajectories_v10_2"
OUT.mkdir(parents=True, exist_ok=True)

U = np.array([0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56])

DPI = 600
EPS = 1e-12

# ------------------------------------------------------------
# VISUAL HIERARCHY
# ------------------------------------------------------------

# >= 7 states: hero
# 4–6 states: supporting
# 3 states: context
# <=2 states: absent from hero
STYLE = {
    "hero": {
        "lw": 4.0,
        "alpha": 0.98,
        "node": 20,
        "z": 8,
    },
    "support": {
        "lw": 2.0,
        "alpha": 0.58,
        "node": 11,
        "z": 6,
    },
    "context": {
        "lw": 0.95,
        "alpha": 0.20,
        "node": 6,
        "z": 4,
    },
}

N_CURVE = 180

# Background field opacity.
DELTA_ALPHA = 0.28

# Tissue support.
CELL_ALPHA = 0.07
CELL_SIZE = 0.20
HULL_ALPHA = 0.42
HULL_LW = 0.75


def save(fig, stem):
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def load_inputs():
    cells = pd.read_parquet(
        V7 / "prcc_cell_multiscale_gw.parquet",
        columns=["x", "y"],
    )

    cells = cells[
        np.isfinite(cells["x"]) &
        np.isfinite(cells["y"])
    ].copy()

    domains = pd.read_csv(
        V10 / "domains_q80.csv"
    )

    relations = pd.read_csv(
        V10 / "relations_q80.csv"
    )

    tracks = pd.read_csv(
        V101 / "primary_backbone_tracks.csv"
    )

    summary = pd.read_csv(
        V101 / "track_summary.csv"
    )

    return cells, domains, relations, tracks, summary


def load_v9_field():
    p = V9 / "relational_flow_fields.npz"
    if not p.exists():
        raise FileNotFoundError(p)

    z = np.load(p, allow_pickle=False)

    xc = np.asarray(z["x_centers"], float)
    yc = np.asarray(z["y_centers"], float)

    A = np.asarray(z["D_u0.40"], float) - np.asarray(
        z["D_u0.32"], float
    )

    return xc, yc, A


def specimen_scale(cells):
    xr = float(cells.x.max() - cells.x.min())
    yr = float(cells.y.max() - cells.y.min())
    return max(math.hypot(xr, yr), EPS)


def track_class(n):
    if n >= 7:
        return "hero"
    if n >= 4:
        return "support"
    if n >= 3:
        return "context"
    return "omit"


def smooth_track(T, track_id, scale):
    x = T.centroid_x.to_numpy(float)
    y = T.centroid_y.to_numpy(float)
    n = len(T)

    if n == 2:
        p0 = np.array([x[0], y[0]])
        p1 = np.array([x[1], y[1]])

        d = p1 - p0
        L = np.linalg.norm(d)

        if L <= EPS:
            return x, y

        perp = np.array([-d[1], d[0]]) / L
        sign = 1 if int(track_id) % 2 == 0 else -1

        bend = min(
            0.06 * scale,
            0.28 * L,
        )

        c = 0.5 * (p0 + p1) + sign * bend * perp

        t = np.linspace(0, 1, N_CURVE)

        P = (
            ((1 - t) ** 2)[:, None] * p0 +
            (2 * (1 - t) * t)[:, None] * c +
            (t ** 2)[:, None] * p1
        )

        return P[:, 0], P[:, 1]

    t = np.linspace(0, 1, n)
    k = min(3, n - 1)

    try:
        tck, _ = splprep(
            [x, y],
            u=t,
            s=0,
            k=k,
        )

        tt = np.linspace(0, 1, N_CURVE)
        xs, ys = splev(tt, tck)

        return np.asarray(xs), np.asarray(ys)

    except Exception:
        tt = np.linspace(0, 1, N_CURVE)

        return (
            np.interp(tt, t, x),
            np.interp(tt, t, y),
        )


def draw_tissue(ax, cells):
    # Very faint real-cell cloud.
    ax.scatter(
        cells.x,
        cells.y,
        s=CELL_SIZE,
        c="0.28",
        alpha=CELL_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # Tissue boundary from convex hull, presentation only.
    pts = cells[["x", "y"]].to_numpy(float)

    hull = ConvexHull(pts)
    hp = pts[hull.vertices]
    hp = np.vstack([hp, hp[0]])

    ax.plot(
        hp[:, 0],
        hp[:, 1],
        color="0.25",
        lw=HULL_LW,
        alpha=HULL_ALPHA,
        zorder=2,
    )


def draw_delta_field(ax, xc, yc, A):
    finite = np.isfinite(A)

    q = np.quantile(
        np.abs(A[finite]),
        0.98,
    )

    # Keep zero neutral and clip only display extremes.
    im = ax.imshow(
        np.ma.masked_invalid(A),
        extent=[
            xc.min(), xc.max(),
            yc.min(), yc.max(),
        ],
        origin="lower",
        cmap="RdBu_r",
        vmin=-q,
        vmax=q,
        interpolation="bilinear",
        alpha=DELTA_ALPHA,
        zorder=0,
        aspect="equal",
    )

    return im, q


def colored_curve(
    ax,
    xs,
    ys,
    style,
):
    pts = np.column_stack([xs, ys])

    segs = np.stack(
        [pts[:-1], pts[1:]],
        axis=1,
    )

    t = np.linspace(0, 1, len(xs) - 1)

    lc = LineCollection(
        segs,
        cmap="viridis",
        norm=plt.Normalize(0, 1),
        linewidths=style["lw"],
        alpha=style["alpha"],
        zorder=style["z"],
        capstyle="round",
        joinstyle="round",
    )

    lc.set_array(t)
    ax.add_collection(lc)


def terminal_arrow(
    ax,
    xs,
    ys,
    style,
):
    if len(xs) < 10:
        return

    # Tiny arrow only over the final few percent.
    i = max(0, len(xs) - 7)

    terminal_color = plt.get_cmap("viridis")(1.0)

    a = FancyArrowPatch(
        (xs[i], ys[i]),
        (xs[-1], ys[-1]),
        arrowstyle="-|>",
        mutation_scale=(
            8 if style["lw"] >= 3 else 5.5
        ),
        linewidth=max(0.7, style["lw"] * 0.55),
        color=terminal_color,
        alpha=style["alpha"],
        shrinkA=0,
        shrinkB=0,
        zorder=style["z"] + 1,
    )

    ax.add_patch(a)


def draw_nodes(
    ax,
    T,
    style,
    distortion_norm,
):
    cmap = plt.get_cmap("magma")

    for _, r in T.iterrows():
        c = cmap(
            distortion_norm(
                float(r.mean_distortion)
            )
        )

        ax.scatter(
            [r.centroid_x],
            [r.centroid_y],
            s=style["node"],
            c=[c],
            alpha=style["alpha"],
            edgecolors="white",
            linewidths=0.35,
            zorder=style["z"] + 2,
        )


def label_complete_track(
    ax,
    T,
    track_number,
):
    """
    Label hierarchy coordinate only on complete seven-state tracks.
    Small labels offset from knots.
    """
    offsets = [
        (5, 5),
        (5, -9),
        (5, 5),
        (5, -9),
        (5, 5),
        (5, -9),
        (5, 5),
    ]

    for j, (_, r) in enumerate(T.iterrows()):
        dx, dy = offsets[j % len(offsets)]

        ax.annotate(
            f"{r.u:.2f}",
            (r.centroid_x, r.centroid_y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=5.8,
            color="0.12",
            alpha=0.88,
            zorder=20,
        )

    # Track ID label at beginning.
    first = T.iloc[0]

    ax.annotate(
        f"T{track_number}",
        (first.centroid_x, first.centroid_y),
        xytext=(-12, 4),
        textcoords="offset points",
        fontsize=7.2,
        fontweight="bold",
        color="0.08",
        zorder=21,
    )


def render_hero(
    cells,
    tracks,
    summary,
    xc,
    yc,
    A,
):
    scale = specimen_scale(cells)

    # Distortion normalization comes from displayed track nodes.
    vals = tracks.mean_distortion.to_numpy(float)

    lo = float(np.quantile(vals, 0.02))
    hi = float(np.quantile(vals, 0.98))

    norm = plt.Normalize(lo, hi)

    fig, ax = plt.subplots(
        figsize=(8.0, 7.6)
    )

    im, q = draw_delta_field(
        ax, xc, yc, A
    )

    draw_tissue(ax, cells)

    # Draw weakest first.
    summary = summary.copy()
    summary["class"] = summary.n_nodes.map(track_class)

    summary = summary[
        summary["class"] != "omit"
    ]

    rank = {
        "context": 0,
        "support": 1,
        "hero": 2,
    }

    summary["rank"] = summary["class"].map(rank)

    summary = summary.sort_values(
        ["rank", "mean_edge_score"],
        ascending=[True, True],
    )

    complete_count = 0

    for _, s in summary.iterrows():
        tid = int(s.track_id)
        cls = s["class"]
        style = STYLE[cls]

        T = (
            tracks[tracks.track_id == tid]
            .sort_values("track_order")
            .copy()
        )

        xs, ys = smooth_track(
            T,
            tid,
            scale,
        )

        colored_curve(
            ax,
            xs,
            ys,
            style,
        )

        terminal_arrow(
            ax,
            xs,
            ys,
            style,
        )

        draw_nodes(
            ax,
            T,
            style,
            norm,
        )

        if cls == "hero":
            complete_count += 1

            label_complete_track(
                ax,
                T,
                complete_count,
            )

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Compact legends rather than huge bars.
    sm_u = plt.cm.ScalarMappable(
        norm=plt.Normalize(U.min(), U.max()),
        cmap="viridis",
    )
    sm_u.set_array([])

    cb_u = fig.colorbar(
        sm_u,
        ax=ax,
        location="bottom",
        fraction=0.025,
        pad=0.018,
        shrink=0.40,
        aspect=30,
    )

    cb_u.set_ticks(
        [0.08, 0.32, 0.56]
    )
    cb_u.set_label(
        "SUTRA hierarchy coordinate, $u$",
        fontsize=7.5,
    )
    cb_u.ax.tick_params(labelsize=6.5)

    fig.tight_layout(pad=0.05)

    save(
        fig,
        "hero_relational_trajectories_v10_2",
    )


def render_complete_tracks_only(
    cells,
    tracks,
    summary,
    xc,
    yc,
    A,
):
    """
    Extremely clean candidate: only trajectories represented at all
    seven hierarchy coordinates.
    """
    keep = summary[
        summary.n_nodes >= 7
    ].copy()

    scale = specimen_scale(cells)

    vals = tracks.mean_distortion.to_numpy(float)
    norm = plt.Normalize(
        np.quantile(vals, 0.02),
        np.quantile(vals, 0.98),
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 7.6)
    )

    draw_delta_field(
        ax, xc, yc, A
    )

    draw_tissue(ax, cells)

    for j, (_, s) in enumerate(
        keep.sort_values(
            "mean_edge_score",
            ascending=False,
        ).iterrows(),
        start=1,
    ):
        tid = int(s.track_id)

        T = (
            tracks[tracks.track_id == tid]
            .sort_values("track_order")
        )

        xs, ys = smooth_track(
            T,
            tid,
            scale,
        )

        colored_curve(
            ax,
            xs,
            ys,
            STYLE["hero"],
        )

        terminal_arrow(
            ax,
            xs,
            ys,
            STYLE["hero"],
        )

        draw_nodes(
            ax,
            T,
            STYLE["hero"],
            norm,
        )

        label_complete_track(
            ax,
            T,
            j,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout(pad=0.05)

    save(
        fig,
        "hero_two_complete_trajectories_v10_2",
    )


def render_no_field(
    cells,
    tracks,
    summary,
):
    """
    Control candidate to determine whether the V9 field helps or
    merely makes the hero busy.
    """
    scale = specimen_scale(cells)

    vals = tracks.mean_distortion.to_numpy(float)
    norm = plt.Normalize(
        np.quantile(vals, 0.02),
        np.quantile(vals, 0.98),
    )

    fig, ax = plt.subplots(
        figsize=(8.0, 7.6)
    )

    draw_tissue(ax, cells)

    S = summary[
        summary.n_nodes >= 3
    ].copy()

    S["class"] = S.n_nodes.map(track_class)

    rank = {
        "context": 0,
        "support": 1,
        "hero": 2,
    }

    S["rank"] = S["class"].map(rank)

    S = S.sort_values(
        ["rank", "mean_edge_score"]
    )

    complete_count = 0

    for _, s in S.iterrows():
        tid = int(s.track_id)
        cls = s["class"]

        T = (
            tracks[tracks.track_id == tid]
            .sort_values("track_order")
        )

        xs, ys = smooth_track(
            T,
            tid,
            scale,
        )

        colored_curve(
            ax,
            xs,
            ys,
            STYLE[cls],
        )

        terminal_arrow(
            ax,
            xs,
            ys,
            STYLE[cls],
        )

        draw_nodes(
            ax,
            T,
            STYLE[cls],
            norm,
        )

        if cls == "hero":
            complete_count += 1
            label_complete_track(
                ax,
                T,
                complete_count,
            )

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout(pad=0.05)

    save(
        fig,
        "hero_relational_trajectories_no_field_v10_2",
    )


def export_selected(summary):
    S = summary.copy()
    S["display_class"] = S.n_nodes.map(
        track_class
    )

    S.to_csv(
        OUT / "trajectory_display_hierarchy.csv",
        index=False,
    )

    print("\nDisplay hierarchy:")
    print(
        S.sort_values(
            ["n_nodes", "mean_edge_score"],
            ascending=[False, False],
        ).to_string(index=False)
    )


def main():
    cells, domains, relations, tracks, summary = load_inputs()

    xc, yc, A = load_v9_field()

    print("cells:", len(cells))
    print("tracks:", summary.shape[0])

    print(
        "7-state:",
        int(np.sum(summary.n_nodes >= 7))
    )
    print(
        "4–6-state:",
        int(np.sum(
            (summary.n_nodes >= 4) &
            (summary.n_nodes <= 6)
        ))
    )
    print(
        "3-state:",
        int(np.sum(summary.n_nodes == 3))
    )

    finite = np.isfinite(A)

    print(
        "V9 .32→.40 field:",
        "mean=",
        float(np.nanmean(A)),
        "median=",
        float(np.nanmedian(A)),
        "positive fraction=",
        float(np.mean(A[finite] > 0)),
    )

    export_selected(summary)

    render_hero(
        cells,
        tracks,
        summary,
        xc,
        yc,
        A,
    )

    render_complete_tracks_only(
        cells,
        tracks,
        summary,
        xc,
        yc,
        A,
    )

    render_no_field(
        cells,
        tracks,
        summary,
    )

    provenance = {
        "version": "domain_trajectories_v10_2",
        "analysis_changed": False,
        "domains_changed": False,
        "correspondences_changed": False,
        "track_decomposition_changed": False,
        "primary_domain_quantile": 0.80,
        "visual_hierarchy": {
            "hero": "tracks represented at all 7 hierarchy coordinates",
            "support": "tracks represented at 4-6 coordinates",
            "context": "tracks represented at exactly 3 coordinates",
            "omitted_from_hero": "tracks represented at <=2 coordinates",
        },
        "background_field": (
            "Frozen V9 cell-derived spatial relational-displacement "
            "change D(x,0.40)-D(x,0.32), used only as contextual "
            "background for the strongest amplification interval."
        ),
        "curve_color": (
            "SUTRA hierarchy coordinate progression, not velocity "
            "magnitude or biological time."
        ),
        "node_color": (
            "mean GW relational displacement of the frozen V10 "
            "domain."
        ),
        "trajectory_interpretation": (
            "persistent correspondence of spatial relational-"
            "displacement domains across ordered SUTRA hierarchy "
            "states."
        ),
        "not_interpretable_as": [
            "cell motion",
            "RNA velocity",
            "lineage",
            "biological time",
            "causal transport",
            "disease progression",
        ],
    }

    (OUT / "provenance.json").write_text(
        json.dumps(
            provenance,
            indent=2,
        )
    )

    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
