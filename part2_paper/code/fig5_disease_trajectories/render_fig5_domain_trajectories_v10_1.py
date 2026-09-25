#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from scipy.interpolate import splprep, splev


ROOT = Path.home() / "Desktop" / "SUTRA"

V7 = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "structural_trajectory_v7"
)

V9 = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "relational_flow_v9"
)

V10 = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "domain_trajectories_v10"
)

OUT = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "domain_trajectories_v10_1"
)
OUT.mkdir(parents=True, exist_ok=True)

U = np.array([0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56])

DOMAINS = V10 / "domains_q80.csv"
RELATIONS = V10 / "relations_q80.csv"

DPI = 600

# ------------------------------------------------------------
# Presentation controls only
# ------------------------------------------------------------

TISSUE_SIZE = 0.42
TISSUE_ALPHA = 0.30

NODE_MIN = 10
NODE_MAX = 35

TRACK_LW_MIN = 1.05
TRACK_LW_MAX = 2.55

BRANCH_LW = 0.70

# Smooth track interpolation.
N_SPLINE = 100

# Deterministic perpendicular curvature added to 2-point tracks.
TWO_POINT_CURVATURE = 0.075

# Prevent extreme visual curves.
MAX_CURVE_FRACTION = 0.12

# Keep only primary Hungarian relations in hero backbone.
HERO_PRIMARY_ONLY = True

# Minimum number of observed hierarchy states in a hero trajectory.
MIN_TRACK_NODES = 2

EPS = 1e-12


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


def load():
    if not DOMAINS.exists():
        raise FileNotFoundError(DOMAINS)

    if not RELATIONS.exists():
        raise FileNotFoundError(RELATIONS)

    D = pd.read_csv(DOMAINS)
    E = pd.read_csv(RELATIONS)

    cells = pd.read_parquet(
        V7 / "prcc_cell_multiscale_gw.parquet",
        columns=["x", "y"],
    )

    cells = cells[
        np.isfinite(cells["x"]) &
        np.isfinite(cells["y"])
    ].copy()

    return cells, D, E


def node_key(u, domain_id):
    return (round(float(u), 8), int(domain_id))


def node_lookup(D):
    out = {}

    for _, r in D.iterrows():
        out[node_key(r["u"], r["domain_id"])] = r

    return out


def primary_edges(E):
    if "primary_hungarian" not in E.columns:
        raise RuntimeError("V10 primary_hungarian column missing")

    x = E[E["primary_hungarian"].astype(bool)].copy()

    return x.reset_index(drop=True)


def build_primary_tracks(D, E):
    """
    Build maximal directed paths from the frozen V10 primary backbone.

    Since primary Hungarian matching is at most one-to-one within each
    adjacent-scale transition, a persistent domain can be represented
    as a path across hierarchy coordinates.

    No matching is recomputed here.
    """
    E = primary_edges(E)

    nxt = {}
    prv = {}

    for _, e in E.iterrows():
        a = node_key(e["u_from"], e["domain_from"])
        b = node_key(e["u_to"], e["domain_to"])

        # Defensive audit: the primary backbone should not branch
        # within the same global graph.
        if a in nxt and nxt[a] != b:
            raise RuntimeError(
                f"Primary backbone branches at {a}: "
                f"{nxt[a]} versus {b}"
            )

        if b in prv and prv[b] != a:
            raise RuntimeError(
                f"Primary backbone merges at {b}: "
                f"{prv[b]} versus {a}"
            )

        nxt[a] = b
        prv[b] = a

    nodes = set(nxt) | set(prv)

    starts = sorted(
        [k for k in nodes if k not in prv],
        key=lambda z: (z[0], z[1]),
    )

    tracks = []
    visited = set()

    for s in starts:
        track = [s]
        visited.add(s)

        cur = s

        while cur in nxt:
            cur = nxt[cur]

            if cur in visited:
                raise RuntimeError("Cycle in hierarchy trajectory graph")

            track.append(cur)
            visited.add(cur)

        if len(track) >= MIN_TRACK_NODES:
            tracks.append(track)

    # Any unvisited nodes indicate malformed primary structure.
    orphan = nodes - visited

    if orphan:
        raise RuntimeError(
            f"Unvisited primary nodes: {len(orphan)}"
        )

    return tracks


def track_dataframe(track, lookup):
    rows = []

    for k in track:
        if k not in lookup:
            raise RuntimeError(f"Missing domain node {k}")

        rows.append(dict(lookup[k]))

    return pd.DataFrame(rows).sort_values("u")


def specimen_scale(cells):
    xr = float(cells["x"].max() - cells["x"].min())
    yr = float(cells["y"].max() - cells["y"].min())

    return max(math.hypot(xr, yr), EPS)


def two_point_curve(x, y, scale, sign):
    """
    Quadratic Bezier-like curve for a two-node trajectory.

    The control point is displaced perpendicular to the chord.
    """
    p0 = np.array([x[0], y[0]], dtype=float)
    p1 = np.array([x[1], y[1]], dtype=float)

    d = p1 - p0
    L = float(np.linalg.norm(d))

    if L <= EPS:
        return (
            np.linspace(x[0], x[1], N_SPLINE),
            np.linspace(y[0], y[1], N_SPLINE),
        )

    perp = np.array([-d[1], d[0]]) / L

    bend = min(
        TWO_POINT_CURVATURE * scale,
        MAX_CURVE_FRACTION * scale,
        0.32 * L,
    )

    control = 0.5 * (p0 + p1) + sign * bend * perp

    t = np.linspace(0, 1, N_SPLINE)

    pts = (
        ((1 - t) ** 2)[:, None] * p0 +
        (2 * (1 - t) * t)[:, None] * control +
        (t ** 2)[:, None] * p1
    )

    return pts[:, 0], pts[:, 1]


def smooth_track(x, y, track_index, scale):
    n = len(x)

    if n == 2:
        # Alternating but deterministic curvature prevents all
        # two-point tracks bending in exactly the same direction.
        sign = 1 if track_index % 2 == 0 else -1

        return two_point_curve(
            x, y, scale, sign
        )

    # Parameterization is hierarchy order, not spatial arc length.
    t = np.linspace(0, 1, n)

    # Interpolating spline: passes through the observed centroids.
    # k <= 3 as required by splprep.
    k = min(3, n - 1)

    try:
        tck, _ = splprep(
            [x, y],
            u=t,
            s=0.0,
            k=k,
        )

        tt = np.linspace(0, 1, N_SPLINE)

        xs, ys = splev(tt, tck)

        return np.asarray(xs), np.asarray(ys)

    except Exception:
        # Conservative fallback: piecewise interpolation.
        tt = np.linspace(0, 1, N_SPLINE)

        return (
            np.interp(tt, t, x),
            np.interp(tt, t, y),
        )


def track_strength(track, E):
    scores = []

    for a, b in zip(track[:-1], track[1:]):
        ua, da = a
        ub, db = b

        q = E[
            np.isclose(E["u_from"], ua) &
            (E["domain_from"].astype(int) == da) &
            np.isclose(E["u_to"], ub) &
            (E["domain_to"].astype(int) == db) &
            E["primary_hungarian"].astype(bool)
        ]

        if len(q):
            scores.append(float(q.iloc[0]["score"]))

    if not scores:
        return 0.0

    return float(np.mean(scores))


def add_arrowhead(ax, xs, ys, color, alpha, lw):
    """
    Only a short terminal arrowhead, not another full straight edge.
    """
    if len(xs) < 4:
        return

    i0 = max(0, len(xs) - 7)

    arrow = FancyArrowPatch(
        (xs[i0], ys[i0]),
        (xs[-1], ys[-1]),
        arrowstyle="-|>",
        mutation_scale=8.5,
        linewidth=lw,
        color=color,
        alpha=alpha,
        shrinkA=0,
        shrinkB=0,
        zorder=6,
    )

    ax.add_patch(arrow)


def draw_track(
    ax,
    T,
    track_index,
    scale,
    score,
    dmin,
    dmax,
    cmap,
):
    x = T["centroid_x"].to_numpy(float)
    y = T["centroid_y"].to_numpy(float)
    u = T["u"].to_numpy(float)
    d = T["mean_distortion"].to_numpy(float)

    xs, ys = smooth_track(
        x, y, track_index, scale
    )

    # Hierarchy progression is encoded along the curve.
    # The curve itself is not colored by increase/decrease.
    ts = np.linspace(0, 1, len(xs))

    points = np.column_stack([xs, ys])
    segs = np.stack(
        [points[:-1], points[1:]],
        axis=1,
    )

    lc = LineCollection(
        segs,
        cmap="viridis",
        norm=plt.Normalize(0, 1),
        linewidths=(
            TRACK_LW_MIN +
            (TRACK_LW_MAX - TRACK_LW_MIN) *
            np.clip(score, 0, 1)
        ),
        alpha=0.76,
        zorder=4,
    )

    lc.set_array(ts[:-1])
    ax.add_collection(lc)

    terminal_color = plt.get_cmap("viridis")(0.98)

    add_arrowhead(
        ax,
        xs,
        ys,
        terminal_color,
        0.90,
        TRACK_LW_MIN +
        0.8 * np.clip(score, 0, 1),
    )

    # Observed domain centroids remain visible, but much smaller
    # than V10.
    node_sizes = (
        NODE_MIN +
        (NODE_MAX - NODE_MIN) *
        (u - U.min()) /
        max(U.max() - U.min(), EPS)
    )

    node_alpha = (
        0.45 +
        0.50 *
        (u - U.min()) /
        max(U.max() - U.min(), EPS)
    )

    for xx, yy, dd, ss, aa in zip(
        x, y, d, node_sizes, node_alpha
    ):
        c = cmap(
            np.clip(
                (dd - dmin) /
                max(dmax - dmin, EPS),
                0, 1,
            )
        )

        ax.scatter(
            [xx],
            [yy],
            s=ss,
            c=[c],
            alpha=aa,
            edgecolors="white",
            linewidths=0.35,
            zorder=5,
        )


def hero(cells, D, E, tracks):
    lookup = node_lookup(D)
    scale = specimen_scale(cells)

    vals = D["mean_distortion"].to_numpy(float)
    dmin = float(np.quantile(vals, 0.02))
    dmax = float(np.quantile(vals, 0.98))

    cmap = plt.get_cmap("magma")

    strengths = [
        track_strength(t, E)
        for t in tracks
    ]

    fig, ax = plt.subplots(
        figsize=(7.4, 7.4)
    )

    # Anatomy should be visible enough to read tissue structure.
    ax.scatter(
        cells["x"],
        cells["y"],
        s=TISSUE_SIZE,
        c="0.52",
        alpha=TISSUE_ALPHA,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # Stronger/longer trajectories first, weak ones later would
    # obscure them; therefore draw weak first.
    order = sorted(
        range(len(tracks)),
        key=lambda i: (
            strengths[i],
            len(tracks[i]),
        ),
    )

    for i in order:
        T = track_dataframe(
            tracks[i],
            lookup,
        )

        draw_track(
            ax,
            T,
            i,
            scale,
            strengths[i],
            dmin,
            dmax,
            cmap,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    # Small hierarchy legend only.
    sm_u = plt.cm.ScalarMappable(
        norm=plt.Normalize(U.min(), U.max()),
        cmap="viridis",
    )
    sm_u.set_array([])

    cb1 = fig.colorbar(
        sm_u,
        ax=ax,
        fraction=0.025,
        pad=0.008,
        shrink=0.52,
    )
    cb1.set_label(
        "Hierarchy coordinate, $u$",
        fontsize=8,
    )
    cb1.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.08)
    save(fig, "hero_curved_relational_trajectories")


def persistent_only(cells, D, E, tracks):
    """
    More selective presentation: tracks spanning >=3 observed
    hierarchy states. Selection criterion is purely persistence.
    """
    keep = [
        t for t in tracks
        if len(t) >= 3
    ]

    lookup = node_lookup(D)
    scale = specimen_scale(cells)

    vals = D["mean_distortion"].to_numpy(float)
    dmin = float(np.quantile(vals, 0.02))
    dmax = float(np.quantile(vals, 0.98))
    cmap = plt.get_cmap("magma")

    strengths = [
        track_strength(t, E)
        for t in keep
    ]

    fig, ax = plt.subplots(
        figsize=(7.4, 7.4)
    )

    ax.scatter(
        cells["x"],
        cells["y"],
        s=TISSUE_SIZE,
        c="0.50",
        alpha=0.27,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    order = sorted(
        range(len(keep)),
        key=lambda i: strengths[i],
    )

    for i in order:
        T = track_dataframe(
            keep[i],
            lookup,
        )

        draw_track(
            ax,
            T,
            i,
            scale,
            strengths[i],
            dmin,
            dmax,
            cmap,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    sm_u = plt.cm.ScalarMappable(
        norm=plt.Normalize(U.min(), U.max()),
        cmap="viridis",
    )
    sm_u.set_array([])

    cb = fig.colorbar(
        sm_u,
        ax=ax,
        fraction=0.025,
        pad=0.008,
        shrink=0.52,
    )
    cb.set_label(
        "Hierarchy coordinate, $u$",
        fontsize=8,
    )
    cb.ax.tick_params(labelsize=7)

    fig.tight_layout(pad=0.08)
    save(fig, "hero_persistent_curved_trajectories")

    return keep


def audit_curved_network(cells, D, E):
    """
    All frozen V10 correspondences, including secondary split/merge
    relations. Curved independently for readability.

    This is an audit asset, not the proposed hero panel.
    """
    fig, ax = plt.subplots(figsize=(7.4, 7.4))

    ax.scatter(
        cells["x"],
        cells["y"],
        s=0.35,
        c="0.70",
        alpha=0.24,
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    scale = specimen_scale(cells)

    vals = D["mean_distortion"].to_numpy(float)
    dmin = float(np.quantile(vals, 0.02))
    dmax = float(np.quantile(vals, 0.98))

    cmap = plt.get_cmap("magma")

    lookup = node_lookup(D)

    for idx, e in E.iterrows():
        p0 = np.array(
            [e["x_from"], e["y_from"]],
            dtype=float,
        )
        p1 = np.array(
            [e["x_to"], e["y_to"]],
            dtype=float,
        )

        dvec = p1 - p0
        L = np.linalg.norm(dvec)

        if L <= EPS:
            continue

        perp = np.array(
            [-dvec[1], dvec[0]]
        ) / L

        # Deterministic curvature sign based on node IDs.
        code = (
            int(e["domain_from"]) * 31 +
            int(e["domain_to"]) * 17 +
            int(round(e["u_from"] * 100))
        )

        sign = 1 if code % 2 == 0 else -1

        bend = min(
            0.045 * scale,
            0.25 * L,
        )

        c = 0.5 * (p0 + p1) + sign * bend * perp

        t = np.linspace(0, 1, 45)

        pts = (
            ((1 - t) ** 2)[:, None] * p0 +
            (2 * (1 - t) * t)[:, None] * c +
            (t ** 2)[:, None] * p1
        )

        primary = bool(e["primary_hungarian"])

        ax.plot(
            pts[:, 0],
            pts[:, 1],
            color=(
                "#333333"
                if primary else "#777777"
            ),
            lw=(
                1.0 + 1.2 * e["score"]
                if primary else BRANCH_LW
            ),
            alpha=(
                0.58 if primary else 0.22
            ),
            zorder=2,
        )

    # Small nodes.
    for _, r in D.iterrows():
        c = cmap(
            np.clip(
                (r["mean_distortion"] - dmin) /
                max(dmax - dmin, EPS),
                0, 1,
            )
        )

        ax.scatter(
            [r["centroid_x"]],
            [r["centroid_y"]],
            s=10,
            c=[c],
            edgecolors="white",
            linewidths=0.25,
            zorder=3,
        )

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout(pad=0.05)
    save(fig, "audit_all_curved_correspondences")


def export_tracks(D, E, tracks):
    lookup = node_lookup(D)

    rows = []

    for tid, track in enumerate(tracks):
        strength = track_strength(
            track, E
        )

        for order, k in enumerate(track):
            r = lookup[k]

            rows.append({
                "track_id": tid,
                "track_order": order,
                "n_track_nodes": len(track),
                "mean_edge_score": strength,
                "u": float(r["u"]),
                "domain_id":
                    int(r["domain_id"]),
                "centroid_x":
                    float(r["centroid_x"]),
                "centroid_y":
                    float(r["centroid_y"]),
                "n_pixels":
                    int(r["n_pixels"]),
                "mean_distortion":
                    float(r["mean_distortion"]),
            })

    T = pd.DataFrame(rows)

    T.to_csv(
        OUT / "primary_backbone_tracks.csv",
        index=False,
    )

    summary = (
        T.groupby("track_id")
        .agg(
            n_nodes=("u", "size"),
            u_start=("u", "min"),
            u_end=("u", "max"),
            mean_edge_score=(
                "mean_edge_score", "first"
            ),
            max_distortion=(
                "mean_distortion", "max"
            ),
            mean_distortion=(
                "mean_distortion", "mean"
            ),
        )
        .reset_index()
        .sort_values(
            ["n_nodes", "mean_edge_score"],
            ascending=[False, False],
        )
    )

    summary.to_csv(
        OUT / "track_summary.csv",
        index=False,
    )

    return T, summary


def main():
    cells, D, E = load()

    print("domains:", len(D))
    print("all frozen relations:", len(E))
    print(
        "primary relations:",
        int(E["primary_hungarian"].astype(bool).sum())
    )

    tracks = build_primary_tracks(
        D, E
    )

    T, S = export_tracks(
        D, E, tracks
    )

    print("\nPrimary trajectories:", len(tracks))
    print(
        "tracks spanning >=3 states:",
        int(np.sum(S["n_nodes"] >= 3)),
    )
    print(
        "tracks spanning >=4 states:",
        int(np.sum(S["n_nodes"] >= 4)),
    )

    print("\nLongest tracks:")
    print(
        S.head(20).to_string(index=False)
    )

    hero(
        cells,
        D,
        E,
        tracks,
    )

    persistent = persistent_only(
        cells,
        D,
        E,
        tracks,
    )

    audit_curved_network(
        cells,
        D,
        E,
    )

    provenance = {
        "version": "domain_trajectories_v10_1",
        "analysis_changed": False,
        "domain_source": str(DOMAINS),
        "relation_source": str(RELATIONS),
        "primary_quantile": 0.80,
        "hero_edges": (
            "frozen V10 primary Hungarian correspondence "
            "backbone only"
        ),
        "track_definition": (
            "maximal consecutive paths through the frozen "
            "one-to-one primary correspondence backbone"
        ),
        "curve_definition": (
            "interpolating spatial spline through observed domain "
            "centroids; two-node paths use deterministic quadratic "
            "Bezier curvature"
        ),
        "persistent_panel": (
            "same frozen primary tracks restricted to trajectories "
            "observed at three or more hierarchy coordinates"
        ),
        "audit_panel": (
            "all frozen V10 primary and secondary correspondences "
            "rendered as curved edges"
        ),
        "interpretation": (
            "curves connect corresponding relational-displacement "
            "domains across adjacent SUTRA organizational scales; "
            "they do not represent cell motion, RNA velocity, "
            "lineage, biological time, or causal transport"
        ),
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
