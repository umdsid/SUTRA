#!/usr/bin/env python3

from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.vq import kmeans2
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import pdist


ROOT = Path.home() / "Desktop" / "SUTRA"

IN = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "structural_trajectory_v7"
)

V4 = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "gw_biology_v4"
)

OUT = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "trajectory_maps_v8"
)

OUT.mkdir(parents=True, exist_ok=True)

U = np.array([0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56])
GWCOLS = [f"gw_u{u:.2f}" for u in U]

# Spatial resolution only. Territories are defined solely from x,y.
N_TERRITORIES = 96
SEED = 1729

DPI = 600


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


def weighted_mean(values, weights):
    ok = np.isfinite(values) & np.isfinite(weights)
    if not np.any(ok):
        return np.nan
    return np.average(values[ok], weights=weights[ok])


def robust_scale_xy(xy):
    lo = np.nanpercentile(xy, 1, axis=0)
    hi = np.nanpercentile(xy, 99, axis=0)

    span = hi - lo
    span[span <= 0] = 1.0

    z = (xy - lo) / span
    return z


def deterministic_spatial_territories(df):
    """
    Territory construction is intentionally blind to all GW values.

    kmeans2 receives only scaled x,y coordinates.
    """
    xy = df[["x", "y"]].to_numpy(float)

    if not np.all(np.isfinite(xy)):
        raise RuntimeError("Non-finite spatial coordinates")

    z = robust_scale_xy(xy)

    rng = np.random.default_rng(SEED)

    # Deterministic spatial seed points distributed through the specimen.
    # Start from one random cell, then farthest-point sampling in x,y.
    n = len(z)

    seeds = np.empty((N_TERRITORIES, 2), dtype=float)

    first = int(rng.integers(0, n))
    seeds[0] = z[first]

    d2 = np.sum((z - seeds[0]) ** 2, axis=1)

    for k in range(1, N_TERRITORIES):
        idx = int(np.argmax(d2))
        seeds[k] = z[idx]

        dk = np.sum((z - seeds[k]) ** 2, axis=1)
        d2 = np.minimum(d2, dk)

    centroids, labels = kmeans2(
        z,
        seeds,
        iter=100,
        minit="matrix",
        missing="raise",
    )

    labels = labels.astype(int)

    if len(np.unique(labels)) != N_TERRITORIES:
        raise RuntimeError(
            "Spatial territory construction produced empty territories"
        )

    return labels, centroids


def build_territory_tables(df):
    rows = []

    for t, g in df.groupby("territory", sort=True):
        r = {
            "territory": int(t),
            "n_cells": int(len(g)),
            "x_centroid": float(g["x"].mean()),
            "y_centroid": float(g["y"].mean()),
        }

        for u, col in zip(U, GWCOLS):
            x = g[col].to_numpy(float)
            finite = np.isfinite(x)

            r[f"n_finite_u{u:.2f}"] = int(finite.sum())

            r[f"mean_u{u:.2f}"] = (
                float(np.mean(x[finite]))
                if finite.any() else np.nan
            )

            r[f"median_u{u:.2f}"] = (
                float(np.median(x[finite]))
                if finite.any() else np.nan
            )

            r[f"q90_u{u:.2f}"] = (
                float(np.quantile(x[finite], 0.90))
                if finite.any() else np.nan
            )

        rows.append(r)

    return pd.DataFrame(rows)


def trajectory_matrix(T, prefix="mean"):
    cols = [f"{prefix}_u{u:.2f}" for u in U]
    return T[cols].to_numpy(float)


def order_by_trajectory(M):
    """
    Ordering only affects visualization.
    It does NOT define territories or any statistic.
    """
    X = M.copy()

    # Fill rare missing territory/scale values by column median.
    for j in range(X.shape[1]):
        med = np.nanmedian(X[:, j])
        X[~np.isfinite(X[:, j]), j] = med

    # Standardize columns only for ordering, not for displayed values.
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1

    Z = (X - mu) / sd

    if len(Z) <= 2:
        return np.arange(len(Z))

    L = linkage(
        pdist(Z, metric="euclidean"),
        method="average",
    )

    return leaves_list(L)


def plot_absolute_heatmap(T, M, order):
    X = M[order]

    finite = X[np.isfinite(X)]

    # Preserve one common scale across all hierarchy coordinates.
    # Clip only the extreme display tail; CSV retains raw values.
    vmax = float(np.quantile(finite, 0.99))
    vmin = 0.0

    fig, ax = plt.subplots(figsize=(4.4, 7.0))

    im = ax.imshow(
        X,
        aspect="auto",
        interpolation="nearest",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(U)))
    ax.set_xticklabels([f"{u:.2f}" for u in U])
    ax.set_xlabel("SUTRA hierarchy coordinate, $u$")
    ax.set_ylabel("Spatial territories")

    ax.set_yticks([])

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label("Mean PRCC GW conditional distortion")

    fig.tight_layout()
    save(fig, "structural_trajectory_absolute")


def plot_centered_heatmap(M, order):
    # Center each territory around its own seven-scale median.
    baseline = np.nanmedian(M, axis=1, keepdims=True)
    C = M - baseline

    X = C[order]
    finite = np.abs(X[np.isfinite(X)])

    vmax = float(np.quantile(finite, 0.99))
    vmax = max(vmax, 1e-12)

    fig, ax = plt.subplots(figsize=(4.4, 7.0))

    im = ax.imshow(
        X,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(len(U)))
    ax.set_xticklabels([f"{u:.2f}" for u in U])
    ax.set_xlabel("SUTRA hierarchy coordinate, $u$")
    ax.set_ylabel("Same spatial territories")
    ax.set_yticks([])

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025)
    cb.set_label(
        "Change from territory's multiscale median"
    )

    fig.tight_layout()
    save(fig, "structural_trajectory_centered")

    return C


def plot_territory_map(df, T, order):
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))

    territory_to_rank = {
        int(T.iloc[i]["territory"]): int(rank[i])
        for i in range(len(T))
    }

    c = df["territory"].map(territory_to_rank).to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    ax.scatter(
        df["x"],
        df["y"],
        c=c,
        s=1.2,
        cmap="turbo",
        rasterized=True,
        linewidths=0,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    fig.tight_layout(pad=0)
    save(fig, "spatial_territories")


def plot_u040_map(df):
    x = df["gw_u0.40"].to_numpy(float)
    finite = x[np.isfinite(x)]

    vmax = float(np.quantile(finite, 0.99))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))

    # Background cells outside dominant component.
    missing = ~np.isfinite(x)

    if missing.any():
        ax.scatter(
            df.loc[missing, "x"],
            df.loc[missing, "y"],
            s=0.8,
            c="0.88",
            rasterized=True,
            linewidths=0,
        )

    ok = ~missing

    sc = ax.scatter(
        df.loc[ok, "x"],
        df.loc[ok, "y"],
        c=x[ok],
        s=1.3,
        cmap="magma",
        vmin=0,
        vmax=vmax,
        rasterized=True,
        linewidths=0,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    cb = fig.colorbar(
        sc,
        ax=ax,
        fraction=0.035,
        pad=0.015,
    )
    cb.set_label(
        "PRCC GW conditional distortion, $u=0.40$"
    )

    fig.tight_layout(pad=0.15)
    save(fig, "PRCC_GW_distortion_u040")


def plot_global_gw():
    p = IN / "scale_summary.csv"
    S = pd.read_csv(p)

    fig, ax = plt.subplots(figsize=(4.2, 2.8))

    ax.plot(
        S["u_target"],
        S["gw_distance"],
        marker="o",
        lw=1.6,
        ms=4,
    )

    ax.set_xlabel("SUTRA hierarchy coordinate, $u$")
    ax.set_ylabel("$d_{GW}$")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save(fig, "global_GW_trajectory")


def export_ordered_tables(T, M, C, order):
    ordered = T.iloc[order].copy().reset_index(drop=True)
    ordered.insert(0, "heatmap_row", np.arange(len(ordered)))

    ordered.to_csv(
        OUT / "territory_metadata_ordered.csv",
        index=False,
    )

    abs_df = pd.DataFrame(
        M[order],
        columns=[f"u_{u:.2f}" for u in U],
    )
    abs_df.insert(0, "heatmap_row", np.arange(len(abs_df)))
    abs_df.insert(
        1,
        "territory",
        T.iloc[order]["territory"].to_numpy(),
    )

    abs_df.to_csv(
        OUT / "territory_absolute_matrix.csv",
        index=False,
    )

    cen_df = pd.DataFrame(
        C[order],
        columns=[f"u_{u:.2f}" for u in U],
    )
    cen_df.insert(0, "heatmap_row", np.arange(len(cen_df)))
    cen_df.insert(
        1,
        "territory",
        T.iloc[order]["territory"].to_numpy(),
    )

    cen_df.to_csv(
        OUT / "territory_centered_matrix.csv",
        index=False,
    )


def summarize_patterns(T, M):
    early = np.nanmean(M[:, :3], axis=1)
    middle = M[:, 4]             # u=.40
    late = np.nanmean(M[:, 5:], axis=1)

    eps = 1e-12

    S = T[
        ["territory", "n_cells", "x_centroid", "y_centroid"]
    ].copy()

    S["early_mean_008_024"] = early
    S["u040"] = middle
    S["late_mean_048_056"] = late

    S["rise_early_to_u040"] = middle - early
    S["change_u040_to_late"] = late - middle
    S["late_minus_early"] = late - early

    # Descriptive trajectory classes only.
    # Threshold based on distribution of observed changes, not significance.
    rise = S["rise_early_to_u040"].to_numpy()
    post = S["change_u040_to_late"].to_numpy()

    rise_thr = np.nanmedian(np.abs(rise - np.nanmedian(rise)))
    post_thr = np.nanmedian(np.abs(post - np.nanmedian(post)))

    rise_thr = max(float(rise_thr), eps)
    post_thr = max(float(post_thr), eps)

    classes = []

    for a, b in zip(rise, post):
        if a > rise_thr and b > post_thr:
            c = "progressive_increase"
        elif a > rise_thr and b < -post_thr:
            c = "u040_enriched_then_reduced"
        elif a <= rise_thr and b > post_thr:
            c = "late_recruitment"
        elif a < -rise_thr:
            c = "early_enriched"
        else:
            c = "comparatively_stable"

        classes.append(c)

    S["descriptive_pattern"] = classes

    S.to_csv(
        OUT / "territory_trajectory_summary.csv",
        index=False,
    )

    counts = (
        S["descriptive_pattern"]
        .value_counts()
        .rename_axis("pattern")
        .reset_index(name="n_territories")
    )

    counts.to_csv(
        OUT / "trajectory_pattern_counts.csv",
        index=False,
    )

    return S, counts


def main():
    p = IN / "prcc_cell_multiscale_gw.parquet"

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)

    required = {
        "cell_index", "cell_id", "x", "y",
        *GWCOLS,
    }

    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing V7 columns: {sorted(missing)}"
        )

    print("Loaded:", p)
    print("shape:", df.shape)

    # Audit same finite-cell support.
    support = {}

    for u, col in zip(U, GWCOLS):
        n = int(np.isfinite(df[col]).sum())
        support[f"{u:.2f}"] = n
        print(f"u={u:.2f}: finite cells={n}")

    if len(set(support.values())) != 1:
        raise RuntimeError(
            "Finite-cell support differs across hierarchy scales"
        )

    print("\nConstructing spatial-only territories...")
    labels, centroids = deterministic_spatial_territories(df)

    df["territory"] = labels

    df.to_parquet(
        OUT / "prcc_cells_with_spatial_territories.parquet",
        index=False,
    )

    T = build_territory_tables(df)
    T.to_csv(
        OUT / "territory_statistics.csv",
        index=False,
    )

    print(
        "territories:",
        len(T),
        "min cells:",
        int(T["n_cells"].min()),
        "median cells:",
        float(T["n_cells"].median()),
        "max cells:",
        int(T["n_cells"].max()),
    )

    M = trajectory_matrix(T, prefix="mean")

    order = order_by_trajectory(M)

    C = plot_centered_heatmap(M, order)

    plot_absolute_heatmap(T, M, order)
    plot_territory_map(df, T, order)
    plot_u040_map(df)
    plot_global_gw()

    export_ordered_tables(T, M, C, order)

    summary, counts = summarize_patterns(T, M)

    print("\nTrajectory-pattern counts:")
    print(counts.to_string(index=False))

    # Export a compact audit package that I can inspect directly.
    provenance = {
        "version": "trajectory_maps_v8",
        "input": str(p),
        "n_cells": int(len(df)),
        "n_territories": int(N_TERRITORIES),
        "territory_definition": (
            "deterministic k-means on scaled Level-0 x,y only; "
            "no GW distortion, expression, hierarchy label, "
            "or disease molecular information used"
        ),
        "territory_ordering": (
            "average-linkage clustering of standardized seven-scale "
            "territory distortion trajectories; ordering is display-only"
        ),
        "absolute_heatmap": (
            "raw territory mean conditional GW distortion; "
            "one common color normalization across all seven scales; "
            "display clipped at global 99th percentile"
        ),
        "centered_heatmap": (
            "territory mean minus that territory's seven-scale median; "
            "visualizes within-territory reorganization only"
        ),
        "hierarchy_coordinate_warning": (
            "u is organizational scale, not biological time"
        ),
        "u_values": U.tolist(),
        "finite_cell_support": support,
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
