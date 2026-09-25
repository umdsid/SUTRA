#!/usr/bin/env python3

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter


ROOT = Path.home() / "Desktop" / "SUTRA"

IN = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "structural_trajectory_v7"
)

OUT = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "relational_flow_v9"
)

OUT.mkdir(parents=True, exist_ok=True)

U = np.array(
    [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56],
    dtype=float,
)

GWCOLS = [f"gw_u{u:.2f}" for u in U]

# Spatial field parameters.
# These affect visualization/resolution only, not the frozen GW values.
NX = 180
NY = 180

# Gaussian smoothing is performed after cell-to-grid aggregation.
SIGMA = 2.0

# Minimum smoothed support required for displaying a field.
SUPPORT_FRACTION = 0.035

# Sparse stream seed density.
STREAM_DENSITY = 1.25

# Avoid letting tiny numerical gradients determine arrow direction.
GRADIENT_DISPLAY_QUANTILE = 0.35

# Robust display clipping only.
DISPLAY_Q = 0.99

DPI = 600
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


def load_cells():
    p = IN / "prcc_cell_multiscale_gw.parquet"

    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)

    required = {"cell_index", "cell_id", "x", "y", *GWCOLS}
    missing = required - set(df.columns)

    if missing:
        raise RuntimeError(
            f"Missing required V7 columns: {sorted(missing)}"
        )

    # Exact common support across all hierarchy coordinates.
    good = (
        np.isfinite(df["x"].to_numpy(float)) &
        np.isfinite(df["y"].to_numpy(float))
    )

    for c in GWCOLS:
        good &= np.isfinite(df[c].to_numpy(float))

    out = df.loc[good].copy().reset_index(drop=True)

    print("V7 cells:", len(df))
    print("common finite-support cells:", len(out))

    return out


def grid_geometry(df):
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    # Tiny padding avoids boundary indexing ambiguity.
    xmin, xmax = np.min(x), np.max(x)
    ymin, ymax = np.min(y), np.max(y)

    px = 0.005 * max(xmax - xmin, EPS)
    py = 0.005 * max(ymax - ymin, EPS)

    xmin -= px
    xmax += px
    ymin -= py
    ymax += py

    xedges = np.linspace(xmin, xmax, NX + 1)
    yedges = np.linspace(ymin, ymax, NY + 1)

    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])

    return xedges, yedges, xc, yc


def weighted_grid(x, y, value, xedges, yedges):
    """
    Cell-level values -> regular spatial field.

    Numerator and support are smoothed separately, then divided.
    This avoids treating empty space as zero distortion.
    """
    finite = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(value)
    )

    Hn, _, _ = np.histogram2d(
        y[finite],
        x[finite],
        bins=[yedges, xedges],
        weights=value[finite],
    )

    Hw, _, _ = np.histogram2d(
        y[finite],
        x[finite],
        bins=[yedges, xedges],
    )

    num = gaussian_filter(
        Hn,
        sigma=SIGMA,
        mode="constant",
        cval=0.0,
    )

    den = gaussian_filter(
        Hw,
        sigma=SIGMA,
        mode="constant",
        cval=0.0,
    )

    field = np.full_like(num, np.nan, dtype=float)

    # Relative support criterion; no zeros inserted into empty tissue.
    threshold = SUPPORT_FRACTION * np.nanmax(den)
    mask = den >= threshold

    field[mask] = num[mask] / den[mask]

    return field, den, mask


def build_fields(df):
    x = df["x"].to_numpy(float)
    y = df["y"].to_numpy(float)

    xedges, yedges, xc, yc = grid_geometry(df)

    D = {}
    support = {}
    masks = {}

    for u, col in zip(U, GWCOLS):
        field, den, mask = weighted_grid(
            x,
            y,
            df[col].to_numpy(float),
            xedges,
            yedges,
        )

        D[u] = field
        support[u] = den
        masks[u] = mask

    return D, support, masks, xc, yc


def transition_fields(D, xc, yc):
    """
    For adjacent hierarchy coordinates:

        A = D(u_next) - D(u_now)
        V = grad_x A

    A is the scalar recruitment/relaxation field.
    V is used only for spatial streamline direction.
    """
    dx = float(np.mean(np.diff(xc)))
    dy = float(np.mean(np.diff(yc)))

    transitions = {}

    for ua, ub in zip(U[:-1], U[1:]):
        A = D[ub] - D[ua]

        finite = np.isfinite(A)

        # Gradient cannot operate sensibly through NaNs.
        # Fill outside-tissue values only for derivative computation,
        # then restore the tissue mask immediately afterward.
        tmp = A.copy()

        if np.any(finite):
            med = float(np.nanmedian(tmp))
        else:
            med = 0.0

        tmp[~finite] = med

        gy, gx = np.gradient(tmp, dy, dx)

        gx[~finite] = np.nan
        gy[~finite] = np.nan

        mag = np.sqrt(gx * gx + gy * gy)

        transitions[(ua, ub)] = {
            "delta": A,
            "vx": gx,
            "vy": gy,
            "magnitude": mag,
            "mask": finite,
        }

    return transitions


def transition_stats(transitions):
    rows = []

    for (ua, ub), q in transitions.items():
        A = q["delta"]
        M = q["magnitude"]

        ok = np.isfinite(A)
        g = np.isfinite(M)

        vals = A[ok]
        mags = M[g]

        row = {
            "u_from": ua,
            "u_to": ub,
            "n_grid_finite": int(ok.sum()),
            "mean_delta": float(np.mean(vals)),
            "median_delta": float(np.median(vals)),
            "mean_abs_delta": float(np.mean(np.abs(vals))),
            "q90_abs_delta": float(
                np.quantile(np.abs(vals), 0.90)
            ),
            "fraction_positive": float(np.mean(vals > 0)),
            "fraction_negative": float(np.mean(vals < 0)),
            "mean_gradient_magnitude": float(
                np.mean(mags)
            ),
            "q90_gradient_magnitude": float(
                np.quantile(mags, 0.90)
            ),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def plot_delta_map(A, xc, yc, stem, label):
    ok = np.isfinite(A)

    vmax = float(
        np.quantile(
            np.abs(A[ok]),
            DISPLAY_Q,
        )
    )
    vmax = max(vmax, EPS)

    fig, ax = plt.subplots(figsize=(5.4, 5.4))

    im = ax.imshow(
        A,
        origin="lower",
        extent=[xc[0], xc[-1], yc[0], yc[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
    )

    ax.set_axis_off()

    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.015,
    )
    cb.set_label(label)

    fig.tight_layout(pad=0.15)
    save(fig, stem)


def plot_stream_map(q, xc, yc, stem, delta_label):
    A = q["delta"]
    vx = q["vx"].copy()
    vy = q["vy"].copy()
    mag = q["magnitude"]

    ok = np.isfinite(A)

    vmax = float(
        np.quantile(np.abs(A[ok]), DISPLAY_Q)
    )
    vmax = max(vmax, EPS)

    # Remove weakest gradients from streamline construction.
    gm = mag[np.isfinite(mag)]

    if len(gm):
        threshold = float(
            np.quantile(
                gm,
                GRADIENT_DISPLAY_QUANTILE,
            )
        )
    else:
        threshold = np.inf

    weak = (
        ~np.isfinite(mag) |
        (mag < threshold)
    )

    vx[weak] = 0.0
    vy[weak] = 0.0

    # streamplot handles masked arrays better than NaN arrays.
    Ux = np.ma.masked_where(~ok, vx)
    Uy = np.ma.masked_where(~ok, vy)

    fig, ax = plt.subplots(figsize=(5.6, 5.6))

    im = ax.imshow(
        A,
        origin="lower",
        extent=[xc[0], xc[-1], yc[0], yc[-1]],
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        interpolation="bilinear",
        aspect="equal",
        alpha=0.90,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        ax.streamplot(
            xc,
            yc,
            Ux,
            Uy,
            density=STREAM_DENSITY,
            color="black",
            linewidth=0.55,
            arrowsize=0.65,
            minlength=0.18,
            maxlength=3.5,
            integration_direction="both",
        )

    ax.set_axis_off()

    cb = fig.colorbar(
        im,
        ax=ax,
        fraction=0.035,
        pad=0.015,
    )
    cb.set_label(delta_label)

    fig.tight_layout(pad=0.15)
    save(fig, stem)


def plot_triptych(transitions, xc, yc):
    # Early / middle / late representative adjacent transitions.
    selected = [
        (0.16, 0.24),
        (0.32, 0.40),
        (0.48, 0.56),
    ]

    all_abs = []

    for key in selected:
        A = transitions[key]["delta"]
        all_abs.append(np.abs(A[np.isfinite(A)]))

    vmax = float(
        np.quantile(
            np.concatenate(all_abs),
            DISPLAY_Q,
        )
    )
    vmax = max(vmax, EPS)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 4.2),
        constrained_layout=True,
    )

    im = None

    for ax, key in zip(axes, selected):
        ua, ub = key
        q = transitions[key]

        A = q["delta"]
        vx = q["vx"].copy()
        vy = q["vy"].copy()
        mag = q["magnitude"]

        ok = np.isfinite(A)

        gm = mag[np.isfinite(mag)]
        threshold = (
            float(
                np.quantile(
                    gm,
                    GRADIENT_DISPLAY_QUANTILE,
                )
            )
            if len(gm) else np.inf
        )

        weak = (
            ~np.isfinite(mag) |
            (mag < threshold)
        )

        vx[weak] = 0.0
        vy[weak] = 0.0

        im = ax.imshow(
            A,
            origin="lower",
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )

        Ux = np.ma.masked_where(~ok, vx)
        Uy = np.ma.masked_where(~ok, vy)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            ax.streamplot(
                xc,
                yc,
                Ux,
                Uy,
                density=1.05,
                color="black",
                linewidth=0.45,
                arrowsize=0.58,
                minlength=0.20,
                maxlength=3.0,
                integration_direction="both",
            )

        ax.set_title(
            f"$u={ua:.2f}\\rightarrow{ub:.2f}$",
            fontsize=10,
        )
        ax.set_axis_off()

    cb = fig.colorbar(
        im,
        ax=axes,
        shrink=0.74,
        pad=0.01,
    )
    cb.set_label(
        "Change in PRCC GW conditional distortion"
    )

    save(fig, "relational_flow_triptych")


def plot_all_delta_heatmaps(transitions, xc, yc):
    keys = list(transitions.keys())

    vals = []

    for key in keys:
        A = transitions[key]["delta"]
        vals.append(np.abs(A[np.isfinite(A)]))

    vmax = float(
        np.quantile(
            np.concatenate(vals),
            DISPLAY_Q,
        )
    )
    vmax = max(vmax, EPS)

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(10.8, 7.0),
        constrained_layout=True,
    )

    axes = axes.ravel()
    im = None

    for ax, key in zip(axes, keys):
        ua, ub = key
        A = transitions[key]["delta"]

        im = ax.imshow(
            A,
            origin="lower",
            extent=[xc[0], xc[-1], yc[0], yc[-1]],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )

        ax.set_title(
            f"{ua:.2f} → {ub:.2f}",
            fontsize=9,
        )
        ax.set_axis_off()

    cb = fig.colorbar(
        im,
        ax=axes.tolist(),
        shrink=0.80,
        pad=0.01,
    )

    cb.set_label(
        "Change in PRCC GW conditional distortion"
    )

    save(fig, "all_adjacent_delta_fields")


def export_fields(D, transitions, xc, yc):
    # Compact numerical cache; no need ever to rebuild the spatial field.
    arrays = {
        "u_values": U,
        "x_centers": xc,
        "y_centers": yc,
    }

    for u in U:
        arrays[f"D_u{u:.2f}"] = D[u]

    for (ua, ub), q in transitions.items():
        tag = f"{ua:.2f}_{ub:.2f}"

        arrays[f"delta_{tag}"] = q["delta"]
        arrays[f"vx_{tag}"] = q["vx"]
        arrays[f"vy_{tag}"] = q["vy"]
        arrays[f"magnitude_{tag}"] = q["magnitude"]

    np.savez_compressed(
        OUT / "relational_flow_fields.npz",
        **arrays,
    )


def main():
    df = load_cells()

    print("\nBuilding continuous spatial distortion fields...")

    D, support, masks, xc, yc = build_fields(df)

    transitions = transition_fields(D, xc, yc)

    stats = transition_stats(transitions)

    stats.to_csv(
        OUT / "transition_statistics.csv",
        index=False,
    )

    print("\nAdjacent-scale field statistics:")
    print(stats.to_string(index=False))

    export_fields(D, transitions, xc, yc)

    # All six adjacent transitions: scalar change maps.
    for key, q in transitions.items():
        ua, ub = key

        tag = f"u{ua:.2f}_to_u{ub:.2f}"

        plot_delta_map(
            q["delta"],
            xc,
            yc,
            f"delta_{tag}",
            (
                "Change in PRCC GW conditional distortion "
                f"({ua:.2f} → {ub:.2f})"
            ),
        )

        plot_stream_map(
            q,
            xc,
            yc,
            f"flow_{tag}",
            (
                "Change in PRCC GW conditional distortion "
                f"({ua:.2f} → {ub:.2f})"
            ),
        )

    plot_triptych(transitions, xc, yc)
    plot_all_delta_heatmaps(transitions, xc, yc)

    provenance = {
        "version": "relational_flow_v9",
        "input": str(
            IN / "prcc_cell_multiscale_gw.parquet"
        ),
        "n_common_cells": int(len(df)),
        "u_values": U.tolist(),
        "grid": [int(NY), int(NX)],
        "gaussian_sigma_grid_cells": float(SIGMA),
        "support_fraction": float(SUPPORT_FRACTION),
        "field_definition": (
            "D(x,u) is a support-normalized Gaussian-smoothed "
            "spatial field of inherited PRCC GW conditional "
            "distortion on common finite-support Level-0 cells."
        ),
        "change_definition": (
            "Delta_u D(x) = D(x,u_next) - D(x,u)."
        ),
        "vector_definition": (
            "v_u(x) = spatial gradient of Delta_u D(x)."
        ),
        "streamline_interpretation": (
            "Streamlines visualize the spatial direction of "
            "change in the multiscale relational-distortion field. "
            "They are not cell motion, RNA velocity, lineage, "
            "biological time, causal information flow, or disease "
            "progression."
        ),
        "hierarchy_coordinate": (
            "SUTRA organizational coordinate; not biological time."
        ),
        "outcome_used_for_smoothing": True,
        "inference": (
            "descriptive visualization of frozen GW localization; "
            "not inferential significance."
        ),
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
