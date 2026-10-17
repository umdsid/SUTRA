#!/usr/bin/env python3
"""
SUTRA cross-tissue organizational-family analysis v1.0
======================================================

PURPOSE
-------
Post-process frozen/materialized SUTRA hierarchy outputs to ask:

1. Do distinct tissues exhibit related organizational trajectory shapes?
2. Can trajectory form be separated from specimen-specific rate/amplitude?
3. Do healthy/disease pairs preserve organizational form while changing
   molecular/mechanical/communication/geometric content?

IMPORTANT
---------
* READ ONLY.
* Does NOT reconstruct SUTRA.
* Does NOT alter hierarchy thresholds, mergers, terminal states, or calibration.
* Does NOT assume a power law.
* Does NOT claim universality, an RG fixed point, or equilibrium criticality.
* With five specimens, cross-specimen comparisons are descriptive.

Outputs:
    organizational_family_v1/
        discovered_tables.csv
        selected_trajectories.csv
        normalized_trajectories.csv
        functional_distance_matrix.csv
        pairwise_functional_distances.csv
        family_model_comparison.csv
        healthy_disease_form_content.csv
        observable_inventory.csv
        analysis_manifest.json
        figures/
            SuppFig7_A_normalized_trajectories.pdf/png
            SuppFig7_B_distance_matrix.pdf/png
            SuppFig7_C_family_residuals.pdf/png
            SuppFig7_D_form_vs_content.pdf/png
            SuppFig7_organizational_family.pdf/png
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# Frozen specimen definitions
# ---------------------------------------------------------------------

SPECIMENS = {
    "healthy_reference": {
        "label": "Healthy brain",
        "organ": "brain",
        "condition": "healthy",
    },
    "alzheimers": {
        "label": "Alzheimer disease",
        "organ": "brain",
        "condition": "disease",
    },
    "gbm_reference_addon": {
        "label": "Glioblastoma",
        "organ": "brain",
        "condition": "disease",
    },
    "nondiseased_kidney": {
        "label": "Nondiseased kidney",
        "organ": "kidney",
        "condition": "healthy",
    },
    "prcc": {
        "label": "PRCC",
        "organ": "kidney",
        "condition": "disease",
    },
}

ALIASES = {
    "healthy_reference": [
        "healthy_reference", "healthy_brain", "healthy-brain",
        "brain_healthy", "healthybrain",
    ],
    "alzheimers": [
        "alzheimers", "alzheimer", "ad_brain", "ad-brain",
    ],
    "gbm_reference_addon": [
        "gbm_reference_addon", "gbm", "glioblastoma",
    ],
    "nondiseased_kidney": [
        "nondiseased_kidney", "non_diseased_kidney",
        "healthy_kidney", "normal_kidney",
    ],
    "prcc": [
        "prcc", "papillary_renal", "papillaryrenal",
    ],
}

HEALTH_DISEASE_PAIRS = [
    ("healthy_reference", "gbm_reference_addon", "Brain: healthy vs GBM"),
    ("nondiseased_kidney", "prcc", "Kidney: nondiseased vs PRCC"),
]


# ---------------------------------------------------------------------
# Candidate schema names
# ---------------------------------------------------------------------

U_CANDIDATES = [
    "u",
    "hierarchy_progress",
    "progress",
    "normalized_progress",
    "hierarchy_coordinate",
    "u_progress",
]

NODE_CANDIDATES = [
    "n_nodes",
    "node_count",
    "n_current",
    "current_node_count",
    "n_objects",
    "object_count",
]

STEP_CANDIDATES = [
    "microstep",
    "step",
    "iteration",
    "merge_step",
    "state_index",
    "level",
]

OBSERVABLE_PATTERNS = {
    "node_fraction": [
        r"node.*fraction",
        r"object.*fraction",
        r"remaining.*node",
    ],
    "interface_fraction": [
        r"admissible.*fraction",
        r"eligible.*fraction",
        r"interface.*fraction",
        r"remaining.*interface",
    ],
    "largest_mass_fraction": [
        r"largest.*mass",
        r"max.*mass.*fraction",
        r"dominant.*mass",
    ],
    "mean_merger_mass": [
        r"mean.*merge.*mass",
        r"mean.*merger.*mass",
        r"merge.*mass.*mean",
    ],
    "median_merger_mass": [
        r"median.*merge.*mass",
        r"median.*merger.*mass",
        r"merge.*mass.*median",
    ],
    "molecular": [
        r"molecular.*contrib",
        r"molecular.*fraction",
        r"molecular.*share",
        r"mol.*contrib",
    ],
    "mechanics": [
        r"mechanic.*contrib",
        r"mechanic.*fraction",
        r"mechanic.*share",
    ],
    "communication": [
        r"communication.*contrib",
        r"communication.*fraction",
        r"communication.*share",
        r"signaling.*contrib",
    ],
    "geometry": [
        r"geometry.*contrib",
        r"geometric.*contrib",
        r"geometry.*fraction",
    ],
    "topology": [
        r"topology.*contrib",
        r"topological.*contrib",
        r"topology.*fraction",
    ],
}

CHANNEL_NAMES = {
    "molecular",
    "mechanics",
    "communication",
    "geometry",
    "topology",
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 2**20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def normalize_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")


def infer_specimen(path: Path) -> Optional[str]:
    text = normalize_name(str(path))
    for specimen, aliases in ALIASES.items():
        for alias in aliases:
            if normalize_name(alias) in text:
                return specimen
    return None


def read_table(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() in {".tsv", ".txt"}:
            return pd.read_csv(path, sep="\t")
    except Exception as exc:
        print(f"[WARN] Could not read {path}: {exc}", file=sys.stderr)
    return None


def find_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    lookup = {normalize_name(c): c for c in df.columns}
    for candidate in candidates:
        key = normalize_name(candidate)
        if key in lookup:
            return lookup[key]
    return None


def find_pattern_column(
    df: pd.DataFrame,
    patterns: list[str],
) -> Optional[str]:
    for c in df.columns:
        name = normalize_name(c)
        for pattern in patterns:
            if re.search(pattern, name):
                if pd.api.types.is_numeric_dtype(df[c]):
                    return c
    return None


def robust_scale(y: np.ndarray) -> float:
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        return 1.0
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        return float(iqr)
    sd = np.std(y)
    if sd > 0:
        return float(sd)
    r = np.ptp(y)
    return float(r if r > 0 else 1.0)


def normalize_curve(y: np.ndarray) -> np.ndarray:
    """
    Remove offset and amplitude while preserving trajectory shape.
    """
    y = np.asarray(y, float)
    finite = np.isfinite(y)
    if finite.sum() < 3:
        return np.full_like(y, np.nan)

    med = np.nanmedian(y)
    scale = robust_scale(y)
    return (y - med) / scale


def interp_curve(
    u: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    u = np.asarray(u, float)
    y = np.asarray(y, float)

    ok = np.isfinite(u) & np.isfinite(y)
    u, y = u[ok], y[ok]

    if len(u) < 3:
        return np.full(len(grid), np.nan)

    order = np.argsort(u)
    u, y = u[order], y[order]

    # Average duplicate hierarchy coordinates.
    tmp = pd.DataFrame({"u": u, "y": y}).groupby("u", as_index=False).mean()
    u = tmp["u"].to_numpy()
    y = tmp["y"].to_numpy()

    if len(u) < 3:
        return np.full(len(grid), np.nan)

    out = np.full(len(grid), np.nan)
    inside = (grid >= u.min()) & (grid <= u.max())
    out[inside] = np.interp(grid[inside], u, y)
    return out


def functional_distance(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)))


# ---------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------

@dataclass
class TableRecord:
    specimen: str
    path: Path
    nrows: int
    ncols: int
    u_col: Optional[str]
    node_col: Optional[str]
    step_col: Optional[str]
    sha256: str


def discover_tables(root: Path) -> tuple[list[TableRecord], dict[Path, pd.DataFrame]]:
    records = []
    loaded = {}

    paths = (
        list(root.rglob("*.parquet"))
        + list(root.rglob("*.csv"))
        + list(root.rglob("*.tsv"))
    )

    print(f"[INFO] Found {len(paths)} candidate tables")

    for i, path in enumerate(sorted(set(paths)), 1):
        specimen = infer_specimen(path)
        if specimen is None:
            continue

        df = read_table(path)
        if df is None or df.empty:
            continue

        u_col = find_column(df, U_CANDIDATES)
        node_col = find_column(df, NODE_CANDIDATES)
        step_col = find_column(df, STEP_CANDIDATES)

        records.append(
            TableRecord(
                specimen=specimen,
                path=path,
                nrows=len(df),
                ncols=len(df.columns),
                u_col=u_col,
                node_col=node_col,
                step_col=step_col,
                sha256=sha256_file(path),
            )
        )
        loaded[path] = df

        if i % 100 == 0:
            print(f"[INFO] scanned {i}/{len(paths)}")

    return records, loaded


# ---------------------------------------------------------------------
# Hierarchy coordinate
# ---------------------------------------------------------------------

def construct_u(
    df: pd.DataFrame,
    u_col: Optional[str],
    node_col: Optional[str],
    step_col: Optional[str],
) -> Optional[np.ndarray]:

    if u_col is not None:
        u = pd.to_numeric(df[u_col], errors="coerce").to_numpy(float)
        finite = u[np.isfinite(u)]
        if len(finite) >= 3:
            # If already 0..1, preserve exactly.
            if finite.min() >= -1e-8 and finite.max() <= 1 + 1e-8:
                return u

            # Otherwise normalize only if clearly a progress coordinate.
            lo, hi = np.nanmin(u), np.nanmax(u)
            if hi > lo:
                return (u - lo) / (hi - lo)

    if node_col is not None:
        n = pd.to_numeric(df[node_col], errors="coerce").to_numpy(float)
        finite = n[np.isfinite(n)]
        if len(finite) >= 3:
            n0 = finite[0]
            nt = finite[-1]
            denom = n0 - nt
            if abs(denom) > 0:
                return (n0 - n) / denom

    if step_col is not None:
        s = pd.to_numeric(df[step_col], errors="coerce").to_numpy(float)
        finite = s[np.isfinite(s)]
        if len(finite) >= 3:
            lo, hi = np.nanmin(s), np.nanmax(s)
            if hi > lo:
                return (s - lo) / (hi - lo)

    return None


# ---------------------------------------------------------------------
# Extract observables
# ---------------------------------------------------------------------

def extract_candidates(
    records: list[TableRecord],
    loaded: dict[Path, pd.DataFrame],
) -> pd.DataFrame:

    rows = []

    for rec in records:
        df = loaded[rec.path]
        u = construct_u(df, rec.u_col, rec.node_col, rec.step_col)

        if u is None:
            continue

        # Explicit node fraction if node count exists.
        if rec.node_col is not None:
            n = pd.to_numeric(df[rec.node_col], errors="coerce").to_numpy(float)
            if np.isfinite(n).sum() >= 3:
                n0 = n[np.isfinite(n)][0]
                if n0 != 0:
                    for ui, yi in zip(u, n / n0):
                        rows.append({
                            "specimen": rec.specimen,
                            "observable": "node_fraction",
                            "u": ui,
                            "value": yi,
                            "source": str(rec.path),
                            "source_column": rec.node_col,
                        })

        # Pattern-derived observables.
        for obs, patterns in OBSERVABLE_PATTERNS.items():
            col = find_pattern_column(df, patterns)
            if col is None:
                continue

            y = pd.to_numeric(df[col], errors="coerce").to_numpy(float)

            if np.isfinite(y).sum() < 3:
                continue

            for ui, yi in zip(u, y):
                rows.append({
                    "specimen": rec.specimen,
                    "observable": obs,
                    "u": ui,
                    "value": yi,
                    "source": str(rec.path),
                    "source_column": col,
                })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out = out[np.isfinite(out["u"]) & np.isfinite(out["value"])]
    out = out[(out["u"] >= -1e-6) & (out["u"] <= 1 + 1e-6)]

    return out


def choose_best_source(candidates: pd.DataFrame) -> pd.DataFrame:
    """
    If the same observable is discovered in multiple tables, choose the
    source with the greatest number of distinct hierarchy coordinates.

    No values are silently averaged across different source tables.
    """
    if candidates.empty:
        return candidates

    counts = (
        candidates
        .groupby(["specimen", "observable", "source", "source_column"])
        ["u"].nunique()
        .reset_index(name="n_u")
    )

    chosen = (
        counts
        .sort_values(
            ["specimen", "observable", "n_u"],
            ascending=[True, True, False]
        )
        .drop_duplicates(["specimen", "observable"])
    )

    keys = chosen[
        ["specimen", "observable", "source", "source_column"]
    ].copy()

    out = candidates.merge(
        keys,
        on=["specimen", "observable", "source", "source_column"],
        how="inner",
    )

    return out


# ---------------------------------------------------------------------
# Common-grid trajectories
# ---------------------------------------------------------------------

def build_common_grid(
    selected: pd.DataFrame,
    n_grid: int = 201,
) -> pd.DataFrame:

    grid = np.linspace(0.0, 1.0, n_grid)
    rows = []

    for (specimen, observable), g in selected.groupby(
        ["specimen", "observable"]
    ):
        u = g["u"].to_numpy(float)
        y = g["value"].to_numpy(float)

        curve = interp_curve(u, y, grid)
        z = normalize_curve(curve)

        for ui, raw, norm in zip(grid, curve, z):
            rows.append({
                "specimen": specimen,
                "observable": observable,
                "u": ui,
                "value_interp": raw,
                "value_shape": norm,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Functional distance
# ---------------------------------------------------------------------

def compute_distances(curves: pd.DataFrame):
    specimens = [s for s in SPECIMENS if s in curves["specimen"].unique()]
    observables = sorted(curves["observable"].unique())

    # Require an observable to exist for >= 3 specimens.
    usable_obs = []
    for obs in observables:
        n = curves[curves.observable == obs]["specimen"].nunique()
        if n >= 3:
            usable_obs.append(obs)

    pair_rows = []

    for i, a in enumerate(specimens):
        for b in specimens[i + 1:]:
            ds = []

            for obs in usable_obs:
                ga = curves[
                    (curves.specimen == a) &
                    (curves.observable == obs)
                ].sort_values("u")
                gb = curves[
                    (curves.specimen == b) &
                    (curves.observable == obs)
                ].sort_values("u")

                if ga.empty or gb.empty:
                    continue

                d = functional_distance(
                    ga["value_shape"].to_numpy(),
                    gb["value_shape"].to_numpy(),
                )

                if np.isfinite(d):
                    ds.append(d)
                    pair_rows.append({
                        "specimen_a": a,
                        "specimen_b": b,
                        "observable": obs,
                        "distance": d,
                    })

            if ds:
                pair_rows.append({
                    "specimen_a": a,
                    "specimen_b": b,
                    "observable": "__aggregate__",
                    "distance": float(np.sqrt(np.mean(np.square(ds)))),
                })

    pairwise = pd.DataFrame(pair_rows)

    matrix = pd.DataFrame(
        np.nan,
        index=specimens,
        columns=specimens,
        dtype=float,
    )

    for s in specimens:
        matrix.loc[s, s] = 0.0

    agg = pairwise[pairwise.observable == "__aggregate__"]

    for _, row in agg.iterrows():
        a, b = row.specimen_a, row.specimen_b
        matrix.loc[a, b] = row.distance
        matrix.loc[b, a] = row.distance

    return pairwise, matrix, usable_obs


# ---------------------------------------------------------------------
# Shared-family model
# ---------------------------------------------------------------------

def shared_family_analysis(curves: pd.DataFrame) -> pd.DataFrame:
    """
    Nonparametric shared-family test.

    For each observable:
      1. use shape-normalized specimen trajectories;
      2. construct leave-one-specimen-out family mean;
      3. compute held-out RMSE;
      4. compare with a specimen-specific smooth baseline estimated by
         leave-every-fifth-point-out interpolation.

    This is intentionally descriptive with n=5.
    """

    rows = []

    for obs, gobs in curves.groupby("observable"):
        specimens = sorted(gobs["specimen"].unique())

        if len(specimens) < 3:
            continue

        pivot = gobs.pivot(
            index="u",
            columns="specimen",
            values="value_shape",
        ).sort_index()

        for specimen in specimens:
            y = pivot[specimen].to_numpy(float)

            others = [
                c for c in pivot.columns
                if c != specimen
            ]

            family = np.nanmean(
                pivot[others].to_numpy(float),
                axis=1,
            )

            shared_rmse = functional_distance(y, family)

            # Specimen-specific interpolation baseline.
            # Hold out every fifth finite point.
            u = pivot.index.to_numpy(float)
            finite = np.isfinite(y)
            idx = np.where(finite)[0]

            specific_rmse = np.nan

            if len(idx) >= 20:
                test_idx = idx[::5]
                train_mask = finite.copy()
                train_mask[test_idx] = False

                if train_mask.sum() >= 3:
                    pred = np.interp(
                        u[test_idx],
                        u[train_mask],
                        y[train_mask],
                    )
                    specific_rmse = float(
                        np.sqrt(
                            np.mean(
                                (pred - y[test_idx]) ** 2
                            )
                        )
                    )

            rows.append({
                "observable": obs,
                "specimen": specimen,
                "shared_family_LOSO_RMSE": shared_rmse,
                "specimen_specific_interpolation_RMSE": specific_rmse,
                "shared_over_specific": (
                    shared_rmse / specific_rmse
                    if np.isfinite(shared_rmse)
                    and np.isfinite(specific_rmse)
                    and specific_rmse > 0
                    else np.nan
                ),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Healthy-disease form vs content
# ---------------------------------------------------------------------

def form_content_analysis(
    curves: pd.DataFrame,
    usable_obs: list[str],
) -> pd.DataFrame:

    rows = []

    structural = [
        o for o in usable_obs
        if o not in CHANNEL_NAMES
    ]

    channels = [
        o for o in usable_obs
        if o in CHANNEL_NAMES
    ]

    for healthy, disease, label in HEALTH_DISEASE_PAIRS:
        if healthy not in curves.specimen.unique():
            continue
        if disease not in curves.specimen.unique():
            continue

        form_ds = []
        content_ds = []

        for obs in structural:
            a = curves[
                (curves.specimen == healthy) &
                (curves.observable == obs)
            ].sort_values("u")
            b = curves[
                (curves.specimen == disease) &
                (curves.observable == obs)
            ].sort_values("u")

            if a.empty or b.empty:
                continue

            d = functional_distance(
                a.value_shape.to_numpy(),
                b.value_shape.to_numpy(),
            )

            if np.isfinite(d):
                form_ds.append(d)

        for obs in channels:
            a = curves[
                (curves.specimen == healthy) &
                (curves.observable == obs)
            ].sort_values("u")
            b = curves[
                (curves.specimen == disease) &
                (curves.observable == obs)
            ].sort_values("u")

            if a.empty or b.empty:
                continue

            # For biological content we retain the interpolated values,
            # but robust-standardize jointly so units do not dominate.
            ya = a.value_interp.to_numpy(float)
            yb = b.value_interp.to_numpy(float)

            joint = np.concatenate([ya, yb])
            scale = robust_scale(joint)
            center = np.nanmedian(joint)

            d = functional_distance(
                (ya - center) / scale,
                (yb - center) / scale,
            )

            if np.isfinite(d):
                content_ds.append(d)

        form = (
            float(np.sqrt(np.mean(np.square(form_ds))))
            if form_ds else np.nan
        )

        content = (
            float(np.sqrt(np.mean(np.square(content_ds))))
            if content_ds else np.nan
        )

        rows.append({
            "comparison": label,
            "healthy": healthy,
            "disease": disease,
            "form_distance": form,
            "content_distance": content,
            "n_form_observables": len(form_ds),
            "n_content_channels": len(content_ds),
            "content_to_form_ratio": (
                content / form
                if np.isfinite(content)
                and np.isfinite(form)
                and form > 0
                else np.nan
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------

def specimen_label(s):
    return SPECIMENS.get(s, {}).get("label", s)


def plot_trajectories(
    curves: pd.DataFrame,
    usable_obs: list[str],
    outdir: Path,
):
    """
    Separate brain and kidney panels.
    """
    preferred = [
        "node_fraction",
        "interface_fraction",
        "largest_mass_fraction",
        "mean_merger_mass",
    ]

    obs = next(
        (o for o in preferred if o in usable_obs),
        usable_obs[0] if usable_obs else None,
    )

    if obs is None:
        return

    fig, axes = plt.subplots(
        1, 2,
        figsize=(10.5, 4.1),
        constrained_layout=True,
    )

    for ax, organ in zip(axes, ["brain", "kidney"]):
        ss = [
            s for s, meta in SPECIMENS.items()
            if meta["organ"] == organ
            and s in curves.specimen.unique()
        ]

        for s in ss:
            g = curves[
                (curves.specimen == s) &
                (curves.observable == obs)
            ].sort_values("u")

            if not g.empty:
                ax.plot(
                    g.u,
                    g.value_shape,
                    lw=1.8,
                    label=specimen_label(s),
                )

        ax.set_xlabel("Normalized hierarchy progress, $u$")
        ax.set_ylabel(f"Shape-normalized {obs.replace('_', ' ')}")
        ax.set_title(organ.capitalize())
        ax.legend(frameon=False, fontsize=8)

    fig.savefig(
        outdir / "SuppFig7_A_normalized_trajectories.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "SuppFig7_A_normalized_trajectories.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_distance_matrix(matrix: pd.DataFrame, outdir: Path):
    if matrix.empty:
        return

    fig, ax = plt.subplots(figsize=(6.2, 5.3))
    im = ax.imshow(matrix.to_numpy(float), aspect="equal")

    labels = [specimen_label(x) for x in matrix.index]

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(labels)):
        for j in range(len(labels)):
            x = matrix.iloc[i, j]
            if np.isfinite(x):
                ax.text(
                    j, i, f"{x:.2f}",
                    ha="center", va="center",
                    fontsize=7,
                )

    ax.set_title("Organizational trajectory distance")
    fig.colorbar(im, ax=ax, label="Functional distance")

    fig.savefig(
        outdir / "SuppFig7_B_distance_matrix.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "SuppFig7_B_distance_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_family_residuals(family: pd.DataFrame, outdir: Path):
    if family.empty:
        return

    summary = (
        family.groupby("observable")
        ["shared_family_LOSO_RMSE"]
        .median()
        .sort_values()
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.bar(
        np.arange(len(summary)),
        summary.values,
    )
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(
        [x.replace("_", " ") for x in summary.index],
        rotation=45,
        ha="right",
        fontsize=8,
    )
    ax.set_ylabel("Median held-out family RMSE")
    ax.set_title("Shared-family reconstruction error")

    fig.savefig(
        outdir / "SuppFig7_C_family_residuals.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "SuppFig7_C_family_residuals.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_form_content(fc: pd.DataFrame, outdir: Path):
    if fc.empty:
        return

    x = np.arange(len(fc))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7.0, 4.4))

    ax.bar(
        x - width / 2,
        fc.form_distance,
        width,
        label="Organizational form",
    )

    ax.bar(
        x + width / 2,
        fc.content_distance,
        width,
        label="Biological content",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(fc.comparison, fontsize=9)
    ax.set_ylabel("Functional distance")
    ax.set_title("Healthy–disease: form versus content")
    ax.legend(frameon=False)

    fig.savefig(
        outdir / "SuppFig7_D_form_vs_content.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "SuppFig7_D_form_vs_content.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def compose_summary(
    curves: pd.DataFrame,
    matrix: pd.DataFrame,
    family: pd.DataFrame,
    fc: pd.DataFrame,
    usable_obs: list[str],
    outdir: Path,
):
    fig, axes = plt.subplots(
        2, 2,
        figsize=(11, 8.5),
        constrained_layout=True,
    )

    # A
    ax = axes[0, 0]
    preferred = [
        "node_fraction",
        "interface_fraction",
        "largest_mass_fraction",
    ]
    obs = next(
        (o for o in preferred if o in usable_obs),
        usable_obs[0] if usable_obs else None,
    )

    if obs:
        for s in SPECIMENS:
            g = curves[
                (curves.specimen == s) &
                (curves.observable == obs)
            ].sort_values("u")
            if not g.empty:
                ax.plot(g.u, g.value_shape, lw=1.5, label=specimen_label(s))

        ax.set_xlabel("$u$")
        ax.set_ylabel("Shape-normalized trajectory")
        ax.set_title(f"A  {obs.replace('_', ' ')}")
        ax.legend(frameon=False, fontsize=6)

    # B
    ax = axes[0, 1]
    if not matrix.empty:
        im = ax.imshow(matrix.to_numpy(float), aspect="equal")
        labs = [specimen_label(x) for x in matrix.index]
        ax.set_xticks(range(len(labs)))
        ax.set_yticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=45, ha="right", fontsize=6)
        ax.set_yticklabels(labs, fontsize=6)
        ax.set_title("B  Organizational distance")
        fig.colorbar(im, ax=ax, fraction=0.046)

    # C
    ax = axes[1, 0]
    if not family.empty:
        summ = (
            family.groupby("observable")
            ["shared_family_LOSO_RMSE"]
            .median()
            .sort_values()
        )
        ax.bar(np.arange(len(summ)), summ.values)
        ax.set_xticks(np.arange(len(summ)))
        ax.set_xticklabels(
            [x.replace("_", " ") for x in summ.index],
            rotation=45,
            ha="right",
            fontsize=6,
        )
        ax.set_ylabel("Median LOSO RMSE")
        ax.set_title("C  Shared-family residual")

    # D
    ax = axes[1, 1]
    if not fc.empty:
        x = np.arange(len(fc))
        w = 0.35
        ax.bar(x - w/2, fc.form_distance, w, label="Form")
        ax.bar(x + w/2, fc.content_distance, w, label="Content")
        ax.set_xticks(x)
        ax.set_xticklabels(fc.comparison, fontsize=7)
        ax.set_ylabel("Functional distance")
        ax.set_title("D  Healthy–disease decomposition")
        ax.legend(frameon=False, fontsize=7)

    fig.savefig(
        outdir / "SuppFig7_organizational_family.pdf",
        bbox_inches="tight",
    )
    fig.savefig(
        outdir / "SuppFig7_organizational_family.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "root",
        type=Path,
        help="Frozen SUTRA results root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("organizational_family_v1"),
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=201,
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    out = args.out.expanduser().resolve()
    figdir = out / "figures"

    out.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(f"Results root does not exist: {root}")

    print("=" * 72)
    print("SUTRA ORGANIZATIONAL-FAMILY ANALYSIS v1.0")
    print("=" * 72)
    print(f"Frozen root : {root}")
    print(f"Output      : {out}")
    print()

    records, loaded = discover_tables(root)

    discovery = pd.DataFrame([
        {
            "specimen": r.specimen,
            "path": str(r.path),
            "nrows": r.nrows,
            "ncols": r.ncols,
            "u_col": r.u_col,
            "node_col": r.node_col,
            "step_col": r.step_col,
            "sha256": r.sha256,
        }
        for r in records
    ])

    discovery.to_csv(out / "discovered_tables.csv", index=False)

    if discovery.empty:
        raise SystemExit(
            "No specimen-resolved CSV/Parquet tables were discovered."
        )

    print("\n[INFO] Tables by specimen:")
    print(discovery.groupby("specimen").size().to_string())

    candidates = extract_candidates(records, loaded)

    if candidates.empty:
        raise SystemExit(
            "\nNo trajectory observables were recognized automatically.\n"
            "See discovered_tables.csv. The script has intentionally stopped "
            "rather than guessing the schema."
        )

    inventory = (
        candidates.groupby(
            ["specimen", "observable", "source", "source_column"]
        )
        .agg(
            n=("value", "size"),
            n_u=("u", "nunique"),
            u_min=("u", "min"),
            u_max=("u", "max"),
        )
        .reset_index()
    )
    inventory.to_csv(out / "observable_inventory.csv", index=False)

    selected = choose_best_source(candidates)
    selected.to_csv(out / "selected_trajectories.csv", index=False)

    print("\n[INFO] Selected observables:")
    print(
        selected.groupby(["specimen", "observable"])
        .size()
        .unstack(fill_value=0)
        .to_string()
    )

    curves = build_common_grid(selected, n_grid=args.grid)
    curves.to_csv(out / "normalized_trajectories.csv", index=False)

    pairwise, matrix, usable_obs = compute_distances(curves)

    pairwise.to_csv(
        out / "pairwise_functional_distances.csv",
        index=False,
    )
    matrix.to_csv(out / "functional_distance_matrix.csv")

    family = shared_family_analysis(curves)
    family.to_csv(
        out / "family_model_comparison.csv",
        index=False,
    )

    fc = form_content_analysis(curves, usable_obs)
    fc.to_csv(
        out / "healthy_disease_form_content.csv",
        index=False,
    )

    plot_trajectories(curves, usable_obs, figdir)
    plot_distance_matrix(matrix, figdir)
    plot_family_residuals(family, figdir)
    plot_form_content(fc, figdir)
    compose_summary(
        curves,
        matrix,
        family,
        fc,
        usable_obs,
        figdir,
    )

    manifest = {
        "analysis": "SUTRA organizational-family analysis",
        "version": "1.0",
        "root": str(root),
        "n_discovered_tables": int(len(discovery)),
        "specimens_found": sorted(
            discovery["specimen"].unique().tolist()
        ),
        "usable_observables": usable_obs,
        "grid_points": args.grid,
        "claims_boundary": {
            "descriptive_cross_specimen_analysis": True,
            "universal_scaling_law_claimed": False,
            "universal_exponent_claimed": False,
            "equilibrium_criticality_claimed": False,
            "rg_fixed_point_claimed": False,
            "hierarchy_reconstructed": False,
        },
        "health_disease_pairs": [
            {
                "healthy": a,
                "disease": b,
                "label": c,
            }
            for a, b, c in HEALTH_DISEASE_PAIRS
        ],
    }

    with (out / "analysis_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 72)
    print("RESULTS")
    print("=" * 72)

    if not matrix.empty:
        print("\nAggregate organizational distance:")
        print(matrix.round(4).to_string())

    if not family.empty:
        print("\nShared-family held-out error:")
        print(
            family.groupby("observable")
            ["shared_family_LOSO_RMSE"]
            .agg(["median", "max", "count"])
            .round(4)
            .to_string()
        )

    if not fc.empty:
        print("\nHealthy–disease form/content comparison:")
        print(fc.round(4).to_string(index=False))

    print("\n[DONE]")
    print(f"Results: {out}")
    print(f"Figure : {figdir / 'SuppFig7_organizational_family.pdf'}")
    print()
    print(
        "Interpretation must be based on the numerical output. "
        "Do not infer a shared law merely because curves appear similar."
    )


if __name__ == "__main__":
    main()
