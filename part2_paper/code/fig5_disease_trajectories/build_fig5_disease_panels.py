#!/usr/bin/env python3

"""
SUTRA Figure 5
Disease trajectory preflight + independent panel materialization.

ROOT
----
~/Desktop/SUTRA

OUTPUT
------
~/Desktop/SUTRA/results/Fig5_Disease_Trajectories

SCIENTIFIC QUESTION
-------------------
How does pathology deform multiscale tissue organization relative
to the corresponding nondiseased reference?

RULES
-----
1. Brain and kidney remain separate.
2. Existing materialized/cached outputs only.
3. Never rerun hierarchy construction.
4. Never touch raw SUTRA data.
5. Never infer a disease diagnosis merely because a table differs
   from the reference.
6. Candidate classification from names is used for discovery only.
7. Emit audit tables whenever a final panel cannot yet be justified.
"""

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATHS
# ============================================================

HOME = Path.home()

SUTRA_ROOT = HOME / "Desktop" / "SUTRA"
RESULTS_ROOT = SUTRA_ROOT / "results"

OUT_ROOT = RESULTS_ROOT / "Fig5_Disease_Trajectories"

PANELS = OUT_ROOT / "panels"
AUDIT = OUT_ROOT / "audit"
TABLES = OUT_ROOT / "tables"
MANIFEST = OUT_ROOT / "manifest"

for d in [
    PANELS,
    AUDIT,
    TABLES,
    MANIFEST,
]:
    d.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# KNOWN REFERENCE SPECIMENS
# ============================================================

REFERENCE_NAMES = {
    "brain": "healthy_reference",
    "kidney": "nondiseased_kidney",
}


REFERENCE_TERMS = {
    "healthy",
    "healthy_reference",
    "reference",
    "nondiseased",
    "nondiseased_kidney",
    "non_diseased",
    "control",
}


DISEASE_TERMS = {
    "disease",
    "diseased",
    "pathology",
    "pathological",
    "tumor",
    "tumour",
    "cancer",
    "gbm",
    "glioma",
    "fibrosis",
    "fibrotic",
    "ckd",
    "aki",
    "injury",
}


# ============================================================
# PLOT STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def clean_axes(ax):

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        labelsize=9,
    )


def save_panel(fig, stem):

    png = PANELS / f"{stem}.png"
    pdf = PANELS / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print("BUILT:", png.name)


# ============================================================
# STRING / DISCOVERY UTILITIES
# ============================================================

def norm_name(x):

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(x).lower(),
    ).strip("_")


def classify_organ(text):

    n = norm_name(text)

    brain_terms = (
        "brain",
        "gbm",
        "glioma",
        "cortex",
        "cortical",
    )

    kidney_terms = (
        "kidney",
        "renal",
        "ckd",
        "aki",
    )

    if any(
        x in n
        for x in brain_terms
    ):
        return "brain"

    if any(
        x in n
        for x in kidney_terms
    ):
        return "kidney"

    return None


def is_reference_like(text):

    n = norm_name(text)

    return any(
        x in n
        for x in REFERENCE_TERMS
    )


def is_disease_like(text):

    n = norm_name(text)

    return any(
        x in n
        for x in DISEASE_TERMS
    )


# ============================================================
# CONTROLLED RESULTS-TREE WALK
# ============================================================

def controlled_walk(
    root,
    max_depth=5,
):

    """
    Walk only the SUTRA results tree.

    Explicitly excludes:
        results/Fig5_Disease_Trajectories

    This is NOT a global filesystem scan.
    """

    root = root.resolve()
    excluded = OUT_ROOT.resolve()

    files = []

    frontier = [
        (root, 0)
    ]

    while frontier:

        directory, depth = frontier.pop()

        if depth > max_depth:
            continue

        try:
            entries = list(
                directory.iterdir()
            )

        except Exception:
            continue

        for p in entries:

            try:
                rp = p.resolve()
            except Exception:
                continue

            if (
                rp == excluded
                or excluded in rp.parents
            ):
                continue

            if p.is_file():

                files.append(p)

            elif p.is_dir():

                frontier.append(
                    (p, depth + 1)
                )

    return files


# ============================================================
# INVENTORY
# ============================================================

print()
print("=" * 76)
print("SUTRA FIG. 5 — RESULTS INVENTORY")
print("=" * 76)

all_files = controlled_walk(
    RESULTS_ROOT,
    max_depth=5,
)

print(
    "Files inspected inside results:",
    len(all_files),
)


records = []

for p in all_files:

    n = norm_name(p)

    interesting = any(
        token in n
        for token in [
            "trajectory",
            "hierarchy",
            "collective",
            "disease",
            "pathology",
            "cell",
            "spatial",
            "expression",
            "molecular",
            "program",
            "gene",
            "scale",
            "level",
        ]
    )

    if not interesting:
        continue

    records.append({
        "path": str(p),
        "relative_path":
            str(
                p.relative_to(
                    RESULTS_ROOT
                )
            ),
        "filename": p.name,
        "parent": p.parent.name,
        "suffix": p.suffix.lower(),
        "organ_guess":
            classify_organ(str(p)),
        "reference_like":
            is_reference_like(str(p)),
        "disease_like":
            is_disease_like(str(p)),
    })


inventory = pd.DataFrame(
    records
)


inventory.to_csv(
    AUDIT /
    "fig5_results_inventory.csv",
    index=False,
)


print(
    "Potentially relevant files:",
    len(inventory),
)


# ============================================================
# TABLE READER
# ============================================================

def read_table(path):

    path = Path(path)

    try:

        suffix = path.suffix.lower()

        if suffix == ".csv":

            return pd.read_csv(
                path
            )

        if suffix == ".tsv":

            return pd.read_csv(
                path,
                sep="\t",
            )

        if suffix in {
            ".parquet",
            ".pq",
        }:

            return pd.read_parquet(
                path
            )

    except Exception as e:

        warnings.warn(
            f"Could not read "
            f"{path}: {e}"
        )

    return None


# ============================================================
# HIERARCHY COLUMN DETECTION
# ============================================================

def detect_u(df):

    low = {
        c.lower(): c
        for c in df.columns
    }

    exact = [
        "u",
        "removed_fraction",
        "hierarchy_progression",
        "fraction_removed",
    ]

    for x in exact:

        if x in low:
            return low[x]

    for c in df.columns:

        lc = c.lower()

        if (
            "hierarchy_progress" in lc
            or "removed_fraction" in lc
        ):
            return c

    return None


def detect_metrics(df):

    metrics = []

    exact = [
        "phi",
        "m1_fraction",
        "chi",
        "breadth",
        "gini",
        "top1_mass_fraction",
        "m1",
        "m2",
        "m3",
        "n_components",
        "n_nodes",
    ]

    low = {
        c.lower(): c
        for c in df.columns
    }

    for x in exact:

        if x in low:
            metrics.append(
                low[x]
            )

    for c in df.columns:

        if c in metrics:
            continue

        if not pd.api.types.is_numeric_dtype(
            df[c]
        ):
            continue

        lc = c.lower()

        if any(
            x in lc
            for x in [
                "fraction",
                "gini",
                "mass",
                "breadth",
                "collective",
            ]
        ):

            metrics.append(c)

    return metrics


# ============================================================
# TRAJECTORY TABLE DISCOVERY
# ============================================================

trajectory_tables = []

if len(inventory):

    for _, row in inventory.iterrows():

        p = Path(
            row["path"]
        )

        if p.suffix.lower() not in {
            ".csv",
            ".tsv",
            ".parquet",
            ".pq",
        }:
            continue

        n = norm_name(p)

        if not any(
            x in n
            for x in [
                "trajectory",
                "hierarchy",
                "collective",
            ]
        ):
            continue

        d = read_table(p)

        if d is None:
            continue

        if len(d) < 2:
            continue

        u = detect_u(d)

        metrics = detect_metrics(d)

        if u is None:
            continue

        if not metrics:
            continue

        trajectory_tables.append({
            "path": p,
            "data": d,
            "u": u,
            "metrics": metrics,
            "organ":
                row["organ_guess"],
            "reference_like":
                bool(
                    row[
                        "reference_like"
                    ]
                ),
            "disease_like":
                bool(
                    row[
                        "disease_like"
                    ]
                ),
        })


trajectory_inventory = []

for x in trajectory_tables:

    trajectory_inventory.append({
        "path": str(x["path"]),
        "organ": x["organ"],
        "reference_like":
            x["reference_like"],
        "disease_like":
            x["disease_like"],
        "u_column": x["u"],
        "metrics":
            "|".join(
                x["metrics"]
            ),
        "n_rows":
            len(x["data"]),
    })


pd.DataFrame(
    trajectory_inventory
).to_csv(
    AUDIT /
    "fig5_trajectory_tables.csv",
    index=False,
)


print(
    "Readable trajectory tables:",
    len(trajectory_tables),
)


# ============================================================
# SAMPLE LABEL
# ============================================================

def sample_label(path):

    path = Path(path)

    # Prefer immediate parent because
    # specimen-level outputs are usually
    # stored beneath specimen directories.
    label = path.parent.name

    if label.lower() in {
        "results",
        "tables",
        "trajectory",
        "trajectories",
    }:

        label = path.stem

    return label


# ============================================================
# ORGAN TABLE SELECTION
# ============================================================

def organ_tables(organ):

    return [
        x
        for x in trajectory_tables
        if x["organ"] == organ
    ]


# ============================================================
# CHOOSE COMMON METRIC
# ============================================================

def choose_metric(items):

    if not items:
        return None

    shared = set(
        items[0]["metrics"]
    )

    for x in items[1:]:

        shared &= set(
            x["metrics"]
        )

    preferred = [
        "phi",
        "m1_fraction",
        "top1_mass_fraction",
        "gini",
        "breadth",
        "chi",
    ]

    lower_map = {
        x.lower(): x
        for x in shared
    }

    for p in preferred:

        if p in lower_map:
            return lower_map[p]

    if shared:

        return sorted(
            shared
        )[0]

    return None


# ============================================================
# A/B — MULTISCALE DISEASE TRAJECTORIES
# ============================================================

built = []


for organ, panel_letter in [
    ("brain", "A"),
    ("kidney", "B"),
]:

    items = organ_tables(
        organ
    )

    refs = [
        x
        for x in items
        if x["reference_like"]
    ]

    disease = [
        x
        for x in items
        if x["disease_like"]
    ]

    comparison = refs + disease

    metric = choose_metric(
        comparison
    )

    if (
        not refs
        or not disease
        or metric is None
    ):

        print()
        print(
            f"SKIP {panel_letter} "
            f"{organ} trajectory"
        )

        print(
            " references:",
            len(refs),
            " disease-like:",
            len(disease),
            " metric:",
            metric,
        )

        continue

    fig, ax = plt.subplots(
        figsize=(5.4, 4.4)
    )

    for x in comparison:

        d = x["data"]

        u = x["u"]

        z = d[
            [u, metric]
        ].copy()

        z = z[
            np.isfinite(z[u])
            & np.isfinite(
                z[metric]
            )
        ]

        if len(z) < 2:
            continue

        if x["reference_like"]:

            lw = 2.8
            alpha = 1.0
            label = (
                "Reference: "
                + sample_label(
                    x["path"]
                )
            )

        else:

            lw = 1.8
            alpha = 0.85
            label = (
                "Disease: "
                + sample_label(
                    x["path"]
                )
            )

        ax.plot(
            z[u],
            z[metric],
            lw=lw,
            alpha=alpha,
            label=label,
        )

    ax.set_xlabel(
        "Hierarchy progression, $u$"
    )

    ax.set_ylabel(metric)

    ax.set_title(
        "Brain"
        if organ == "brain"
        else "Kidney",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        fontsize=7,
    )

    clean_axes(ax)

    fig.tight_layout()

    stem = (
        f"{panel_letter}_"
        f"{organ}_disease_trajectory"
    )

    save_panel(
        fig,
        stem,
    )

    built.append(stem)


# ============================================================
# INTERPOLATION
# ============================================================

def interpolate(
    df,
    ucol,
    metric,
    grid,
):

    x = df[
        ucol
    ].to_numpy(float)

    y = df[
        metric
    ].to_numpy(float)

    ok = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[ok]
    y = y[ok]

    if len(x) < 2:
        return None

    order = np.argsort(x)

    x = x[order]
    y = y[order]

    x, idx = np.unique(
        x,
        return_index=True,
    )

    y = y[idx]

    if len(x) < 2:
        return None

    return np.interp(
        grid,
        x,
        y,
        left=np.nan,
        right=np.nan,
    )


# ============================================================
# C/D — REFERENCE-RELATIVE DISEASE DIVERGENCE
# ============================================================

peak_rows = []


for organ, panel_letter in [
    ("brain", "C"),
    ("kidney", "D"),
]:

    items = organ_tables(
        organ
    )

    refs = [
        x
        for x in items
        if x["reference_like"]
    ]

    disease = [
        x
        for x in items
        if x["disease_like"]
    ]

    if not refs or not disease:

        print()
        print(
            f"SKIP {panel_letter} "
            f"{organ} divergence:"
            " missing reference or disease."
        )

        continue

    # Use first unambiguous reference.
    ref = refs[0]

    metric = choose_metric(
        [ref] + disease
    )

    if metric is None:

        print(
            f"SKIP {panel_letter}: "
            "no shared hierarchy metric."
        )

        continue

    fig, ax = plt.subplots(
        figsize=(5.4, 4.4)
    )

    plotted = 0

    for dis in disease:

        r = ref["data"]
        d = dis["data"]

        ru = ref["u"]
        du = dis["u"]

        lo = max(
            np.nanmin(r[ru]),
            np.nanmin(d[du]),
        )

        hi = min(
            np.nanmax(r[ru]),
            np.nanmax(d[du]),
        )

        if (
            not np.isfinite(lo)
            or not np.isfinite(hi)
            or hi <= lo
        ):
            continue

        grid = np.linspace(
            lo,
            hi,
            400,
        )

        yr = interpolate(
            r,
            ru,
            metric,
            grid,
        )

        yd = interpolate(
            d,
            du,
            metric,
            grid,
        )

        if (
            yr is None
            or yd is None
        ):
            continue

        delta = yd - yr

        label = sample_label(
            dis["path"]
        )

        ax.plot(
            grid,
            delta,
            lw=1.8,
            label=label,
        )

        finite = np.isfinite(
            delta
        )

        if np.any(finite):

            j = np.nanargmax(
                np.abs(delta)
            )

            ax.scatter(
                grid[j],
                delta[j],
                s=38,
                zorder=5,
            )

            peak_rows.append({
                "organ": organ,
                "sample": label,
                "reference":
                    sample_label(
                        ref["path"]
                    ),
                "metric": metric,
                "peak_u":
                    float(grid[j]),
                "peak_delta":
                    float(delta[j]),
                "abs_peak_delta":
                    float(
                        abs(delta[j])
                    ),
            })

        plotted += 1

    if not plotted:

        plt.close(fig)
        continue

    ax.axhline(
        0,
        lw=0.9,
        alpha=0.65,
    )

    ax.set_xlabel(
        "Hierarchy progression, $u$"
    )

    ax.set_ylabel(
        f"Disease − reference\n"
        f"({metric})"
    )

    ax.set_title(
        "Brain"
        if organ == "brain"
        else "Kidney",
        fontweight="bold",
    )

    ax.legend(
        frameon=False,
        fontsize=7,
    )

    clean_axes(ax)

    fig.tight_layout()

    stem = (
        f"{panel_letter}_"
        f"{organ}_reference_relative_divergence"
    )

    save_panel(
        fig,
        stem,
    )

    built.append(stem)


pd.DataFrame(
    peak_rows
).to_csv(
    TABLES /
    "peak_disease_divergence.csv",
    index=False,
)


# ============================================================
# E/F — SPATIAL CANDIDATE AUDIT
# ============================================================

spatial_rows = []


if len(inventory):

    for _, row in inventory.iterrows():

        p = Path(
            row["path"]
        )

        if p.suffix.lower() not in {
            ".csv",
            ".tsv",
            ".parquet",
            ".pq",
        }:
            continue

        n = norm_name(p)

        if not any(
            x in n
            for x in [
                "cell",
                "spatial",
                "coordinate",
                "level0",
            ]
        ):
            continue

        d = read_table(p)

        if d is None:
            continue

        low = {
            c.lower(): c
            for c in d.columns
        }

        if (
            "x" not in low
            or "y" not in low
        ):
            continue

        spatial_rows.append({
            "path": str(p),
            "relative_path":
                str(
                    p.relative_to(
                        RESULTS_ROOT
                    )
                ),
            "organ":
                row["organ_guess"],
            "reference_like":
                row["reference_like"],
            "disease_like":
                row["disease_like"],
            "n_rows":
                len(d),
            "x_column":
                low["x"],
            "y_column":
                low["y"],
            "columns":
                "|".join(
                    d.columns
                ),
        })


pd.DataFrame(
    spatial_rows
).to_csv(
    AUDIT /
    "fig5_spatial_candidates.csv",
    index=False,
)


# ============================================================
# G/H — MOLECULAR CANDIDATE AUDIT
# ============================================================

molecular_rows = []


if len(inventory):

    for _, row in inventory.iterrows():

        p = Path(
            row["path"]
        )

        n = norm_name(p)

        if not any(
            x in n
            for x in [
                "expression",
                "gene",
                "molecular",
                "contrast",
                "program",
                "pathway",
            ]
        ):
            continue

        molecular_rows.append({
            "path": str(p),
            "relative_path":
                str(
                    p.relative_to(
                        RESULTS_ROOT
                    )
                ),
            "organ":
                row["organ_guess"],
            "reference_like":
                row["reference_like"],
            "disease_like":
                row["disease_like"],
            "suffix":
                p.suffix.lower(),
        })


pd.DataFrame(
    molecular_rows
).to_csv(
    AUDIT /
    "fig5_molecular_candidates.csv",
    index=False,
)


# ============================================================
# MANIFEST
# ============================================================

manifest = {
    "schema_version":
        "sutra.fig5.disease_trajectory.v1",
    "sutra_root":
        str(SUTRA_ROOT),
    "results_root":
        str(RESULTS_ROOT),
    "output_root":
        str(OUT_ROOT),
    "scientific_question":
        (
            "How does pathology deform "
            "multiscale tissue organization "
            "relative to nondiseased reference?"
        ),
    "intended_figure": {
        "A":
            "brain multiscale disease trajectory",
        "B":
            "kidney multiscale disease trajectory",
        "C":
            "brain reference-relative divergence",
        "D":
            "kidney reference-relative divergence",
        "E":
            "brain spatial localization of maximal disease departure",
        "F":
            "kidney spatial localization of maximal disease departure",
        "G":
            "brain molecular/program association with disease departure",
        "H":
            "kidney molecular/program association with disease departure",
    },
    "rules": [
        "Brain and kidney are never pooled.",
        "Only existing SUTRA results are read.",
        "Hierarchy construction is never rerun.",
        "Raw data are not modified.",
        "Filename classification is discovery metadata, not biological inference.",
        "Missing evidence produces an audit record rather than a fabricated panel.",
    ],
    "built_panels":
        built,
}


with open(
    MANIFEST /
    "FIG5_MANIFEST.json",
    "w",
) as f:

    json.dump(
        manifest,
        f,
        indent=2,
    )


# ============================================================
# README
# ============================================================

readme = """
SUTRA FIGURE 5 — DISEASE TRAJECTORIES
=====================================

Scientific question
-------------------
How does pathology deform multiscale tissue organization relative
to the corresponding nondiseased reference?

Current candidate architecture
------------------------------
A  Brain multiscale disease trajectory
B  Kidney multiscale disease trajectory
C  Brain reference-relative divergence
D  Kidney reference-relative divergence
E  Brain spatial localization of maximal disease departure
F  Kidney spatial localization of maximal disease departure
G  Brain molecular/program association
H  Kidney molecular/program association

Important
---------
Panels are only generated when the existing results tree provides
sufficient evidence to identify the required comparison.

The audit directory is therefore part of the deliverable, not an
error condition. It tells us exactly which materialized SUTRA
outputs can support E-H without inventing sample identity or biology.
"""

(
    OUT_ROOT /
    "README.txt"
).write_text(
    readme
)


# ============================================================
# FINAL REPORT
# ============================================================

print()
print("=" * 76)
print("FIG. 5 PREFLIGHT COMPLETE")
print("=" * 76)

print()
print("OUTPUT ROOT:")
print(OUT_ROOT)

print()
print("BUILT PANELS:")

if built:

    for x in built:
        print(" ", x)

else:

    print(
        "  none yet — inspect audit tables"
    )

print()
print("AUDIT:")
print(
    AUDIT /
    "fig5_results_inventory.csv"
)
print(
    AUDIT /
    "fig5_trajectory_tables.csv"
)
print(
    AUDIT /
    "fig5_spatial_candidates.csv"
)
print(
    AUDIT /
    "fig5_molecular_candidates.csv"
)

print()
print("NEXT:")
print(
    "Zip the complete Fig5_Disease_Trajectories "
    "directory and upload it."
)
