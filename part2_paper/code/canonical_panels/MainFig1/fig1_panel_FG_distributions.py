#!/usr/bin/env python3

import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


ROOT = Path(
    os.environ.get(
        "SUTRA_ROOT",
        str(Path.home() / "Desktop" / "SUTRA"),
    )
)

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
    / "healthy_reference"
)

CAND = (
    LEDGER
    / "candidate_boundaries"
    / "step_000000.parquet"
)

FIGROOT = Path(
    os.environ.get(
        "SUTRA_FIG1_OUT",
        str(ROOT / "figures" / "fig1"),
    )
)

OUT = FIGROOT / "panels"
SRC = FIGROOT / "source_data"

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

SRC.mkdir(
    parents=True,
    exist_ok=True,
)


CHANNELS = [
    ("molecular_effective", "Molecular"),
    ("mechanics_effective", "Mechanics"),
    (
        "communication_support_effective",
        "Communication",
    ),
    ("geometry_effective", "Geometry"),
]


# ============================================================
# Entire Level-0 candidate population
# ============================================================

cols = [
    *[x[0] for x in CHANNELS],
    "composite_merge_cost",
    "merge_cost_threshold",
    "merge_allowed",
    "admissible",
]

c = pq.read_table(
    CAND,
    columns=cols,
).to_pandas()


# Production code defines admissible == merge_allowed.
# Use merge_allowed explicitly because it is the threshold
# decision we want to visualize here.
allowed = c[
    c["merge_allowed"].astype(bool)
].copy()

retained = c[
    ~c["merge_allowed"].astype(bool)
].copy()


print(
    "Level-0 candidates:",
    len(c),
)

print(
    "admissible:",
    len(allowed),
)

print(
    "retained:",
    len(retained),
)

print(
    "fractions:",
    len(allowed) / len(c),
    len(retained) / len(c),
)


# ============================================================
# Convert each continuous channel independently to its
# percentile within the ENTIRE Level-0 candidate population.
# ============================================================

def percentile(
    reference,
    values,
):

    ref = np.asarray(
        reference,
        dtype=float,
    )

    ref = np.sort(
        ref[np.isfinite(ref)]
    )

    x = np.asarray(
        values,
        dtype=float,
    )

    result = np.full(
        len(x),
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(x)

    result[finite] = (
        100.0
        * np.searchsorted(
            ref,
            x[finite],
            side="right",
        )
        / len(ref)
    )

    return result


for col, label in CHANNELS:

    reference = c[
        col
    ].to_numpy(
        dtype=float,
    )

    allowed[label] = percentile(
        reference,
        allowed[col].to_numpy(
            dtype=float,
        ),
    )

    retained[label] = percentile(
        reference,
        retained[col].to_numpy(
            dtype=float,
        ),
    )


# ============================================================
# Small provenance summary
# ============================================================

summary = {
    "all_level0_candidates": len(c),
    "admissible_level0": len(allowed),
    "retained_level0": len(retained),
    "admissible_fraction": (
        len(allowed) / len(c)
    ),
    "retained_fraction": (
        len(retained) / len(c)
    ),
}

import json

(
    SRC
    / "panel_FG_population_summary.json"
).write_text(
    json.dumps(
        summary,
        indent=2,
    )
    + "\n"
)


# ============================================================
# Plot
# ============================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelweight": "bold",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def make(
    df,
    color,
    stem,
):

    fig, ax = plt.subplots(
        figsize=(4.7, 2.85),
        facecolor="white",
    )

    values = [
        df[label]
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .to_numpy()
        for _, label in CHANNELS
    ]

    bp = ax.boxplot(
        values,
        patch_artist=True,
        widths=.52,
        showfliers=False,
        medianprops=dict(
            color=color,
            linewidth=1.5,
        ),
        whiskerprops=dict(
            color=color,
            linewidth=1.0,
        ),
        capprops=dict(
            color=color,
            linewidth=1.0,
        ),
    )

    for box in bp["boxes"]:

        box.set(
            facecolor=color,
            edgecolor=color,
            alpha=.18,
            linewidth=1.1,
        )

    # Keep the zero-percentile baseline visibly above the lower frame.
    # This is display clearance only; percentile values are unchanged.
    ax.set_ylim(
        -4,
        100,
    )

    ax.set_ylabel(
        "Level-0 percentile",
        fontsize=7.5,
        weight="bold",
    )

    ax.set_xticks(
        range(1, 5),
    )

    ax.set_xticklabels(
        [
            x[1]
            for x in CHANNELS
        ],
        fontsize=7.1,
    )

    ax.tick_params(
        axis="y",
        labelsize=7,
    )

    ax.grid(
        axis="y",
        alpha=.14,
        linewidth=.6,
    )

    ax.spines[
        "top"
    ].set_visible(False)

    ax.spines[
        "right"
    ].set_visible(False)

    fig.tight_layout(
        pad=.45,
    )

    fig.savefig(
        OUT / f"{stem}.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=.02,
    )

    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=.02,
    )

    plt.close(fig)


make(
    allowed,
    "#16853d",
    "panel_F",
)

make(
    retained,
    "#b33b3b",
    "panel_G",
)

print()
print("F/G COMPLETE")
