from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable


CASES = [
    "epsilon_low",
    "epsilon_ceiling",
    "rescue_tighter",
    "rescue_looser",
    "soft_early",
    "soft_late",
    "frontier_low",
    "frontier_high",
    "eps_low_soft_late",
    "eps_rescue_high_soft_early",
]

LABELS = {
    "epsilon_low": "threshold ↓",
    "epsilon_ceiling": "threshold ceiling",
    "rescue_tighter": "rescue tighter",
    "rescue_looser": "rescue looser",
    "soft_early": "schedule early",
    "soft_late": "schedule late",
    "frontier_low": "frontier ↓",
    "frontier_high": "frontier ↑",
    "eps_low_soft_late": "joint: threshold + schedule",
    "eps_rescue_high_soft_early": "joint: rescue + schedule",
}

FAMILY = {
    "epsilon_low": "threshold",
    "epsilon_ceiling": "threshold",
    "rescue_tighter": "rescue",
    "rescue_looser": "rescue",
    "soft_early": "schedule",
    "soft_late": "schedule",
    "frontier_low": "frontier",
    "frontier_high": "frontier",
    "eps_low_soft_late": "joint",
    "eps_rescue_high_soft_early": "joint",
}

METRICS = [
    ("VI_bits_vs_ref", "VI (bits)", "Blues"),
    ("node_relerr_vs_ref", "|ΔN| / Nref", "Greens"),
    ("top1_absdiff_vs_ref", "|Δ top-1% mass|", "Oranges"),
    ("gini_absdiff_vs_ref", "|Δ Gini|", "Reds"),
]

SAMPLES = [
    ("healthy_reference", "Healthy brain", "brain", "A"),
    ("alzheimers", "Alzheimer's disease", "brain", "B"),
    ("gbm_reference_addon", "Glioblastoma", "brain", "C"),
    ("nondiseased_kidney", "Nondiseased kidney", "kidney", "D"),
    ("prcc", "Papillary renal cell carcinoma", "kidney", "E"),
]

FAMILY_COLORS = {
    "threshold": "#3B82C4",
    "rescue": "#4C9A61",
    "schedule": "#E58A2B",
    "frontier": "#C84A4A",
    "joint": "#9B59B6",
}

SAMPLE_MARKERS = {
    "healthy_reference": "o",
    "alzheimers": "^",
    "gbm_reference_addon": "D",
    "nondiseased_kidney": "s",
    "prcc": "P",
}


def save(fig, out, stem):
    fig.savefig(
        out / f"{stem}.png",
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        out / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print("WROTE", out / f"{stem}.png")
    print("WROTE", out / f"{stem}.pdf")


def load(data):
    return (
        pd.read_csv(data / "brain_hrm_matched_trajectory.csv"),
        pd.read_csv(data / "kidney_hrm_matched_trajectory.csv"),
        pd.read_csv(data / "brain_hrm_terminal_summary.csv"),
        pd.read_csv(data / "kidney_hrm_terminal_summary.csv"),
    )


def validate(brain, kidney, bt, kt):
    expected = {
        "healthy_reference",
        "alzheimers",
        "gbm_reference_addon",
        "nondiseased_kidney",
        "prcc",
    }

    traj = set(brain["sample"]) | set(kidney["sample"])
    term = set(bt["sample"]) | set(kt["sample"])

    if traj != expected:
        raise RuntimeError(
            f"Trajectory specimen contract failed: {sorted(traj)}"
        )

    if term != expected:
        raise RuntimeError(
            f"Terminal specimen contract failed: {sorted(term)}"
        )

    for sample, _, organ, _ in SAMPLES:
        tab = brain if organ == "brain" else kidney
        q = tab[tab["sample"] == sample]

        if len(q) != 55:
            raise RuntimeError(
                f"{sample}: expected 55 trajectory rows, got {len(q)}"
            )

        if q["case"].nunique() != 11:
            raise RuntimeError(
                f"{sample}: expected 11 cases"
            )

        if q["target_removed_fraction"].nunique() != 5:
            raise RuntimeError(
                f"{sample}: expected 5 matched states"
            )

    terminals = pd.concat([bt, kt], ignore_index=True)
    pert = terminals[terminals["case"].isin(CASES)]

    if len(pert) != 50:
        raise RuntimeError(
            f"Expected 50 perturbed terminal runs, got {len(pert)}"
        )

    stop = (
        pert["stop_reason"]
        == "no_contextually_admissible_boundaries"
    )

    if not stop.all():
        raise RuntimeError(
            "Not all perturbed runs reached the expected stop condition"
        )


def make_norms(brain, kidney):
    norms = {}

    for metric, _, _ in METRICS:
        vals = np.r_[
            brain.loc[brain["case"] != "reference", metric].to_numpy(float),
            kidney.loc[kidney["case"] != "reference", metric].to_numpy(float),
        ]
        vals = vals[np.isfinite(vals)]

        if not len(vals):
            raise RuntimeError(f"No finite values for {metric}")

        vmax = float(vals.max())
        if vmax <= 0:
            vmax = 1.0

        norms[metric] = Normalize(0.0, vmax)

    return norms


def heat(fig, ax, df, sample, title, norms):
    sub = df[
        (df["sample"] == sample)
        & df["case"].isin(CASES)
    ].copy()

    ts = sorted(sub["target_removed_fraction"].unique())
    W = len(ts)
    gap = 0.70

    for k, (metric, lab, cmap) in enumerate(METRICS):
        z = np.full((len(CASES), W), np.nan)

        for i, case in enumerate(CASES):
            q = (
                sub[sub["case"] == case]
                .set_index("target_removed_fraction")
            )

            for j, t in enumerate(ts):
                if t in q.index:
                    z[i, j] = float(q.loc[t, metric])

        x0 = k * (W + gap)

        x_edges = np.arange(W + 1, dtype=float) + x0 - 0.5
        y_edges = np.arange(len(CASES) + 1, dtype=float) - 0.5

        ax.pcolormesh(
            x_edges,
            y_edges,
            z,
            cmap=cmap,
            norm=norms[metric],
            shading="flat",
            antialiased=False,
            rasterized=False,
        )

        ax.text(
            x0 + (W - 1) / 2,
            -1.22,
            lab,
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

        for j, t in enumerate(ts):
            ax.text(
                x0 + j,
                10.04,
                f"{t:.2f}",
                rotation=45,
                ha="right",
                va="top",
                fontsize=6.4,
            )

        cbax = ax.inset_axes([
            (x0 + 0.05) / (4 * W + 3 * gap),
            -0.155,
            (W - 0.10) / (4 * W + 3 * gap),
            0.032,
        ])

        sm = ScalarMappable(
            norm=norms[metric],
            cmap=cmap,
        )
        sm.set_array([])

        cb = fig.colorbar(
            sm,
            cax=cbax,
            orientation="horizontal",
        )
        cb.ax.tick_params(
            labelsize=5.7,
            length=1.5,
            pad=1,
        )
        cb.outline.set_linewidth(0.4)

    ax.set_xlim(
        -0.5,
        3 * (W + gap) + W - 0.5,
    )
    ax.set_ylim(11.15, -1.85)

    ax.set_yticks(range(len(CASES)))
    ax.set_yticklabels(
        [LABELS[c] for c in CASES],
        fontsize=7.2,
    )
    ax.set_xticks([])

    ax.text(
        np.mean(ax.get_xlim()),
        11.0,
        "matched removed fraction, u",
        ha="center",
        va="top",
        fontsize=7.3,
    )

    ax.set_title(
        title,
        fontsize=10.3,
        fontweight="bold",
        pad=10,
    )

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


def terminal_panel(ax, bt, kt):
    tab = pd.concat([bt, kt], ignore_index=True)
    tab = tab[tab["case"].isin(CASES)].copy()

    for sample, title, organ, _ in SAMPLES:
        q = tab[tab["sample"] == sample]

        for _, r in q.iterrows():
            family = FAMILY[r["case"]]
            color = FAMILY_COLORS[family]

            ax.scatter(
                float(r["terminal_VI_bits_vs_ref"]),
                float(r["terminal_node_relerr_vs_ref"]),
                marker=SAMPLE_MARKERS[sample],
                s=52,
                facecolors=color if organ == "brain" else "none",
                edgecolors=color,
                linewidths=1.05,
                alpha=0.90,
                zorder=3,
            )

    ax.set_xlabel(
        "Terminal partition displacement, VI (bits)",
        fontsize=9,
    )
    ax.set_ylabel(
        "Terminal hierarchy-extent displacement, |ΔN| / Nref",
        fontsize=9,
    )

    ax.set_title(
        "Terminal displacement across five specimens",
        fontsize=10.3,
        fontweight="bold",
        pad=10,
    )

    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)

    ok = (
        tab["stop_reason"]
        == "no_contextually_admissible_boundaries"
    ).sum()

    ax.text(
        0.98,
        0.98,
        (
            f"{ok}/{len(tab)} construction-rule perturbations terminated with\n"
            "no contextually admissible boundaries remaining"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.3,
        fontweight="bold",
    )

    sample_handles = []

    for sample, title, organ, _ in SAMPLES:
        sample_handles.append(
            Line2D(
                [0],
                [0],
                marker=SAMPLE_MARKERS[sample],
                linestyle="none",
                markersize=6,
                markerfacecolor="0.35" if organ == "brain" else "none",
                markeredgecolor="0.25",
                label=title,
            )
        )

    family_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=color,
            markeredgecolor=color,
            label=family,
        )
        for family, color in FAMILY_COLORS.items()
    ]

    leg1 = ax.legend(
        handles=sample_handles,
        loc="upper left",
        bbox_to_anchor=(0.0, -0.18),
        frameon=False,
        fontsize=7,
        ncol=2,
        title="Specimen",
        title_fontsize=7.5,
    )
    ax.add_artist(leg1)

    ax.legend(
        handles=family_handles,
        loc="upper right",
        bbox_to_anchor=(1.0, -0.18),
        frameon=False,
        fontsize=7,
        ncol=3,
        title="Perturbation family",
        title_fontsize=7.5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    args = ap.parse_args()

    data = args.data_root.expanduser().resolve()
    out = args.output_root.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    brain, kidney, bt, kt = load(data)
    validate(brain, kidney, bt, kt)
    norms = make_norms(brain, kidney)

    for sample, title, organ, letter in SAMPLES:
        tab = brain if organ == "brain" else kidney

        fig, ax = plt.subplots(figsize=(13, 3.65))
        heat(fig, ax, tab, sample, title, norms)

        fig.subplots_adjust(
            left=0.14,
            right=0.985,
            top=0.84,
            bottom=0.24,
        )

        save(
            fig,
            out,
            f"SuppHRM_{letter}",
        )

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    terminal_panel(ax, bt, kt)

    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.89,
        bottom=0.29,
    )

    save(fig, out, "SuppHRM_F")

    manifest = {
        "figure": "SuppHRM_All5",
        "status": "STAGED_COMPLETE",
        "mode": "standalone_panels",
        "panels": [
            "SuppHRM_A",
            "SuppHRM_B",
            "SuppHRM_C",
            "SuppHRM_D",
            "SuppHRM_E",
            "SuppHRM_F",
        ],
        "assembled_figure": False,
        "scientific_recalculation": False,
        "source": "MainFig4 frozen HRM tables",
        "normalization": (
            "metric-specific common normalization across all five specimens"
        ),
        "brain_kidney_separate": (
            "A-C brain; D-E kidney; F integrated terminal summary"
        ),
        "terminal_contract": (
            "50/50 construction-rule perturbations terminated with "
            "no contextually admissible boundaries remaining"
        ),
    }

    (
        out / "SuppHRM_manifest.json"
    ).write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("PASS: all-five HRM supplementary family rendered.")


if __name__ == "__main__":
    main()
