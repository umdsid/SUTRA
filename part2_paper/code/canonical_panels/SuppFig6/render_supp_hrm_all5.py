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
        0.5,
        -0.235,
        "Matched removed fraction, u",
        transform=ax.transAxes,
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


def terminal_summary_panel(ax, bt, kt):
    """Compact all-five summary of frozen terminal displacement."""
    tab = pd.concat([bt, kt], ignore_index=True)
    tab = tab[tab["case"].isin(CASES)].copy()

    sample_order = [x[0] for x in SAMPLES]
    sample_labels = [x[1] for x in SAMPLES]

    metrics = [
        ("terminal_VI_bits_vs_ref", "VI\n(bits)", plt.cm.Blues),
        ("terminal_node_relerr_vs_ref", "|ΔN| /\nNref", plt.cm.Greens),
        ("terminal_top1_absdiff_vs_ref", "|Δ top-1%\nmass|", plt.cm.Oranges),
        ("terminal_gini_absdiff_vs_ref", "|Δ Gini|", plt.cm.Reds),
    ]

    M = np.full((len(sample_order), len(metrics)), np.nan)

    for i, sample in enumerate(sample_order):
        q = tab[tab["sample"] == sample]

        for j, (col, _, _) in enumerate(metrics):
            vals = pd.to_numeric(
                q[col],
                errors="coerce",
            ).to_numpy(float)
            vals = vals[np.isfinite(vals)]

            if len(vals):
                M[i, j] = np.median(vals)

    rgba = np.ones((len(sample_order), len(metrics), 4))

    for j, (_, _, cmap) in enumerate(metrics):
        vals = M[:, j]
        finite = np.isfinite(vals)

        if not finite.any():
            continue

        vmax = np.nanmax(vals)

        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0

        for i, value in enumerate(vals):
            if np.isfinite(value):
                # Keep even low values visibly associated with their metric.
                intensity = 0.16 + 0.74 * (value / vmax)
                rgba[i, j] = cmap(intensity)

    ax.imshow(rgba, aspect="auto")

    ax.set_yticks(np.arange(len(sample_labels)))
    ax.set_yticklabels(sample_labels, fontsize=8)

    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels(
        [m[1] for m in metrics],
        fontsize=8,
    )

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            value = M[i, j]

            if not np.isfinite(value):
                continue

            vmax = np.nanmax(M[:, j])
            frac = value / vmax if vmax > 0 else 0

            ax.text(
                j,
                i,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=7.5,
                fontweight="bold" if frac > 0.70 else "normal",
                color="white" if frac > 0.72 else "black",
            )

    ax.set_title(
        "Median terminal displacement across perturbations",
        fontsize=10.3,
        fontweight="bold",
        pad=10,
    )

    ax.set_xlabel(
        "Metric",
        fontsize=8.5,
        labelpad=7,
    )

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.text(
        0.5,
        -0.19,
        "Color intensity is normalized separately within each metric; "
        "printed values are the frozen medians.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
    )


def family_response_panel(ax, bt, kt):
    """Compact specimen-by-family terminal response using frozen metrics only."""
    tab = pd.concat([bt, kt], ignore_index=True)
    tab = tab[tab["case"].isin(CASES)].copy()

    sample_order = [x[0] for x in SAMPLES]
    sample_labels = [x[1] for x in SAMPLES]
    families = ["threshold", "rescue", "schedule", "joint"]

    metrics = [
        ("terminal_VI_bits_vs_ref", "VI", "#3B82C4"),
        ("terminal_node_relerr_vs_ref", "|ΔN| / Nref", "#4C9A61"),
        ("terminal_top1_absdiff_vs_ref", "|Δ top-1% mass|", "#E58A2B"),
        ("terminal_gini_absdiff_vs_ref", "|Δ Gini|", "#C84A4A"),
    ]

    R = np.full((len(sample_order), len(families), len(metrics)), np.nan)
    for i, sample in enumerate(sample_order):
        for j, family in enumerate(families):
            cases = [c for c in CASES if FAMILY[c] == family]
            q = tab[(tab["sample"] == sample) & (tab["case"].isin(cases))]
            for k, (col, _, _) in enumerate(metrics):
                vals = pd.to_numeric(q[col], errors="coerce").to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if len(vals):
                    R[i, j, k] = np.median(vals)

    Z = np.full_like(R, np.nan)
    for k in range(len(metrics)):
        vals = R[:, :, k]
        vmax = np.nanmax(vals)
        Z[:, :, k] = vals / vmax if np.isfinite(vmax) and vmax > 0 else 0.0

    ax.set_xlim(-0.5, len(families) - 0.5)
    ax.set_ylim(len(sample_order) - 0.5, -0.5)

    # Quiet matrix scaffold: enough structure to compare rows/columns without
    # competing with the response profiles.
    for x in np.arange(-0.5, len(families) + 0.5, 1):
        ax.axvline(x, color="0.91", linewidth=0.65, zorder=0)
    for y in np.arange(-0.5, len(sample_order) + 0.5, 1):
        ax.axhline(y, color="0.91", linewidth=0.65, zorder=0)

    offsets = np.array([-0.225, -0.075, 0.075, 0.225])
    width = 0.105
    max_height = 0.58

    for i in range(len(sample_order)):
        baseline = i + 0.31
        for j in range(len(families)):
            finite = np.isfinite(Z[i, j, :])
            positive = finite & (Z[i, j, :] > 0)

            # Exact-zero families (not missing data) are made explicit rather
            # than appearing as an accidental blank column.
            if finite.any() and not positive.any():
                ax.text(j, i, "0", ha="center", va="center",
                        fontsize=7.2, color="0.48", zorder=3)
                continue

            for k, (_, _, color) in enumerate(metrics):
                z = Z[i, j, k]
                if not np.isfinite(z):
                    continue
                height = max_height * z
                if height <= 0:
                    # Zero-valued individual metrics get a short baseline tick,
                    # preserving zero without inventing a nonzero bar.
                    ax.plot([j + offsets[k] - width * 0.38,
                             j + offsets[k] + width * 0.38],
                            [baseline, baseline], color=color,
                            linewidth=1.15, solid_capstyle="butt", zorder=3)
                    continue
                ax.bar(j + offsets[k], height, width=width,
                       bottom=baseline - height, color=color,
                       edgecolor="none", zorder=2)

    ax.set_xticks(np.arange(len(families)))
    ax.set_xticklabels(families, fontsize=8.2)
    ax.set_yticks(np.arange(len(sample_labels)))
    ax.set_yticklabels(sample_labels, fontsize=8.0)
    ax.set_title("Perturbation-family response across specimens",
                 fontsize=10.3, fontweight="bold", pad=10)

    handles = [Line2D([0], [0], color=color, linewidth=5, label=label)
               for _, label, color in metrics]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.105), ncol=4,
              columnspacing=1.5, handlelength=2.2,
              frameon=False, fontsize=7.0)

    ax.text(0.5, -0.225,
            "Median of the two perturbations in each family; bar heights are normalized separately within each metric. Frontier perturbations produced zero displacement for all four reported metrics in all five specimens.",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.7)
    ax.tick_params(length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


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

    # Full landscapes are shown only for pathological specimens.
    # Healthy brain and nondiseased kidney are already represented
    # in Main Figure 4.
    pathological = [
        ("alzheimers", "Alzheimer's disease", "brain", "A"),
        ("gbm_reference_addon", "Glioblastoma", "brain", "B"),
        ("prcc", "Papillary renal cell carcinoma", "kidney", "C"),
    ]

    for sample, title, organ, letter in pathological:
        tab = brain if organ == "brain" else kidney

        fig, ax = plt.subplots(figsize=(13, 3.65))
        heat(fig, ax, tab, sample, title, norms)

        fig.subplots_adjust(
            left=0.14,
            right=0.985,
            top=0.84,
            bottom=0.30,
        )

        save(fig, out, f"SuppHRM_{letter}")

    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    terminal_panel(ax, bt, kt)
    fig.subplots_adjust(
        left=0.12,
        right=0.97,
        top=0.89,
        bottom=0.29,
    )
    save(fig, out, "SuppHRM_D")

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    terminal_summary_panel(ax, bt, kt)
    fig.subplots_adjust(
        left=0.29,
        right=0.985,
        top=0.88,
        bottom=0.25,
    )
    save(fig, out, "SuppHRM_E")

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    family_response_panel(ax, bt, kt)
    fig.subplots_adjust(
        left=0.27,
        right=0.985,
        top=0.88,
        bottom=0.30,
    )
    save(fig, out, "SuppHRM_F")

    manifest = {
        "figure": "Supplementary Figure 6",
        "mode": "standalone_panels",
        "assembled_figure": False,
        "scientific_recalculation": False,
        "panels": {
            "A": "Alzheimer's disease robustness landscape",
            "B": "Glioblastoma robustness landscape",
            "C": "Papillary renal cell carcinoma robustness landscape",
            "D": "Terminal displacement across five specimens",
            "E": "All-five terminal robustness summary",
            "F": "Perturbation-family response across specimens",
        },
        "reference_landscapes_not_repeated": [
            "Healthy brain",
            "Nondiseased kidney",
        ],
        "source": "Frozen Main Figure 4 HRM trajectory and terminal-summary tables",
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("WROTE", out / "manifest.json")


if __name__ == "__main__":
    main()
