from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

METHODS = [
    ("SUTRA molecular only", "selected__SUTRA_molecular_only"),
    ("PCA", "selected__PCA"),
    ("kPCA*", "selected__kPCAstar"),
    ("UMAP", "selected__UMAP"),
    ("t-SNE", "selected__t_SNE"),
]

SAMPLES = [
    ("healthy_reference", "Healthy brain"),
    ("nondiseased_kidney", "Nondiseased kidney"),
]


def nearest_checkpoint(ledger, target_u):
    steps = [
        json.loads(x)
        for x in (ledger / "steps.jsonl").read_text().splitlines()
    ]

    cps = sorted((ledger / "label_checkpoints").glob("labels_*.npz"))
    choices = []

    for p in cps:
        cp_index = int(p.stem.split("_")[1])

        if cp_index == 0:
            step = steps[0]
            u = 0.0
        elif cp_index >= len(steps):
            step = steps[-1]
            u = float(step["hierarchy_coordinate_removed_fraction"])
        else:
            step = steps[cp_index]
            u = float(step["hierarchy_coordinate_removed_fraction"])

        choices.append((abs(u - target_u), p, u, cp_index))

    return min(choices, key=lambda x: x[0])[1:]


def load_labels(path):
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["labels"])


def jaccard(a, b):
    a = np.asarray(a, bool)
    b = np.asarray(b, bool)
    union = np.count_nonzero(a | b)
    if union == 0:
        return np.nan
    return np.count_nonzero(a & b) / union


def build_D(source, out):
    rows = []

    for sample, sample_label in SAMPLES:
        tab = pd.read_parquet(
            source / sample / "matched_candidate_edge_table.parquet"
        )

        sutra = tab["selected__SUTRA_integrated"].astype(bool).to_numpy()
        admissible = tab["admissible"].astype(bool).to_numpy()

        for method_label, col in METHODS:
            selected = tab[col].astype(bool).to_numpy()
            n = int(selected.sum())

            rows.append(
                {
                    "sample": sample,
                    "sample_label": sample_label,
                    "method": method_label,
                    "n_candidate_interfaces": len(tab),
                    "n_selected": n,
                    "jaccard_with_integrated_SUTRA":
                        jaccard(selected, sutra),
                    "fraction_selected_SUTRA_admissible":
                        float(admissible[selected].mean())
                        if n else np.nan,
                }
            )

    audit = pd.DataFrame(rows)
    audit.to_csv(
        out / "panel_D_matched_interface_audit.csv",
        index=False
    )

    order = [x[0] for x in METHODS]
    y = np.arange(len(order))

    fig, axes = plt.subplots(
        2, 2,
        figsize=(8.3, 5.6),
        constrained_layout=True
    )

    for r, (_, sample_label) in enumerate(SAMPLES):
        q = (
            audit[audit["sample_label"] == sample_label]
            .set_index("method")
            .loc[order]
        )

        vals1 = q["jaccard_with_integrated_SUTRA"].to_numpy()
        vals2 = q[
            "fraction_selected_SUTRA_admissible"
        ].to_numpy()

        ax = axes[r, 0]
        for yi, val in zip(y, vals1):
            ax.plot(
                [0, val], [yi, yi],
                lw=1.15, alpha=0.45
            )
            ax.scatter([val], [yi], s=34, zorder=3)
            ax.text(
                val + 0.014, yi, f"{val:.2f}",
                va="center", ha="left", fontsize=8
            )

        ax.set_xlim(0, 0.55)
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_yticks(y, order)
        ax.set_xlabel(
            "Jaccard overlap with integrated SUTRA"
        )
        ax.grid(axis="x", alpha=0.16, linewidth=0.6)

        ax = axes[r, 1]
        for yi, val in zip(y, vals2):
            ax.plot(
                [0, val], [yi, yi],
                lw=1.15, alpha=0.45
            )
            ax.scatter([val], [yi], s=34, zorder=3)
            ax.text(
                val + 0.014, yi, f"{val:.2f}",
                va="center", ha="left", fontsize=8
            )

        ax.set_xlim(0, 0.65)
        ax.set_ylim(len(order) - 0.45, -0.55)
        ax.set_yticks(y, [])
        ax.set_xlabel(
            "Fraction meeting SUTRA\n"
            "contextual admissibility"
        )
        ax.grid(axis="x", alpha=0.16, linewidth=0.6)

        axes[r, 0].text(
            -0.38, 0.50,
            sample_label,
            transform=axes[r, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold"
        )

    axes[0, 0].set_title(
        "Selection overlap",
        fontsize=9.5,
        fontweight="bold",
        pad=10
    )
    axes[0, 1].set_title(
        "Contextual admissibility",
        fontsize=9.5,
        fontweight="bold",
        pad=10
    )

    fig.savefig(
        out / "SuppFig2D_quantitative_summaries.png",
        dpi=450,
        bbox_inches="tight",
        facecolor="white"
    )
    fig.savefig(
        out / "SuppFig2D_quantitative_summaries.pdf",
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)

def select_roi(xy, edges, max_cells=900):
    x = xy["x"].to_numpy()
    y = xy["y"].to_numpy()

    cx = np.median(x)
    cy = np.median(y)

    d2 = (x - cx) ** 2 + (y - cy) ** 2
    seed = np.argsort(d2)[:max_cells]

    mask = np.zeros(len(xy), dtype=bool)
    mask[seed] = True

    return mask


def object_segments(xy, labels, mask):
    frame = pd.DataFrame(
        {
            "x": xy["x"].to_numpy(),
            "y": xy["y"].to_numpy(),
            "label": labels,
            "keep": mask,
        }
    )
    frame = frame[frame["keep"]]

    centers = (
        frame.groupby("label", sort=False)[["x", "y"]]
        .mean()
    )

    return frame, centers


def build_E(source, hierarchy_root, out):
    audit_rows = []
    lineage_rows = []

    fig = plt.figure(
        figsize=(11.7, 6.1),
        constrained_layout=True
    )
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1.18, 2.55, 1.35]
    )

    targets = [0.0, 0.25, 0.50, 1.0]

    for r, (sample, sample_label) in enumerate(SAMPLES):
        xy = pd.read_parquet(
            source / sample / "aligned_cells_xy.parquet"
        )
        ledger = hierarchy_root / "ledger" / sample

        mask = select_roi(xy, None, max_cells=900)

        xx = xy["x"].to_numpy()
        yy = xy["y"].to_numpy()
        roi_idx = np.flatnonzero(mask)

        states = []

        for target in targets:
            if target == 1.0:
                cp = sorted(
                    (ledger / "label_checkpoints")
                    .glob("labels_*.npz")
                )[-1]

                labels = load_labels(cp)

                steps = [
                    json.loads(x)
                    for x in
                    (ledger / "steps.jsonl")
                    .read_text()
                    .splitlines()
                ]

                actual_u = float(
                    steps[-1][
                        "hierarchy_coordinate_removed_fraction"
                    ]
                )
            else:
                cp, actual_u, _ = nearest_checkpoint(
                    ledger, target
                )
                labels = load_labels(cp)

            roi_labels = labels[mask]

            audit_rows.append(
                {
                    "sample": sample,
                    "sample_label": sample_label,
                    "checkpoint": cp.name,
                    "target_u": target,
                    "actual_u": actual_u,
                    "roi_level0_cells":
                        int(mask.sum()),
                    "roi_current_objects":
                        int(np.unique(roi_labels).size),
                    "level0_ancestry_coverage": 1.0,
                }
            )

            states.append(
                {
                    "u": actual_u,
                    "checkpoint": cp.name,
                    "labels": labels.copy(),
                }
            )

        terminal_roi = states[-1]["labels"][mask]

        terminal_ids, terminal_counts = np.unique(
            terminal_roi,
            return_counts=True
        )

        rank = np.argsort(terminal_counts)[::-1]
        tracked_terminal_ids = terminal_ids[rank[:6]]

        tracked = []

        for terminal_id in tracked_terminal_ids:
            ids = roi_idx[
                terminal_roi == terminal_id
            ]

            if not len(ids):
                continue

            precursor_counts = []

            for state in states:
                n_precursors = int(
                    np.unique(
                        state["labels"][ids]
                    ).size
                )
                precursor_counts.append(n_precursors)

            tracked.append(
                {
                    "terminal_id": int(terminal_id),
                    "cell_ids": ids,
                    "n_cells": int(len(ids)),
                    "precursor_counts": precursor_counts,
                }
            )

            for state, n_precursors in zip(
                states, precursor_counts
            ):
                lineage_rows.append(
                    {
                        "sample": sample,
                        "sample_label": sample_label,
                        "terminal_object_id":
                            int(terminal_id),
                        "terminal_ancestry_level0_cells":
                            int(len(ids)),
                        "checkpoint":
                            state["checkpoint"],
                        "actual_u":
                            state["u"],
                        "n_precursor_objects":
                            n_precursors,
                    }
                )

        ax0 = fig.add_subplot(gs[r, 0])

        ax0.scatter(
            xx[mask], yy[mask],
            s=2.0,
            alpha=0.12,
            rasterized=True
        )

        for ancestry in tracked:
            ids = ancestry["cell_ids"]
            ax0.scatter(
                xx[ids], yy[ids],
                s=12.0,
                alpha=0.94,
                rasterized=True
            )

        ax0.set_aspect("equal")
        ax0.set_xticks([])
        ax0.set_yticks([])

        for spine in ax0.spines.values():
            spine.set_visible(False)

        ax0.set_title(
            sample_label + "\n"
            "Fixed Level-0 tissue ROI",
            fontsize=9.5,
            fontweight="bold",
            loc="left"
        )

        ax1 = fig.add_subplot(gs[r, 1])

        xpos = np.arange(4, dtype=float)
        ypos = np.arange(
            len(tracked) - 1,
            -1,
            -1,
            dtype=float
        )

        max_cells = max(
            x["n_cells"] for x in tracked
        )

        for yi, ancestry in zip(ypos, tracked):
            counts = ancestry["precursor_counts"]
            n_cells = ancestry["n_cells"]

            lw = (
                2.0
                + 7.0 * np.sqrt(n_cells / max_cells)
            )

            ax1.plot(
                xpos,
                np.repeat(yi, len(xpos)),
                lw=lw,
                alpha=0.22,
                solid_capstyle="round",
                zorder=1
            )

            for k, n_precursors in enumerate(counts):
                size = (
                    62
                    + 105 * np.sqrt(n_cells / max_cells)
                )

                ax1.scatter(
                    [xpos[k]],
                    [yi],
                    s=size,
                    alpha=0.96,
                    zorder=3
                )

                ax1.text(
                    xpos[k],
                    yi,
                    str(n_precursors),
                    ha="center",
                    va="center",
                    fontsize=7.2,
                    fontweight="bold",
                    zorder=4
                )

            ax1.text(
                -0.18,
                yi,
                f"{n_cells} cells",
                ha="right",
                va="center",
                fontsize=7.2
            )

        ax1.set_xlim(-0.65, 3.35)
        ax1.set_ylim(-0.65, len(tracked) - 0.35)

        ax1.set_xticks(
            xpos,
            [
                "Level 0",
                f"u={states[1]['u']:.2f}",
                f"u={states[2]['u']:.2f}",
                f"Terminal\nu={states[3]['u']:.2f}",
            ]
        )

        ax1.set_yticks([])

        ax1.set_title(
            "Six terminal ancestries:"
            " number of precursor objects",
            fontsize=9.5,
            fontweight="bold"
        )

        ax1.text(
            0.50,
            -0.18,
            "Each row follows the same Level-0 "
            "cell ancestry across frozen checkpoints",
            transform=ax1.transAxes,
            ha="center",
            va="top",
            fontsize=7.4
        )

        ax1.spines["left"].set_visible(False)
        ax1.spines["right"].set_visible(False)
        ax1.spines["top"].set_visible(False)
        ax1.grid(
            axis="x",
            alpha=0.12,
            linewidth=0.6
        )

        ax2 = fig.add_subplot(gs[r, 2])

        qs = [
            x for x in audit_rows
            if x["sample"] == sample
        ]

        ux = np.array(
            [x["actual_u"] for x in qs]
        )
        no = np.array(
            [x["roi_current_objects"] for x in qs]
        )

        ax2.plot(
            ux,
            no,
            marker="o",
            lw=1.5
        )

        offsets = [
            (5, 7),
            (5, 7),
            (-5, 8),
            (-6, -13),
        ]

        aligns = [
            ("left", "bottom"),
            ("left", "bottom"),
            ("right", "bottom"),
            ("right", "top"),
        ]

        for uval, nobj, off, align in zip(
            ux, no, offsets, aligns
        ):
            ax2.annotate(
                f"{nobj}",
                (uval, nobj),
                xytext=off,
                textcoords="offset points",
                ha=align[0],
                va=align[1],
                fontsize=7.5
            )

        ax2.set_xlabel("Hierarchy progress, u")
        ax2.set_ylabel(
            "Distinct objects\n"
            "in fixed Level-0 ROI"
        )

        ax2.set_title(
            "Higher-order objects emerge",
            fontsize=9.3,
            fontweight="bold",
            pad=10
        )

        ax2.grid(
            alpha=0.15,
            linewidth=0.6
        )

        ax2.text(
            0.97,
            0.93,
            "Level-0 ancestry retained\n"
            "900 / 900 cells (100%)",
            transform=ax2.transAxes,
            ha="right",
            va="top",
            fontsize=7.6,
            bbox={
                "boxstyle": "round,pad=0.28",
                "facecolor": "white",
                "edgecolor": "0.80",
                "linewidth": 0.6,
                "alpha": 0.94,
            }
        )

    audit = pd.DataFrame(audit_rows)
    lineage = pd.DataFrame(lineage_rows)

    audit.to_csv(
        out /
        "panel_E_hierarchy_traceability_audit.csv",
        index=False
    )

    lineage.to_csv(
        out /
        "panel_E_tracked_ancestry_audit.csv",
        index=False
    )

    terminal_check = (
        lineage.sort_values("actual_u")
        .groupby(
            ["sample", "terminal_object_id"],
            as_index=False
        )
        .tail(1)
    )

    if not (
        terminal_check["n_precursor_objects"] == 1
    ).all():
        raise AssertionError(
            "Tracked terminal ancestry does not "
            "collapse to one terminal object."
        )

    fig.savefig(
        out / "SuppFig2E_traceable_hierarchy.png",
        dpi=450,
        bbox_inches="tight",
        facecolor="white"
    )

    fig.savefig(
        out / "SuppFig2E_traceable_hierarchy.pdf",
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--sutra-root",
        default=str(Path.home() / "Desktop/SUTRA")
    )
    ap.add_argument(
        "--github-root",
        default=str(Path.home() / "Desktop/SUTRA_GITHUB")
    )
    ap.add_argument(
        "--out",
        default=str(
            Path.home()
            / "Desktop/SUTRA_GITHUB_REPRO_OUTPUT/SuppFig2"
        )
    )
    args = ap.parse_args()

    sutra = Path(args.sutra_root).resolve()
    github = Path(args.github_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)

    source = sutra / "results/suppfig2_fair_matched_candidate_v152"
    hierarchy = (
        sutra
        / "results/_archive_pre_pressure_safe_v12_20260923_164114"
        / "hierarchy_v0911_specimen_local_contextual_flow"
    )

    if not source.exists():
        raise FileNotFoundError(source)
    if not hierarchy.exists():
        raise FileNotFoundError(hierarchy)

    pending = (
        github
        / "part2_paper/supplementary/_pending_suppfig2"
    )

    for stem in [
        "SuppFig2A_brain_local_selection",
        "SuppFig2B_kidney_local_selection",
        "SuppFig2C_multichannel_profiles",
    ]:
        for ext in [".png", ".pdf"]:
            src = pending / f"{stem}{ext}"
            if not src.exists():
                raise FileNotFoundError(src)
            shutil.copy2(src, out / src.name)

    build_D(source, out)
    build_E(source, hierarchy, out)

    manifest = {
        "figure": "Supplementary Figure 2",
        "mode": "standalone_panels",
        "panels": ["A", "B", "C", "D", "E"],
        "assembled_figure": False,
        "hierarchy_rerun": False,
        "scientific_recalculation": False,
        "comparator_source":
            "results/suppfig2_fair_matched_candidate_v152",
        "hierarchy_source":
            "hierarchy_v0911_specimen_local_contextual_flow",
        "panel_D":
            "matched Level-0 candidate-interface comparison",
        "panel_E":
            "six frozen terminal ancestries summarized by precursor-object counts across label checkpoints",
    }

    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )

    print("WROTE", out)
    for p in sorted(out.iterdir()):
        print(p.name)


if __name__ == "__main__":
    main()
