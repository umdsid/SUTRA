#!/usr/bin/env python3

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


ROOT = Path.home() / "Desktop" / "SUTRA"

FIGROOT = (
    ROOT / "results" / "Fig5_Disease_Trajectories"
)

V2 = FIGROOT / "gw_geodesic_v2"
V3 = FIGROOT / "gw_multistart_v3"
V4 = FIGROOT / "gw_biology_v4"

OUT = FIGROOT / "final_v5"
OUT.mkdir(parents=True, exist_ok=True)

U = np.array([0.32, 0.40, 0.48, 0.56], dtype=float)

# Number of molecular trajectories shown in panel E/F.
N_TRAJECTORY_GENES = 8
N_MATRIX_GENES = 18

EPS = 1e-12


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def panel_label(ax, letter):
    ax.text(
        -0.10, 1.06, letter,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
    )


def load_multistart():
    p = V3 / "multistart_scale_summary.csv"
    x = pd.read_csv(p)

    required = [
        "u_target",
        "best_gw_distance",
        "independent_gw_distance",
        "best_start",
    ]

    for c in required:
        if c not in x.columns:
            raise RuntimeError(f"{p.name} missing {c}")

    return x.sort_values("u_target").reset_index(drop=True)


def load_k_convergence():
    p = V2 / "gw_K_convergence.csv"

    if not p.exists():
        return None

    return pd.read_csv(p)


def load_molecular():
    p = (
        V4 / "molecular" /
        "gene_integrated_scale_associations.csv"
    )

    x = pd.read_csv(p)

    required = [
        "u",
        "gene",
        "cell_spearman",
        "object_spearman_sqrt_mass",
        "high_minus_local_logexpr",
    ]

    for c in required:
        if c not in x.columns:
            raise RuntimeError(f"{p.name} missing {c}")

    x = x[x["u"].isin(U)].copy()

    return x


def load_spatial():
    p = (
        V4 / "localization" /
        "prcc_display_u0.40.parquet"
    )

    x = pd.read_parquet(p)

    required = [
        "x",
        "y",
        "gw_distortion",
        "object_label",
    ]

    for c in required:
        if c not in x.columns:
            raise RuntimeError(f"{p.name} missing {c}")

    return x


# ---------------------------------------------------------------------
# Scale-dependent molecular audit
# ---------------------------------------------------------------------

def molecular_scale_audit(mol):
    """
    Construct one row per gene.

    Primary molecular signal:
      object-level rank association with GW distortion.

    Cell-level association is retained as an independent concordance
    check.

    No p-values or causal interpretation are assigned.
    """

    rows = []

    for gene, g in mol.groupby("gene"):
        g = g.sort_values("u")

        # Require complete four-scale observation.
        if len(g) != len(U):
            continue

        if not np.allclose(
            g["u"].to_numpy(float),
            U,
            atol=1e-9,
        ):
            continue

        obj = g[
            "object_spearman_sqrt_mass"
        ].to_numpy(float)

        cell = g[
            "cell_spearman"
        ].to_numpy(float)

        local = g[
            "high_minus_local_logexpr"
        ].to_numpy(float)

        obj_range = float(np.max(obj) - np.min(obj))
        cell_range = float(np.max(cell) - np.min(cell))

        obj_cross = bool(
            (np.max(obj) > 0) and
            (np.min(obj) < 0)
        )

        cell_cross = bool(
            (np.max(cell) > 0) and
            (np.min(cell) < 0)
        )

        # Agreement of the directions at each hierarchy coordinate.
        sign_agreement = float(
            np.mean(
                np.sign(obj) == np.sign(cell)
            )
        )

        # Scale-change coherence:
        # do the cell and object analyses move in the same direction
        # between adjacent scales?
        dobj = np.diff(obj)
        dcell = np.diff(cell)

        change_agreement = float(
            np.mean(
                np.sign(dobj) == np.sign(dcell)
            )
        )

        # Continuous prioritization.
        # This is not a biological score or significance statistic.
        # It simply rewards:
        #   - large object trajectory range
        #   - corresponding cell trajectory range
        #   - agreement between the two analyses
        ranking_score = (
            np.sqrt(
                max(obj_range, 0.0) *
                max(cell_range, 0.0)
            )
            *
            (0.5 + 0.5 * sign_agreement)
            *
            (0.5 + 0.5 * change_agreement)
        )

        rows.append({
            "gene": gene,

            "object_u032": obj[0],
            "object_u040": obj[1],
            "object_u048": obj[2],
            "object_u056": obj[3],

            "cell_u032": cell[0],
            "cell_u040": cell[1],
            "cell_u048": cell[2],
            "cell_u056": cell[3],

            "local_u032": local[0],
            "local_u040": local[1],
            "local_u048": local[2],
            "local_u056": local[3],

            "object_range": obj_range,
            "cell_range": cell_range,

            "object_zero_crossing": obj_cross,
            "cell_zero_crossing": cell_cross,
            "dual_zero_crossing": obj_cross and cell_cross,

            "sign_agreement_fraction": sign_agreement,
            "change_agreement_fraction": change_agreement,

            "ranking_score": ranking_score,
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        [
            "dual_zero_crossing",
            "ranking_score",
            "object_range",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return out


def select_genes(audit):
    """
    Selection is algorithmic.

    First prioritize genes with cell+object zero crossings.
    Fill remaining slots by scale-variation ranking.
    """

    reversing = audit[
        audit["dual_zero_crossing"]
    ].copy()

    trajectory = list(
        reversing.head(N_TRAJECTORY_GENES)["gene"]
    )

    if len(trajectory) < N_TRAJECTORY_GENES:
        for gene in audit["gene"]:
            if gene not in trajectory:
                trajectory.append(gene)

            if len(trajectory) == N_TRAJECTORY_GENES:
                break

    matrix = list(
        reversing.head(N_MATRIX_GENES)["gene"]
    )

    if len(matrix) < N_MATRIX_GENES:
        for gene in audit["gene"]:
            if gene not in matrix:
                matrix.append(gene)

            if len(matrix) == N_MATRIX_GENES:
                break

    return trajectory, matrix


# ---------------------------------------------------------------------
# Figure panels
# ---------------------------------------------------------------------

def panel_A(ax):
    ax.axis("off")
    panel_label(ax, "A")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    boxes = [
        (0.02, 0.59, 0.25, 0.25,
         "Nondiseased kidney\n541 genes"),
        (0.02, 0.16, 0.25, 0.25,
         "PRCC\n541 genes"),
        (0.37, 0.38, 0.25, 0.25,
         "Independent SUTRA\nhierarchies"),
        (0.72, 0.38, 0.25, 0.25,
         "Matched hierarchy\ncoordinate $u$\n+ GW"),
    ]

    for x, y, w, h, txt in boxes:
        r = plt.Rectangle(
            (x, y), w, h,
            fill=False,
            linewidth=1.0,
        )
        ax.add_patch(r)
        ax.text(
            x + w/2,
            y + h/2,
            txt,
            ha="center",
            va="center",
            fontsize=8.5,
        )

    # arrows
    for y in [0.715, 0.285]:
        ax.annotate(
            "",
            xy=(0.37, 0.505),
            xytext=(0.27, y),
            arrowprops=dict(
                arrowstyle="->",
                linewidth=0.9,
            ),
        )

    ax.annotate(
        "",
        xy=(0.72, 0.505),
        xytext=(0.62, 0.505),
        arrowprops=dict(
            arrowstyle="->",
            linewidth=0.9,
        ),
    )

    ax.text(
        0.50, 0.08,
        "Same measured gene panel; "
        "GW compares relational organization",
        ha="center",
        va="center",
        fontsize=8,
    )


def panel_B(ax, multi):
    panel_label(ax, "B")

    x = multi["u_target"].to_numpy(float)
    y = multi["best_gw_distance"].to_numpy(float)

    ax.plot(
        x,
        y,
        marker="o",
        linewidth=1.5,
        markersize=4,
    )

    ax.set_xlabel("Hierarchy coordinate $u$")
    ax.set_ylabel("$d_{GW}$")
    ax.set_title(
        "Relational displacement across scale",
        fontsize=9,
        pad=4,
    )

    ax.set_xlim(0.06, 0.58)

    # Mark largest sampled value without calling it a transition.
    imax = int(np.argmax(y))

    ax.annotate(
        "largest sampled\nGW displacement",
        xy=(x[imax], y[imax]),
        xytext=(x[imax] - 0.13, y[imax] + 0.025),
        fontsize=7.5,
        arrowprops=dict(
            arrowstyle="-",
            linewidth=0.8,
        ),
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_C(ax, spatial):
    panel_label(ax, "C")

    finite = np.isfinite(
        spatial["gw_distortion"].to_numpy(float)
    )

    s = spatial.loc[finite]

    vals = s["gw_distortion"].to_numpy(float)

    # Robust display ceiling only; underlying values are unchanged.
    vmax = float(np.quantile(vals, 0.995))

    sc = ax.scatter(
        s["x"],
        s["y"],
        c=np.minimum(vals, vmax),
        s=0.35,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(
        "PRCC relational displacement at $u=0.40$",
        fontsize=9,
        pad=4,
    )

    cb = plt.colorbar(
        sc,
        ax=ax,
        fraction=0.040,
        pad=0.015,
    )

    cb.set_label(
        "GW conditional distortion",
        fontsize=7.5,
    )

    cb.ax.tick_params(labelsize=6.5)


def panel_D(ax, mol):
    panel_label(ax, "D")

    g = mol[
        np.isclose(mol["u"], 0.40)
    ].copy()

    # Show object association against cell association.
    x = g["cell_spearman"].to_numpy(float)
    y = g[
        "object_spearman_sqrt_mass"
    ].to_numpy(float)

    ax.scatter(
        x,
        y,
        s=8,
        alpha=0.45,
        linewidths=0,
    )

    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)

    # Label strongest convergent genes at this scale.
    same = np.sign(x) == np.sign(y)

    score = np.where(
        same,
        np.sqrt(np.abs(x * y)),
        0.0,
    )

    order = np.argsort(score)[::-1][:10]

    for i in order:
        if score[i] <= 0:
            continue

        ax.text(
            x[i],
            y[i],
            str(g.iloc[i]["gene"]),
            fontsize=6.3,
            ha="left",
            va="bottom",
        )

    ax.set_xlabel(
        "Cell-level expression–distortion association"
    )
    ax.set_ylabel(
        "Object-level expression–distortion association"
    )

    ax.set_title(
        "Molecular identity of displaced architecture",
        fontsize=9,
        pad=4,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_E(ax, audit, genes):
    panel_label(ax, "E")

    for gene in genes:
        r = audit[audit["gene"] == gene].iloc[0]

        y = np.array([
            r["object_u032"],
            r["object_u040"],
            r["object_u048"],
            r["object_u056"],
        ])

        ax.plot(
            U,
            y,
            marker="o",
            linewidth=1.1,
            markersize=3,
            label=gene,
        )

    ax.axhline(0, linewidth=0.7)

    ax.set_xlim(0.30, 0.58)
    ax.set_xticks(U)

    ax.set_xlabel("Hierarchy coordinate $u$")
    ax.set_ylabel(
        "Object-level expression–GW association"
    )

    ax.set_title(
        "Molecular associations reorganize with scale",
        fontsize=9,
        pad=4,
    )

    ax.legend(
        frameon=False,
        fontsize=6.5,
        ncol=2,
        loc="best",
        handlelength=1.5,
        columnspacing=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def panel_F(ax, audit, genes):
    panel_label(ax, "F")

    rows = []

    for gene in genes:
        r = audit[audit["gene"] == gene].iloc[0]

        rows.append([
            r["object_u032"],
            r["object_u040"],
            r["object_u048"],
            r["object_u056"],
        ])

    M = np.asarray(rows, dtype=float)

    lim = float(np.max(np.abs(M)))
    lim = max(lim, 0.05)

    im = ax.imshow(
        M,
        aspect="auto",
        interpolation="nearest",
        norm=TwoSlopeNorm(
            vmin=-lim,
            vcenter=0.0,
            vmax=lim,
        ),
    )

    ax.set_yticks(np.arange(len(genes)))
    ax.set_yticklabels(genes, fontsize=6.5)

    ax.set_xticks(np.arange(len(U)))
    ax.set_xticklabels(
        [f"{u:.2f}" for u in U],
        fontsize=7,
    )

    ax.set_xlabel("Hierarchy coordinate $u$")

    ax.set_title(
        "Scale-resolved molecular program",
        fontsize=9,
        pad=4,
    )

    cb = plt.colorbar(
        im,
        ax=ax,
        fraction=0.040,
        pad=0.02,
    )

    cb.set_label(
        "Object-level association",
        fontsize=7.5,
    )

    cb.ax.tick_params(labelsize=6.5)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    print("=" * 100)
    print("SUTRA FIGURE 5 — FINAL V5")
    print("=" * 100)

    multi = load_multistart()
    mol = load_molecular()
    spatial = load_spatial()

    audit = molecular_scale_audit(mol)

    audit.to_csv(
        OUT / "molecular_scale_reorganization_audit.csv",
        index=False,
    )

    trajectory_genes, matrix_genes = select_genes(audit)

    print("\nTop scale-reorganizing genes")
    print(
        audit[
            [
                "gene",
                "object_range",
                "cell_range",
                "dual_zero_crossing",
                "sign_agreement_fraction",
                "change_agreement_fraction",
                "ranking_score",
            ]
        ].head(30).to_string(index=False)
    )

    print("\nPanel E genes:")
    print(", ".join(trajectory_genes))

    print("\nPanel F genes:")
    print(", ".join(matrix_genes))

    # Export exact panel data.
    audit[
        audit["gene"].isin(trajectory_genes)
    ].to_csv(
        OUT / "panel_E_gene_trajectories.csv",
        index=False,
    )

    audit[
        audit["gene"].isin(matrix_genes)
    ].to_csv(
        OUT / "panel_F_gene_matrix.csv",
        index=False,
    )

    multi.to_csv(
        OUT / "panel_B_gw_trajectory.csv",
        index=False,
    )

    # -------------------------------------------------------------
    # Figure layout
    # -------------------------------------------------------------

    fig = plt.figure(
        figsize=(12.0, 10.2),
        constrained_layout=False,
    )

    gs = fig.add_gridspec(
        3,
        2,
        left=0.065,
        right=0.975,
        bottom=0.065,
        top=0.975,
        wspace=0.30,
        hspace=0.38,
        height_ratios=[0.90, 1.15, 1.05],
    )

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])

    panel_A(axA)
    panel_B(axB, multi)
    panel_C(axC, spatial)
    panel_D(axD, mol)
    panel_E(axE, audit, trajectory_genes)
    panel_F(axF, audit, matrix_genes)

    pdf = OUT / "Figure5_SUTRA_disease_relational_reorganization.pdf"
    png = OUT / "Figure5_SUTRA_disease_relational_reorganization.png"

    fig.savefig(
        pdf,
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    provenance = {
        "version": "Figure5_final_v5",
        "primary_comparison":
            "nondiseased kidney versus PRCC",
        "shared_gene_panel": 541,
        "gw_multistart_input": str(V3),
        "gw_biology_input": str(V4),
        "hierarchy_coordinates_molecular":
            U.tolist(),
        "spatial_display_coordinate": 0.40,
        "gene_selection": (
            "algorithmic prioritization of genes with "
            "cell- and object-level zero crossings followed "
            "by continuous scale-variation ranking"
        ),
        "interpretation": (
            "Specimen-level association between molecular "
            "programs and scale-dependent relational "
            "architectural displacement. No causal or "
            "population-level disease inference."
        ),
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nWROTE")
    print(pdf)
    print(png)
    print("\nCOMPLETE")


if __name__ == "__main__":
    main()
