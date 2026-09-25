#!/usr/bin/env python3

from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm, PowerNorm

ROOT = Path.home() / "Desktop" / "SUTRA"
FROOT = ROOT / "results" / "Fig5_Disease_Trajectories"

V3 = FROOT / "gw_multistart_v3"
V4 = FROOT / "gw_biology_v4"
V5 = FROOT / "final_v5"

OUT = FROOT / "panels_v6"
DATA = OUT / "panel_data"
OUT.mkdir(parents=True, exist_ok=True)
DATA.mkdir(exist_ok=True)

U = np.array([0.32, 0.40, 0.48, 0.56])
DISPLAY_U = 0.40

# ------------------------------------------------------------
# Global figure appearance
# ------------------------------------------------------------

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def clean(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(width=0.7, length=3)


def save(fig, stem, dpi=600):
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
        transparent=False,
    )
    fig.savefig(
        OUT / f"{stem}.png",
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        transparent=False,
    )
    plt.close(fig)


# ============================================================
# LOAD FROZEN INPUTS
# ============================================================

multi = pd.read_csv(V3 / "multistart_scale_summary.csv")

mol = pd.read_csv(
    V4 / "molecular" /
    "gene_integrated_scale_associations.csv"
)

spatial = pd.read_parquet(
    V4 / "localization" /
    "prcc_display_u0.40.parquet"
)

audit = pd.read_csv(
    V5 / "molecular_scale_reorganization_audit.csv"
)

panelE = pd.read_csv(
    V5 / "panel_E_gene_trajectories.csv"
)

panelF = pd.read_csv(
    V5 / "panel_F_gene_matrix.csv"
)

multi.to_csv(DATA / "GW_trajectory.csv", index=False)
panelE.to_csv(DATA / "reversing_program.csv", index=False)
panelF.to_csv(DATA / "molecular_scale_matrix.csv", index=False)


# ============================================================
# 1 — GW TRAJECTORY
# ============================================================

x = multi["u_target"].to_numpy(float)
y = multi["best_gw_distance"].to_numpy(float)

fig, ax = plt.subplots(figsize=(4.3, 3.15))

ax.plot(
    x, y,
    marker="o",
    markersize=4.5,
    linewidth=1.8,
)

# Spatial-display coordinate.
ax.axvline(
    DISPLAY_U,
    linewidth=0.9,
    linestyle="--",
    alpha=0.45,
)

i40 = int(np.argmin(np.abs(x - DISPLAY_U)))

ax.scatter(
    [x[i40]],
    [y[i40]],
    s=45,
    zorder=5,
)

ax.annotate(
    "$u=0.40$",
    xy=(x[i40], y[i40]),
    xytext=(8, 9),
    textcoords="offset points",
    fontsize=8,
)

ax.set_xlabel("Hierarchy coordinate $u$")
ax.set_ylabel("$d_{GW}$")
ax.set_title("Relational displacement across scale", pad=7)

ax.margins(x=0.04, y=0.10)
clean(ax)

fig.tight_layout()
save(fig, "GW_trajectory")


# ============================================================
# 2 — PRCC SPATIAL GW MAP
# ============================================================

finite = np.isfinite(spatial["gw_distortion"].to_numpy(float))
s = spatial.loc[finite].copy()

xx = s["x"].to_numpy(float)
yy = s["y"].to_numpy(float)
dd = s["gw_distortion"].to_numpy(float)

# Robust display scaling. Values remain continuous.
lo = float(np.quantile(dd, 0.02))
hi = float(np.quantile(dd, 0.995))

# Power transform expands the low/intermediate range while preserving
# ordering and continuous quantitative mapping.
norm = PowerNorm(
    gamma=0.58,
    vmin=lo,
    vmax=hi,
    clip=True,
)

# Full quantitative panel.
fig, ax = plt.subplots(figsize=(9.0, 3.3))

# Anatomical substrate.
ax.scatter(
    xx, yy,
    s=0.60,
    c="0.88",
    linewidths=0,
    rasterized=True,
)

sc = ax.scatter(
    xx, yy,
    c=dd,
    norm=norm,
    s=0.62,
    linewidths=0,
    rasterized=True,
)

ax.set_aspect("equal")
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_visible(False)

ax.set_title(
    "PRCC relational displacement at $u=0.40$",
    pad=5,
)

cb = fig.colorbar(
    sc,
    ax=ax,
    fraction=0.022,
    pad=0.012,
)

cb.set_label("GW conditional distortion")
cb.ax.tick_params(labelsize=7)

fig.tight_layout()
save(fig, "PRCC_GW_spatial")


# Raw spatial asset: no title, axes or colorbar.
# Large raster intended for later graphic composition.
fig, ax = plt.subplots(figsize=(12.0, 3.6))

ax.scatter(
    xx, yy,
    s=0.9,
    c="0.90",
    linewidths=0,
    rasterized=True,
)

ax.scatter(
    xx, yy,
    c=dd,
    norm=norm,
    s=0.92,
    linewidths=0,
    rasterized=True,
)

ax.set_aspect("equal")
ax.axis("off")

fig.subplots_adjust(
    left=0,
    right=1,
    bottom=0,
    top=1,
)

fig.savefig(
    OUT / "PRCC_GW_spatial_RAW.png",
    dpi=800,
    bbox_inches="tight",
    pad_inches=0,
    facecolor="white",
)

plt.close(fig)


# ============================================================
# 3 — MOLECULAR ASSOCIATION AT u=.40
# ============================================================

g = mol[np.isclose(mol["u"], DISPLAY_U)].copy()

xc = g["cell_spearman"].to_numpy(float)
yo = g["object_spearman_sqrt_mass"].to_numpy(float)

same = np.sign(xc) == np.sign(yo)

score = np.where(
    same,
    np.sqrt(np.abs(xc * yo)),
    0.0,
)

g["display_score"] = score

fig, ax = plt.subplots(figsize=(4.4, 3.7))

ax.scatter(
    xc,
    yo,
    s=10,
    alpha=0.30,
    linewidths=0,
)

# Emphasize strongest concordant associations.
top = g.nlargest(10, "display_score")

ax.scatter(
    top["cell_spearman"],
    top["object_spearman_sqrt_mass"],
    s=24,
    zorder=4,
)

for _, r in top.iterrows():
    ax.annotate(
        str(r["gene"]),
        (
            r["cell_spearman"],
            r["object_spearman_sqrt_mass"],
        ),
        xytext=(3, 3),
        textcoords="offset points",
        fontsize=7,
    )

ax.axhline(0, linewidth=0.65, alpha=0.5)
ax.axvline(0, linewidth=0.65, alpha=0.5)

ax.set_xlabel(
    "Cell-level expression–distortion association"
)
ax.set_ylabel(
    "Object-level expression–distortion association"
)

ax.set_title(
    "Molecular association with relational displacement",
    pad=7,
)

clean(ax)
fig.tight_layout()

g.to_csv(
    DATA / "molecular_association_u040.csv",
    index=False,
)

save(fig, "molecular_association_u040")


# ============================================================
# 4 — REVERSING PROGRAM
# ============================================================

# Use the algorithmically selected V5 trajectory genes.
genes = panelE["gene"].tolist()

rows = []
for gene in genes:
    r = audit[audit["gene"] == gene]

    if len(r) != 1:
        continue

    r = r.iloc[0]

    vals = np.array([
        r["object_u032"],
        r["object_u040"],
        r["object_u048"],
        r["object_u056"],
    ], dtype=float)

    rows.append((gene, vals))

M = np.vstack([v for _, v in rows])
genes = [g for g, _ in rows]

# Data-derived program centre and interquartile envelope.
program = np.median(M, axis=0)
q25 = np.quantile(M, 0.25, axis=0)
q75 = np.quantile(M, 0.75, axis=0)

fig, ax = plt.subplots(figsize=(5.2, 3.7))

# Individual genes deliberately subordinate.
for gene, vals in rows:
    ax.plot(
        U,
        vals,
        linewidth=0.85,
        alpha=0.36,
    )

# Interquartile program envelope.
ax.fill_between(
    U,
    q25,
    q75,
    alpha=0.13,
    linewidth=0,
)

# Dominant program trajectory.
ax.plot(
    U,
    program,
    linewidth=3.0,
    marker="o",
    markersize=5,
    zorder=10,
)

ax.axhline(
    0,
    linewidth=0.8,
    alpha=0.55,
)

ax.set_xticks(U)
ax.set_xlim(0.305, 0.575)

ax.set_xlabel("Hierarchy coordinate $u$")
ax.set_ylabel(
    "Expression–GW association"
)

ax.set_title(
    "Scale-dependent molecular reorganization",
    pad=7,
)

# Direct labels at the right side reduce legend clutter.
order = np.argsort(M[:, -1])

# Keep labels compact but preserve every selected gene.
ylabels = M[:, -1].copy()

# Simple collision relaxation.
idx = np.argsort(ylabels)
minsep = 0.018

for k in range(1, len(idx)):
    a = idx[k - 1]
    b = idx[k]

    if ylabels[b] - ylabels[a] < minsep:
        ylabels[b] = ylabels[a] + minsep

for i, gene in enumerate(genes):
    ax.annotate(
        gene,
        xy=(U[-1], M[i, -1]),
        xytext=(9, ylabels[i] - M[i, -1]),
        textcoords="offset points",
        fontsize=6.5,
        va="center",
    )

clean(ax)
fig.tight_layout()

program_df = pd.DataFrame({
    "u": U,
    "program_median": program,
    "program_q25": q25,
    "program_q75": q75,
})

program_df.to_csv(
    DATA / "reversing_program_summary.csv",
    index=False,
)

save(fig, "reversing_program")


# ============================================================
# 5 — SCALE-RESOLVED MOLECULAR MATRIX
# ============================================================

# Build trajectories from audit for the V5 matrix genes.
genes = panelF["gene"].tolist()

matrix_rows = []

for gene in genes:
    r = audit[audit["gene"] == gene]

    if len(r) != 1:
        continue

    r = r.iloc[0]

    vals = np.array([
        r["object_u032"],
        r["object_u040"],
        r["object_u048"],
        r["object_u056"],
    ], dtype=float)

    matrix_rows.append((gene, vals))

genes = [g for g, _ in matrix_rows]
M = np.vstack([v for _, v in matrix_rows])

# Data-driven ordering:
# primary = early-to-late change;
# secondary = trajectory range.
delta = M[:, -1] - M[:, 0]
rng = np.ptp(M, axis=1)

order = np.lexsort((-rng, delta))

M = M[order]
genes = [genes[i] for i in order]

lim = float(np.quantile(np.abs(M), 0.98))
lim = max(lim, 0.05)

fig_height = max(4.2, 0.235 * len(genes) + 1.2)

fig, ax = plt.subplots(
    figsize=(5.1, fig_height)
)

im = ax.imshow(
    M,
    aspect="auto",
    interpolation="nearest",
    norm=TwoSlopeNorm(
        vmin=-lim,
        vcenter=0,
        vmax=lim,
    ),
)

ax.set_xticks(np.arange(len(U)))
ax.set_xticklabels(
    [f"{u:.2f}" for u in U]
)

ax.set_yticks(np.arange(len(genes)))
ax.set_yticklabels(genes)

ax.set_xlabel("Hierarchy coordinate $u$")
ax.set_title(
    "Scale-resolved molecular associations",
    pad=7,
)

# Thin zero-crossing markers at the right margin.
for i, gene in enumerate(genes):
    vals = M[i]

    if np.max(vals) > 0 and np.min(vals) < 0:
        ax.text(
            len(U) - 0.43,
            i,
            "•",
            va="center",
            ha="center",
            fontsize=7,
        )

cb = fig.colorbar(
    im,
    ax=ax,
    fraction=0.035,
    pad=0.025,
)

cb.set_label(
    "Object-level expression–GW association"
)

cb.ax.tick_params(labelsize=7)

fig.tight_layout()

matrix_export = pd.DataFrame(
    M,
    columns=[f"u_{u:.2f}" for u in U],
)

matrix_export.insert(0, "gene", genes)

matrix_export.to_csv(
    DATA / "molecular_scale_matrix_ordered.csv",
    index=False,
)

save(fig, "molecular_scale_matrix")


# ============================================================
# PROVENANCE + PACKAGE
# ============================================================

provenance = {
    "version": "Fig5_panels_v6",
    "purpose": (
        "Standalone quantitative assets for subsequent "
        "Figure 5 visual composition."
    ),
    "panel_letters_included": False,
    "flowcharts_included": False,
    "primary_comparison":
        "nondiseased kidney versus PRCC",
    "shared_gene_panel": 541,
    "spatial_display_u": DISPLAY_U,
    "molecular_u": U.tolist(),
    "inputs": {
        "gw_multistart_v3": str(V3),
        "gw_biology_v4": str(V4),
        "final_v5_audit": str(V5),
    },
    "spatial_display": (
        "Continuous PRCC GW conditional distortion; "
        "PowerNorm used only for visualization."
    ),
    "molecular_program": (
        "Median and interquartile range across the "
        "algorithmically selected V5 reversing genes."
    ),
    "interpretation_limit": (
        "Specimen-level relational and molecular "
        "associations; no causal or population-level "
        "disease inference."
    ),
}

(OUT / "provenance.json").write_text(
    json.dumps(provenance, indent=2)
)

zip_base = FROOT / "Fig5_panels_v6"

archive = shutil.make_archive(
    str(zip_base),
    "zip",
    root_dir=OUT,
)

print("\n" + "=" * 80)
print("FIGURE 5 STANDALONE PANELS COMPLETE")
print("=" * 80)

for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(p.name)

print("\nZIP:")
print(archive)
