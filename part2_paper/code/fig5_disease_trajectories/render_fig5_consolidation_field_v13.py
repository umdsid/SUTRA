#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "Desktop" / "SUTRA"

SRC = ROOT / "results/Exact_Consolidation_v12/prcc"
OUT = ROOT / "results/Fig5_Disease_Trajectories/consolidation_field_v13"
OUT.mkdir(parents=True, exist_ok=True)

E = pd.read_parquet(SRC / "exact_level0_edge_consolidation.parquet")
META = json.loads((SRC / "summary.json").read_text())

NX = 190
NY = 190
MIN_EDGES = 3

x = E.x_mid.to_numpy(float)
y = E.y_mid.to_numpy(float)
u = E.consolidation_u.to_numpy(float)
resolved = E.consolidated.to_numpy(bool)
persistent = E.persistent.to_numpy(bool)

xmin, xmax = np.nanmin(x), np.nanmax(x)
ymin, ymax = np.nanmin(y), np.nanmax(y)

xe = np.linspace(xmin, xmax, NX + 1)
ye = np.linspace(ymin, ymax, NY + 1)

ix = np.clip(np.searchsorted(xe, x, side="right") - 1, 0, NX - 1)
iy = np.clip(np.searchsorted(ye, y, side="right") - 1, 0, NY - 1)

flat = iy * NX + ix
nbin = NX * NY

total_count = np.bincount(flat, minlength=nbin)
persistent_count = np.bincount(
    flat,
    weights=persistent.astype(float),
    minlength=nbin
)

P = np.full(nbin, np.nan)
ok = total_count >= MIN_EDGES
P[ok] = persistent_count[ok] / total_count[ok]

# Exact resolved-u median per spatial bin.
C = np.full(nbin, np.nan)

df = pd.DataFrame({
    "bin": flat[resolved],
    "u": u[resolved],
})

med = df.groupby("bin")["u"].median()
cnt = df.groupby("bin")["u"].size()

good = cnt[cnt >= MIN_EDGES].index.to_numpy(int)
C[good] = med.loc[good].to_numpy(float)

C = C.reshape(NY, NX)
P = P.reshape(NY, NX)
N = total_count.reshape(NY, NX)

xc = 0.5 * (xe[:-1] + xe[1:])
yc = 0.5 * (ye[:-1] + ye[1:])
X, Y = np.meshgrid(xc, yc)

# Tissue-support mask. No extrapolation outside occupied bins.
support = N >= MIN_EDGES

np.savez_compressed(
    OUT / "prcc_consolidation_field.npz",
    x_centers=xc,
    y_centers=yc,
    consolidation_median=C,
    persistent_fraction=P,
    edge_count=N,
    support=support,
)

# ---------------------------------------------------------
# Panel 1: consolidation field
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.6))

m = ax.pcolormesh(
    X, Y,
    np.ma.masked_invalid(C),
    shading="nearest",
    cmap="viridis",
    vmin=0,
    vmax=float(META["hierarchy_endpoint_u"]),
    rasterized=True,
)

levels = np.quantile(
    u[resolved],
    [0.20, 0.35, 0.50, 0.65, 0.80]
)

cs = ax.contour(
    X, Y,
    C,
    levels=levels,
    linewidths=0.7,
)

ax.clabel(
    cs,
    inline=True,
    fontsize=6,
    fmt="%.2f",
)

ax.set_aspect("equal")
ax.set_axis_off()

cb = fig.colorbar(m, ax=ax, fraction=0.025, pad=0.008)
cb.set_label("Local median consolidation coordinate, $u_c$", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.05)
fig.savefig(
    OUT / "prcc_consolidation_field.png",
    dpi=600, bbox_inches="tight", facecolor="white"
)
fig.savefig(
    OUT / "prcc_consolidation_field.pdf",
    bbox_inches="tight", facecolor="white"
)
plt.close(fig)

# ---------------------------------------------------------
# Panel 2: persistent-interface field
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.6))

m = ax.pcolormesh(
    X, Y,
    np.ma.masked_invalid(P),
    shading="nearest",
    vmin=0,
    vmax=1,
    rasterized=True,
)

ax.set_aspect("equal")
ax.set_axis_off()

cb = fig.colorbar(m, ax=ax, fraction=0.025, pad=0.008)
cb.set_label("Persistent-interface fraction", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.05)
fig.savefig(
    OUT / "prcc_persistent_interface_field.png",
    dpi=600, bbox_inches="tight", facecolor="white"
)
fig.savefig(
    OUT / "prcc_persistent_interface_field.pdf",
    bbox_inches="tight", facecolor="white"
)
plt.close(fig)

# ---------------------------------------------------------
# Panel 3: contours over persistence
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.2, 6.6))

m = ax.pcolormesh(
    X, Y,
    np.ma.masked_invalid(P),
    shading="nearest",
    vmin=0,
    vmax=1,
    alpha=0.80,
    rasterized=True,
)

cs = ax.contour(
    X, Y,
    C,
    levels=levels,
    linewidths=0.9,
)

ax.clabel(
    cs,
    inline=True,
    fontsize=6,
    fmt="%.2f",
)

ax.set_aspect("equal")
ax.set_axis_off()

cb = fig.colorbar(m, ax=ax, fraction=0.025, pad=0.008)
cb.set_label("Persistent-interface fraction", fontsize=9)
cb.ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.05)
fig.savefig(
    OUT / "prcc_consolidation_contours_on_persistence.png",
    dpi=600, bbox_inches="tight", facecolor="white"
)
fig.savefig(
    OUT / "prcc_consolidation_contours_on_persistence.pdf",
    bbox_inches="tight", facecolor="white"
)
plt.close(fig)

summary = {
    "version": "Fig5_consolidation_field_v13",
    "source_exact_v12": str(SRC),
    "grid": [NY, NX],
    "minimum_edges_per_bin": MIN_EDGES,
    "resolved_edges": int(resolved.sum()),
    "persistent_edges": int(persistent.sum()),
    "persistent_fraction": float(persistent.mean()),
    "contour_levels": levels.tolist(),
    "interpolation_used": False,
    "extrapolation_outside_support": False,
    "field_definition":
        "Median exact event-level consolidation coordinate among "
        "resolved Level-0 interfaces in each occupied spatial bin.",
    "persistence_definition":
        "Fraction of Level-0 interfaces in each occupied spatial bin "
        "that remain distinct at natural hierarchy exhaustion.",
}

(OUT / "provenance.json").write_text(json.dumps(summary, indent=2))

print(json.dumps(summary, indent=2))
print("WROTE:", OUT)
