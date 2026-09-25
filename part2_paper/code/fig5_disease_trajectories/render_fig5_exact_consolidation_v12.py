#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize


ROOT = Path.home() / "Desktop" / "SUTRA"

SRC = (
    ROOT / "results" /
    "Exact_Consolidation_v12" /
    "prcc"
)

OUT = (
    ROOT / "results" /
    "Fig5_Disease_Trajectories" /
    "exact_consolidation_v12"
)
OUT.mkdir(parents=True, exist_ok=True)

DPI = 600


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


E = pd.read_parquet(
    SRC / "exact_level0_edge_consolidation.parquet"
)

meta = json.loads(
    (SRC / "summary.json").read_text()
)

segments = np.stack([
    E[["x_i", "y_i"]].to_numpy(float),
    E[["x_j", "y_j"]].to_numpy(float),
], axis=1)

resolved = E["consolidated"].to_numpy(bool)
persistent = E["persistent"].to_numpy(bool)

u = E["consolidation_u"].to_numpy(float)
umax = float(meta["hierarchy_endpoint_u"])

# ------------------------------------------------------------
# 1. Exact resolved consolidation geometry
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 6.6))

lc = LineCollection(
    segments[resolved],
    array=u[resolved],
    cmap="viridis",
    norm=Normalize(0, umax),
    linewidths=0.32,
    alpha=0.82,
    rasterized=True,
)

ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
ax.set_axis_off()

cb = fig.colorbar(
    lc,
    ax=ax,
    fraction=0.025,
    pad=0.008,
)

cb.set_label(
    "Consolidation coordinate, $u_c$",
    fontsize=9,
)
cb.ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.05)

save(
    fig,
    "prcc_exact_resolved_consolidation_geometry",
)

# ------------------------------------------------------------
# 2. Persistent interfaces alone
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 6.6))

lc = LineCollection(
    segments[persistent],
    linewidths=0.30,
    alpha=0.72,
    rasterized=True,
)

ax.add_collection(lc)
ax.autoscale()
ax.set_aspect("equal")
ax.set_axis_off()

fig.tight_layout(pad=0.05)

save(
    fig,
    "prcc_persistent_interfaces",
)

# ------------------------------------------------------------
# 3. Combined geometry:
#    resolved colored by u_c, persistent emphasized separately
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7.0, 6.6))

lc_res = LineCollection(
    segments[resolved],
    array=u[resolved],
    cmap="viridis",
    norm=Normalize(0, umax),
    linewidths=0.27,
    alpha=0.62,
    rasterized=True,
)

ax.add_collection(lc_res)

lc_per = LineCollection(
    segments[persistent],
    linewidths=0.42,
    alpha=0.78,
    rasterized=True,
)

ax.add_collection(lc_per)

ax.autoscale()
ax.set_aspect("equal")
ax.set_axis_off()

cb = fig.colorbar(
    lc_res,
    ax=ax,
    fraction=0.025,
    pad=0.008,
)

cb.set_label(
    "Consolidation coordinate, $u_c$",
    fontsize=9,
)

cb.ax.tick_params(labelsize=8)

fig.tight_layout(pad=0.05)

save(
    fig,
    "prcc_exact_consolidation_plus_persistent",
)

# ------------------------------------------------------------
# 4. Distribution: resolved u_c + persistent mass
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4.6, 3.5))

bins = np.linspace(
    0,
    umax,
    35,
)

ax.hist(
    u[resolved],
    bins=bins,
    density=True,
    histtype="step",
    linewidth=1.6,
)

ax.set_xlabel(
    "Consolidation coordinate, $u_c$"
)
ax.set_ylabel(
    "Density among consolidating interfaces"
)

ax.text(
    0.98,
    0.95,
    f"Persistent interfaces = "
    f"{persistent.mean():.1%}",
    ha="right",
    va="top",
    transform=ax.transAxes,
    fontsize=8,
)

fig.tight_layout()

save(
    fig,
    "prcc_exact_consolidation_distribution",
)

# ------------------------------------------------------------
# 5. Quantile classes for a less visually continuous version.
# ------------------------------------------------------------

q = np.quantile(
    u[resolved],
    [0.25, 0.50, 0.75],
)

cls = np.full(
    len(E),
    -1,
    dtype=int,
)

cls[
    resolved & (u <= q[0])
] = 0

cls[
    resolved &
    (u > q[0]) &
    (u <= q[1])
] = 1

cls[
    resolved &
    (u > q[1]) &
    (u <= q[2])
] = 2

cls[
    resolved & (u > q[2])
] = 3

fig, ax = plt.subplots(figsize=(7.0, 6.6))

for k in range(4):
    m = cls == k

    lc = LineCollection(
        segments[m],
        linewidths=0.32,
        alpha=0.72,
        rasterized=True,
    )

    ax.add_collection(lc)

# Persistent on top.
lc = LineCollection(
    segments[persistent],
    linewidths=0.48,
    alpha=0.82,
    rasterized=True,
)

ax.add_collection(lc)

ax.autoscale()
ax.set_aspect("equal")
ax.set_axis_off()

fig.tight_layout(pad=0.05)

save(
    fig,
    "prcc_consolidation_quantile_geometry",
)

# ------------------------------------------------------------
# Audit table
# ------------------------------------------------------------

audit = pd.DataFrame({
    "quantity": [
        "n_edges",
        "n_consolidated",
        "n_persistent",
        "persistent_fraction",
        "u_q25",
        "u_q50",
        "u_q75",
        "u_endpoint",
    ],
    "value": [
        len(E),
        int(resolved.sum()),
        int(persistent.sum()),
        float(persistent.mean()),
        float(q[0]),
        float(q[1]),
        float(q[2]),
        umax,
    ],
})

audit.to_csv(
    OUT / "render_audit.csv",
    index=False,
)

provenance = {
    "version": "fig5_exact_consolidation_v12",
    "source": str(SRC),
    "exact_event_level_consolidation": True,
    "checkpoint_interpolation": False,
    "persistent_interfaces_shown": True,
    "gw_overlay_added": False,
    "purpose": (
        "Standalone structural assets prior to final Fig. 5 "
        "composition and GW overlay."
    ),
}

(
    OUT / "provenance.json"
).write_text(
    json.dumps(provenance, indent=2)
)

print(audit.to_string(index=False))
print("\nWROTE:", OUT)
