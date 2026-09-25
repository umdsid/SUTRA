#!/usr/bin/env python3

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(
    os.environ.get(
        "SUTRA_ROOT",
        str(Path.home() / "Desktop" / "SUTRA"),
    )
)
FIGROOT = Path(
    os.environ.get(
        "SUTRA_FIG1_OUT",
        str(ROOT / "figures" / "fig1"),
    )
)
OUT = FIGROOT / "panels"
OUT.mkdir(parents=True, exist_ok=True)

PNG = OUT / "panel_D.png"
PDF = OUT / "panel_D.pdf"

PURPLE = "#762a83"


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


fig, ax = plt.subplots(
    figsize=(4.5, 3.55),
    facecolor="white",
)

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_axis_off()


# ============================================================
# Formula only — no prose title
# ============================================================

ax.text(
    .50,
    .94,
    r"$B_{ij}=1/\sqrt{n^{\mathrm{boundary}}_{ij}}$",
    ha="center",
    va="top",
    fontsize=12,
    color=PURPLE,
)


# ============================================================
# Exact contact-count examples
# ============================================================

examples = [
    (.18, 1),
    (.50, 4),
    (.82, 9),
]


def draw_pair(x, y, n_contacts):

    unit_w = .105
    unit_h = .205
    gap = .040

    left_x = x - gap/2 - unit_w
    right_x = x + gap/2

    left = FancyBboxPatch(
        (left_x, y-unit_h/2),
        unit_w,
        unit_h,
        boxstyle="round,pad=.008,rounding_size=.030",
        facecolor=".90",
        edgecolor=".15",
        linewidth=1.1,
    )

    right = FancyBboxPatch(
        (right_x, y-unit_h/2),
        unit_w,
        unit_h,
        boxstyle="round,pad=.008,rounding_size=.030",
        facecolor=".78",
        edgecolor=".15",
        linewidth=1.1,
    )

    ax.add_patch(left)
    ax.add_patch(right)

    # Exact number of elementary contacts.
    #
    # Contacts are shown as small marks centered in the shared
    # gap. Their vertical locations are schematic; their COUNT
    # is exact.
    if n_contacts == 1:
        ys = np.array([y])
    else:
        ys = np.linspace(
            y - .074,
            y + .074,
            n_contacts,
        )

    for yy in ys:

        ax.plot(
            [
                x - .024,
                x + .024,
            ],
            [
                yy,
                yy,
            ],
            color=PURPLE,
            linewidth=2.2,
            solid_capstyle="round",
            zorder=5,
        )


for x, nb in examples:

    draw_pair(
        x,
        .61,
        nb,
    )

    B = 1 / np.sqrt(nb)

    ax.text(
        x,
        .405,
        rf"$n_b={nb}$",
        ha="center",
        va="top",
        fontsize=8.7,
    )

    ax.text(
        x,
        .335,
        rf"$B={B:.2f}$",
        ha="center",
        va="top",
        fontsize=8.7,
        color=PURPLE,
    )


# ============================================================
# Directional interpretation
# ============================================================

ax.annotate(
    "",
    xy=(.87, .18),
    xytext=(.13, .18),
    arrowprops=dict(
        arrowstyle="-|>",
        linewidth=1.8,
        color=".20",
    ),
)

ax.text(
    .50,
    .245,
    "more shared elementary contacts",
    ha="center",
    va="bottom",
    fontsize=7.4,
    weight="bold",
)

ax.text(
    .50,
    .105,
    "smaller structural penalty",
    ha="center",
    va="top",
    fontsize=7.4,
    weight="bold",
    color=PURPLE,
)


fig.savefig(
    PNG,
    dpi=600,
    bbox_inches="tight",
    pad_inches=.02,
)

fig.savefig(
    PDF,
    bbox_inches="tight",
    pad_inches=.02,
)

plt.close(fig)

print("D COMPLETE")
print(PNG)
