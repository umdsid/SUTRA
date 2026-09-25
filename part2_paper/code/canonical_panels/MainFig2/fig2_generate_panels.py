#!/usr/bin/env python3
"""
SUTRA Main Figure 2 — standalone paper-panel renderer.

This is a Part-2 renderer. It performs no hierarchy reconstruction and no
scientific recalculation.

Frozen inputs
-------------
fig2_crossorgan/
    brain/panels/panel_A.png
    brain/panels/panel_B.png
    brain/panels/panel_C.png
    brain/source_data/panel_D_cost_landscape.csv
    brain/source_data/panel_F_interface_exhaustion.csv

    kidney/panels/panel_A.png
    kidney/panels/panel_B.png
    kidney/panels/panel_C.png
    kidney/source_data/panel_D_cost_landscape.csv
    kidney/source_data/panel_F_interface_exhaustion.csv

fig2_kron_feshbach_dense/
    dense_kron_feshbach_summary.csv

Final standalone mapping
------------------------
A  brain whole-tissue hierarchy
B  brain local hierarchy
C  brain candidate/admissible interface evolution
D  kidney whole-tissue hierarchy
E  kidney local hierarchy
F  kidney candidate/admissible interface evolution
G  brain four-diagnostic trajectory block
H  kidney four-diagnostic trajectory block

For C/F only, neutral residual candidate interfaces in the terminal state are
display-recolored violet. This creates no new edge class: the frozen source
tables show zero admissible interfaces at terminal.

No assembled Figure 2 is produced.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


PURPLE = (112, 82, 190)

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def resolve_root(path: Path, expected: str) -> Path:
    path = Path(path).expanduser().resolve()

    if path.is_dir():
        if (path / expected).exists():
            return path / expected

        if path.name == expected:
            return path

        if expected == "fig2_crossorgan" and (path / "brain").exists():
            return path

        if (
            expected == "fig2_kron_feshbach_dense"
            and (path / "dense_kron_feshbach_summary.csv").exists()
        ):
            return path

        raise FileNotFoundError(
            f"Cannot find {expected} under {path}"
        )

    if path.suffix.lower() == ".zip":
        td = Path(tempfile.mkdtemp(prefix="sutra_fig2_"))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(td)

        hits = list(td.rglob(expected))
        if hits:
            return hits[0]

        raise FileNotFoundError(
            f"{expected} not found in {path}"
        )

    raise FileNotFoundError(path)


def crop_white(im: Image.Image, pad: int = 10, thresh: int = 252) -> Image.Image:
    a = np.asarray(im.convert("RGB"))
    mask = np.any(a < thresh, axis=2)

    if not np.any(mask):
        return im.convert("RGB")

    y, x = np.where(mask)

    return im.crop(
        (
            max(0, int(x.min()) - pad),
            max(0, int(y.min()) - pad),
            min(im.width, int(x.max()) + pad + 1),
            min(im.height, int(y.max()) + pad + 1),
        )
    )


def recolor_terminal_candidates(
    im: Image.Image,
    frac_start: float = 0.755,
) -> Image.Image:
    """
    Display-only terminal-state recoloring inherited from the approved v8
    Figure-2 compositor.

    Neutral residual candidate edges become violet in the terminal quarter.
    At terminal, frozen source data report zero admissible interfaces.
    """
    a = np.asarray(im.convert("RGB")).copy()

    h, w, _ = a.shape
    x0 = int(frac_start * w)

    sub = a[:, x0:, :]
    mx = sub.max(axis=2)
    mn = sub.min(axis=2)
    mean = sub.mean(axis=2)

    neutral = (mx - mn < 10) & (mean > 145) & (mean < 245)

    yy = np.arange(h)[:, None]
    body = (yy > 0.10 * h) & (yy < 0.91 * h)

    sub[neutral & body] = PURPLE
    a[:, x0:, :] = sub

    return Image.fromarray(a)


def load_spatial_panel(path: Path, recolor_terminal: bool = False) -> Image.Image:
    im = Image.open(path).convert("RGB")

    if recolor_terminal:
        im = recolor_terminal_candidates(im)

    return crop_white(im)


def save_raster_panel(
    im: Image.Image,
    outdir: Path,
    letter: str,
) -> None:
    """
    Export a frozen spatial panel as standalone PNG/PDF without stretching it.
    """
    w, h = im.size

    fig_w = 10.0
    fig_h = fig_w * h / w

    fig = plt.figure(
        figsize=(fig_w, fig_h),
        facecolor="white",
    )

    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(im, interpolation="lanczos")
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    png = outdir / f"panel_{letter}.png"
    pdf = outdir / f"panel_{letter}.pdf"

    fig.savefig(
        png,
        dpi=600,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0,
    )

    fig.savefig(
        pdf,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0,
    )

    plt.close(fig)

    print(f"panel {letter}: {png}")


def clean(ax, ylabel: str) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        labelsize=6.2,
        length=2,
        pad=1.2,
    )

    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])

    ax.set_ylabel(
        ylabel,
        fontsize=6.7,
        labelpad=1.2,
    )


def cost_margin(ax, csv: Path) -> None:
    d = pd.read_csv(csv)

    u = d["u"].to_numpy(float)
    tau = np.maximum(
        d["threshold"].to_numpy(float),
        1e-12,
    )

    def z(column: str) -> np.ndarray:
        return np.log10(
            np.maximum(
                d[column].to_numpy(float),
                1e-12,
            )
            / tau
        )

    ax.fill_between(
        u,
        z("q10"),
        z("q90"),
        alpha=0.18,
    )

    ax.plot(
        u,
        z("median"),
        lw=1.45,
    )

    ax.axhline(
        0,
        ls="--",
        lw=0.8,
    )

    clean(
        ax,
        r"$\log_{10}(C/\tau)$",
    )


def admissible_fraction(ax, csv: Path) -> None:
    d = pd.read_csv(csv)

    ax.plot(
        d["u"],
        100 * d["admissible_fraction"],
        lw=1.45,
    )

    ax.axhline(
        0,
        lw=0.6,
    )

    clean(
        ax,
        "Admissible (%)",
    )

    ax.set_ylim(bottom=-2)


def kron_metric(
    ax,
    kron: pd.DataFrame,
    organ: str,
    metric: str,
    ylabel: str,
    zero: bool = False,
) -> None:
    d = (
        kron[kron["organ"] == organ]
        .sort_values("u_mid")
    )

    x = d["u_mid"].to_numpy(float)

    med = d[f"{metric}_median"].to_numpy(float)
    q1 = d[f"{metric}_q25"].to_numpy(float)
    q3 = d[f"{metric}_q75"].to_numpy(float)

    ax.fill_between(
        x,
        q1,
        q3,
        alpha=0.18,
    )

    ax.plot(
        x,
        med,
        lw=1.45,
    )

    if zero:
        ax.axhline(
            0,
            ls="--",
            lw=0.8,
        )

    clean(
        ax,
        ylabel,
    )


def diagnostic_panel(
    fig2_root: Path,
    kron: pd.DataFrame,
    organ: str,
    outdir: Path,
    letter: str,
) -> None:
    """
    Render the approved four-diagnostic organ block as one standalone panel.
    """
    fig = plt.figure(
        figsize=(8.2, 2.25),
        facecolor="white",
    )

    left = 0.055
    right = 0.985
    bottom = 0.22
    top = 0.80
    gap = 0.035

    total_w = right - left
    plot_w = (total_w - 3 * gap) / 4

    axes = []

    for i in range(4):
        x = left + i * (plot_w + gap)
        axes.append(
            fig.add_axes(
                [
                    x,
                    bottom,
                    plot_w,
                    top - bottom,
                ]
            )
        )

    cost_margin(
        axes[0],
        fig2_root
        / organ
        / "source_data"
        / "panel_D_cost_landscape.csv",
    )

    admissible_fraction(
        axes[1],
        fig2_root
        / organ
        / "source_data"
        / "panel_F_interface_exhaustion.csv",
    )

    kron_metric(
        axes[2],
        kron,
        organ,
        "contextual_feshbach_gain",
        r"$\Delta_{\rm SF}$",
        zero=True,
    )

    kron_metric(
        axes[3],
        kron,
        organ,
        "contextual_resistance_error",
        r"$\epsilon_R$",
    )

    titles = [
        "Cost margin",
        "Admissible fraction",
        "Feshbach gain",
        "Resistance error",
    ]

    for ax, title in zip(axes, titles):
        pos = ax.get_position()

        fig.text(
            pos.x0 + pos.width / 2,
            top + 0.045,
            title,
            ha="center",
            va="bottom",
            fontsize=7.2,
            fontweight="bold",
        )

    fig.text(
        0.52,
        0.055,
        r"Hierarchy progress, $u$",
        ha="center",
        va="center",
        fontsize=8.1,
    )

    png = outdir / f"panel_{letter}.png"
    pdf = outdir / f"panel_{letter}.pdf"

    fig.savefig(
        png,
        dpi=600,
        facecolor="white",
    )

    fig.savefig(
        pdf,
        facecolor="white",
    )

    plt.close(fig)

    print(f"panel {letter}: {png}")


def validate_inputs(
    fig2_root: Path,
    kron_root: Path,
) -> None:
    required = [
        fig2_root / "brain" / "panels" / "panel_A.png",
        fig2_root / "brain" / "panels" / "panel_B.png",
        fig2_root / "brain" / "panels" / "panel_C.png",
        fig2_root / "kidney" / "panels" / "panel_A.png",
        fig2_root / "kidney" / "panels" / "panel_B.png",
        fig2_root / "kidney" / "panels" / "panel_C.png",
        fig2_root
        / "brain"
        / "source_data"
        / "panel_D_cost_landscape.csv",
        fig2_root
        / "brain"
        / "source_data"
        / "panel_F_interface_exhaustion.csv",
        fig2_root
        / "kidney"
        / "source_data"
        / "panel_D_cost_landscape.csv",
        fig2_root
        / "kidney"
        / "source_data"
        / "panel_F_interface_exhaustion.csv",
        kron_root / "dense_kron_feshbach_summary.csv",
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing frozen Figure-2 input(s):\n"
            + "\n".join(missing)
        )


def validate_terminal_state(
    fig2_root: Path,
) -> dict:
    audit = {}

    for organ in ("brain", "kidney"):
        p = (
            fig2_root
            / organ
            / "source_data"
            / "panel_F_interface_exhaustion.csv"
        )

        d = pd.read_csv(p).sort_values("u")
        terminal = d.iloc[-1]

        n_candidates = int(
            terminal["candidate_interfaces"]
        )

        n_admissible = int(
            terminal["admissible_interfaces"]
        )

        if n_admissible != 0:
            raise RuntimeError(
                f"{organ}: terminal admissible interfaces "
                f"must equal 0, got {n_admissible}"
            )

        if n_candidates <= 0:
            raise RuntimeError(
                f"{organ}: expected residual terminal "
                "candidate interfaces"
            )

        audit[organ] = {
            "terminal_u": float(terminal["u"]),
            "candidate_interfaces": n_candidates,
            "admissible_interfaces": n_admissible,
        }

    return audit


def build(
    fig2_root: Path,
    kron_root: Path,
    outdir: Path,
) -> None:
    outdir.mkdir(
        parents=True,
        exist_ok=True,
    )

    validate_inputs(
        fig2_root,
        kron_root,
    )

    terminal_audit = validate_terminal_state(
        fig2_root,
    )

    kron = pd.read_csv(
        kron_root
        / "dense_kron_feshbach_summary.csv"
    )

    mapping = {
        "A": ("brain", "A", False),
        "B": ("brain", "B", False),
        "C": ("brain", "C", True),
        "D": ("kidney", "A", False),
        "E": ("kidney", "B", False),
        "F": ("kidney", "C", True),
    }

    for letter, (
        organ,
        source_letter,
        recolor,
    ) in mapping.items():
        im = load_spatial_panel(
            fig2_root
            / organ
            / "panels"
            / f"panel_{source_letter}.png",
            recolor_terminal=recolor,
        )

        save_raster_panel(
            im,
            outdir,
            letter,
        )

    diagnostic_panel(
        fig2_root,
        kron,
        "brain",
        outdir,
        "G",
    )

    diagnostic_panel(
        fig2_root,
        kron,
        "kidney",
        outdir,
        "H",
    )

    manifest = {
        "figure": "Main Figure 2",
        "mode": "standalone_panels",
        "assembled_figure": False,
        "scientific_recalculation": False,
        "panels": {
            "A": "brain whole-tissue hierarchy",
            "B": "brain local hierarchy",
            "C": (
                "brain candidate/admissible "
                "interface evolution"
            ),
            "D": "kidney whole-tissue hierarchy",
            "E": "kidney local hierarchy",
            "F": (
                "kidney candidate/admissible "
                "interface evolution"
            ),
            "G": "brain four-diagnostic trajectory block",
            "H": "kidney four-diagnostic trajectory block",
        },
        "terminal_display": (
            "neutral residual candidate interfaces in "
            "terminal C/F are display-recolored violet; "
            "no new edge class is created"
        ),
        "terminal_audit": terminal_audit,
        "quantitative_source": (
            "dense_kron_feshbach_summary.csv "
            "plus organ-specific frozen Figure-2 tables"
        ),
    }

    (
        outdir
        / "Figure2_standalone_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print()
    print("Figure 2 standalone panels complete.")
    print(
        json.dumps(
            terminal_audit,
            indent=2,
        )
    )


def main() -> None:
    default_root = Path(
        os.environ.get(
            "SUTRA_ROOT",
            str(
                Path.home()
                / "Desktop"
                / "SUTRA"
            ),
        )
    )

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fig2-source",
        default=os.environ.get(
            "SUTRA_FIG2_SOURCE",
            str(
                default_root
                / "figures"
                / "fig2_crossorgan"
            ),
        ),
    )

    parser.add_argument(
        "--kron-source",
        default=os.environ.get(
            "SUTRA_FIG2_KRON_SOURCE",
            str(
                default_root
                / "figures"
                / "fig2_kron_feshbach_dense"
            ),
        ),
    )

    parser.add_argument(
        "--outdir",
        default=os.environ.get(
            "SUTRA_FIG2_OUT",
            str(
                Path.cwd()
                / "main_fig2"
            ),
        ),
    )

    args = parser.parse_args()

    fig2_root = resolve_root(
        Path(args.fig2_source),
        "fig2_crossorgan",
    )

    kron_root = resolve_root(
        Path(args.kron_source),
        "fig2_kron_feshbach_dense",
    )

    build(
        fig2_root,
        kron_root,
        Path(args.outdir).expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
