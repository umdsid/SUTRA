#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
    }
)


SAMPLES = {
    "healthy_reference": {
        "organ": "Healthy brain",
        "short": "brain",
        "letter_pmf": "A",
        "letter_emergence": "C",
    },
    "nondiseased_kidney": {
        "organ": "Nondiseased kidney",
        "short": "kidney",
        "letter_pmf": "B",
        "letter_emergence": "D",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, out: Path, stem: str) -> None:
    fig.savefig(
        out / f"{stem}.png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    fig.savefig(
        out / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        facecolor="white",
    )
    plt.close(fig)


def clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def pmf_panel(src: Path, out: Path, sample: str, meta: dict) -> None:
    p = src / f"{sample}_collective_pmf_event_states.csv"
    d = pd.read_csv(p)

    nums = [
        c for c in d.columns
        if pd.api.types.is_numeric_dtype(d[c])
    ]

    mass = next(
        (c for c in d.columns if c.lower() in ("m", "mass", "object_mass")),
        nums[0],
    )
    prob = next(
        (c for c in d.columns if "prob" in c.lower() or "pmf" in c.lower()),
        nums[-1],
    )
    state = next(
        (
            c for c in d.columns
            if "state" in c.lower() or c.lower() in ("u", "stage")
        ),
        None,
    )

    fig, ax = plt.subplots(figsize=(4.35, 3.55))

    groups = (
        d.groupby(state, sort=False)
        if state is not None
        else [("observed", d)]
    )

    for label, g in groups:
        ax.scatter(
            g[mass],
            g[prob],
            s=16,
            alpha=0.58,
            label=str(label),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Collective-object mass, m", fontsize=11)
    ax.set_ylabel("P(m | m ≥ 2)", fontsize=11)

    if state is not None and d[state].nunique() <= 6:
        ax.legend(frameon=False, fontsize=8, loc="upper right")

    clean(ax)
    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        bottom=0.19,
        top=0.97,
    )

    save(fig, out, f"Fig3{meta['letter_pmf']}")


def emergence_panel(
    src: Path,
    out: Path,
    sample: str,
    meta: dict,
) -> None:
    obs = pd.read_csv(
        src / f"{sample}_collective_trajectory.csv"
    )
    nul = pd.read_csv(
        src / f"{sample}_aggregation_null.csv"
    )

    required_obs = {"u", "phi"}
    required_null = {
        "u",
        "null_lo",
        "null_median",
        "null_hi",
    }

    assert required_obs.issubset(obs.columns)
    assert required_null.issubset(nul.columns)

    fig, ax = plt.subplots(figsize=(4.35, 3.55))

    ax.fill_between(
        nul["u"],
        nul["null_lo"],
        nul["null_hi"],
        alpha=0.18,
        linewidth=0,
        label="Aggregation-only null",
    )

    ax.plot(
        nul["u"],
        nul["null_median"],
        lw=1.2,
    )

    ax.plot(
        obs["u"],
        obs["phi"],
        lw=1.8,
        label="Observed",
    )

    ax.set_xlabel("Removed fraction, u", fontsize=11)
    ax.set_ylabel("Collective fraction, Φ(u)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    clean(ax)
    fig.subplots_adjust(
        left=0.18,
        right=0.97,
        bottom=0.19,
        top=0.97,
    )

    save(fig, out, f"Fig3{meta['letter_emergence']}")


def spatial_panel(
    src: Path,
    out: Path,
    sample: str,
    meta: dict,
) -> None:
    cells = pd.read_parquet(
        src / f"{sample}_level0_coordinates.parquet"
    )

    recipient = pd.read_csv(
        src / f"{sample}_peak_recipient_ids.csv"
    )
    incoming = pd.read_csv(
        src / f"{sample}_peak_incoming_ids.csv"
    )

    for d in (recipient, incoming):
        assert "level0_index" in d.columns

    rec = pd.to_numeric(
        recipient["level0_index"],
        errors="raise",
    ).astype(np.int64)

    inc = pd.to_numeric(
        incoming["level0_index"],
        errors="raise",
    ).astype(np.int64)

    idx = np.unique(
        np.concatenate(
            [
                rec.to_numpy(),
                inc.to_numpy(),
            ]
        )
    )

    assert idx.size > 0
    assert idx.min() >= 0
    assert idx.max() < len(cells)

    mask = np.zeros(len(cells), dtype=bool)
    mask[idx] = True

    fig, ax = plt.subplots(figsize=(6.2, 3.65))

    ax.scatter(
        cells["x"],
        cells["y"],
        s=0.45,
        alpha=0.14,
        linewidths=0,
        rasterized=True,
    )

    ax.scatter(
        cells.loc[mask, "x"],
        cells.loc[mask, "y"],
        s=4.0,
        alpha=0.95,
        linewidths=0,
        rasterized=True,
    )

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.03,
        top=0.97,
    )

    save(fig, out, f"Fig3E_{meta['short']}")


def molecular_panel(
    src: Path,
    out: Path,
    sample: str,
    meta: dict,
) -> None:
    d = pd.read_csv(
        src / f"{sample}_peak_molecular_contrast.csv"
    )

    assert {"gene", "effect_log1p_norm"}.issubset(d.columns)

    q = (
        d.assign(_abs=d["effect_log1p_norm"].abs())
        .nlargest(min(15, len(d)), "_abs")
        .sort_values("effect_log1p_norm")
    )

    fig, ax = plt.subplots(figsize=(5.0, 4.15))

    ax.barh(
        q["gene"].astype(str),
        q["effect_log1p_norm"],
    )

    ax.axvline(0, lw=0.8)

    ax.set_xlabel(
        "Incoming − recipient expression contrast",
        fontsize=10,
    )

    clean(ax)
    ax.spines["left"].set_visible(True)

    fig.subplots_adjust(
        left=0.22,
        right=0.98,
        bottom=0.17,
        top=0.97,
    )

    save(fig, out, f"Fig3F_{meta['short']}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--source-data",
        default=None,
        help=(
            "Frozen Figure 3 source-data directory. "
            "Defaults to part2_paper/main/fig3/source_data."
        ),
    )

    ap.add_argument(
        "--output-root",
        required=True,
    )

    a = ap.parse_args()

    script = Path(__file__).resolve()

    repo = script.parents[4]

    src = (
        Path(a.source_data).expanduser().resolve()
        if a.source_data
        else repo / "part2_paper" / "main" / "fig3" / "source_data"
    )

    out = Path(a.output_root).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    required = []

    for sample in SAMPLES:
        required.extend(
            [
                src / f"{sample}_collective_pmf_event_states.csv",
                src / f"{sample}_collective_trajectory.csv",
                src / f"{sample}_aggregation_null.csv",
                src / f"{sample}_peak_recipient_ids.csv",
                src / f"{sample}_peak_incoming_ids.csv",
                src / f"{sample}_peak_molecular_contrast.csv",
                src / f"{sample}_level0_coordinates.parquet",
            ]
        )

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing frozen Figure 3 inputs:\n" +
            "\n".join(missing)
        )

    for sample, meta in SAMPLES.items():
        pmf_panel(src, out, sample, meta)
        emergence_panel(src, out, sample, meta)
        spatial_panel(src, out, sample, meta)
        molecular_panel(src, out, sample, meta)

    expected = {
        "Fig3A.png",
        "Fig3A.pdf",
        "Fig3B.png",
        "Fig3B.pdf",
        "Fig3C.png",
        "Fig3C.pdf",
        "Fig3D.png",
        "Fig3D.pdf",
        "Fig3E_brain.png",
        "Fig3E_brain.pdf",
        "Fig3E_kidney.png",
        "Fig3E_kidney.pdf",
        "Fig3F_brain.png",
        "Fig3F_brain.pdf",
        "Fig3F_kidney.png",
        "Fig3F_kidney.pdf",
    }

    produced = {
        p.name
        for p in out.iterdir()
        if p.suffix.lower() in {".png", ".pdf"}
    }

    if produced != expected:
        raise RuntimeError(
            f"Unexpected output inventory.\n"
            f"Expected: {sorted(expected)}\n"
            f"Produced: {sorted(produced)}"
        )

    provenance = {
        "figure": "Main Figure 3",
        "mode": "standalone_panels",
        "assembled_figure": False,
        "scientific_recalculation": False,
        "hierarchy_rerun": False,
        "source_data": str(src),
        "panels": {
            "A": "healthy-reference collective-object mass distribution",
            "B": "nondiseased-kidney collective-object mass distribution",
            "C": "healthy-reference observed versus aggregation-only null",
            "D": "nondiseased-kidney observed versus aggregation-only null",
            "E_brain": "healthy-reference dominant recruitment event",
            "E_kidney": "nondiseased-kidney dominant recruitment event",
            "F_brain": "healthy-reference molecular recruitment contrast",
            "F_kidney": "nondiseased-kidney molecular recruitment contrast",
        },
        "input_sha256": {
            p.name: sha256(p)
            for p in required
        },
    }

    (out / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )

    print("WROTE", out)

    for p in sorted(out.iterdir()):
        print(p.name)


if __name__ == "__main__":
    main()
