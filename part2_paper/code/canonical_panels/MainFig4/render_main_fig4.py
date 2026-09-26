from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize
import pyarrow.parquet as pq


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

FC = {
    "threshold": "#2878B5",
    "rescue": "#3A923A",
    "schedule": "#D88419",
    "frontier": "#7953A6",
    "joint": "#B0447A",
}

METRICS = [
    ("VI_bits_vs_ref", "VI (bits)", "Blues"),
    ("node_relerr_vs_ref", "|ΔN| / Nref", "Greens"),
    ("top1_absdiff_vs_ref", "|Δ top-1% mass|", "Oranges"),
    ("gini_absdiff_vs_ref", "|Δ Gini|", "Reds"),
]


def rp(path, cols=None):
    return pq.ParquetFile(path).read(
        columns=cols,
        use_threads=False,
    ).to_pandas()


def canon(a):
    return np.unique(np.asarray(a), return_inverse=True)[1]


def xycols(df):
    lc = {c.lower(): c for c in df}
    for a, b in [
        ("x_centroid", "y_centroid"),
        ("centroid_x", "centroid_y"),
        ("x", "y"),
        ("x_location", "y_location"),
    ]:
        if a in lc and b in lc:
            return lc[a], lc[b]
    raise RuntimeError(
        "No recognized x/y columns: " + str(list(df.columns))
    )


def match(ref, pert):
    r = canon(ref)
    p = canon(pert)

    nr = int(r.max() + 1)
    u, c = np.unique(
        p.astype(np.int64) * nr + r,
        return_counts=True,
    )

    best = {}
    for z, n in zip(u, c):
        pp = int(z // nr)
        rr = int(z % nr)
        if pp not in best or n > best[pp][0]:
            best[pp] = (int(n), rr)

    mp = {k: v[1] for k, v in best.items()}
    ch = np.array(
        [mp.get(int(pp), -1) != int(rr) for pp, rr in zip(p, r)]
    )
    return r, p, mp, ch


def crop(x, y, ch, bins=18, frac=0.3):
    xe = np.linspace(x.min(), x.max(), bins + 1)
    ye = np.linspace(y.min(), y.max(), bins + 1)

    H = np.zeros((bins, bins))
    C = np.zeros_like(H)

    ix = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)

    for a, b, q in zip(ix, iy, ch):
        H[b, a] += 1
        C[b, a] += q

    S = C / np.maximum(H, 1)
    S[H < 15] = -1

    by, bx = np.unravel_index(np.argmax(S), S.shape)

    cx = (xe[bx] + xe[bx + 1]) / 2
    cy = (ye[by] + ye[by + 1]) / 2

    wx = (x.max() - x.min()) * frac
    wy = (y.max() - y.min()) * frac

    return (
        cx - wx / 2,
        cx + wx / 2,
        cy - wy / 2,
        cy + wy / 2,
    )


def colors(ref, pert, mask):
    r, p, mp, ch = match(ref, pert)

    vals, cnt = np.unique(r[mask], return_counts=True)
    vals = vals[np.argsort(cnt)[::-1]][:14]

    cm = plt.get_cmap("tab20")
    rc = {int(v): cm(i % 20) for i, v in enumerate(vals)}
    neutral = (0.84, 0.84, 0.84, 1)

    a = np.array([rc.get(int(v), neutral) for v in r])
    b = np.array(
        [rc.get(mp.get(int(v), -1), neutral) for v in p]
    )

    return a, b, ch


def save(fig, out, stem):
    png = out / f"{stem}.png"
    pdf = out / f"{stem}.pdf"

    fig.savefig(
        png,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        pdf,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print("WROTE", png)
    print("WROTE", pdf)


def draw_panel_a(ax, cells, part, meta):
    ax.axis("off")

    ax.text(
        0.032,
        1.018,
        "Robustness mapping on the same measured tissue",
        fontsize=11.5,
        fontweight="bold",
        transform=ax.transAxes,
        va="top",
    )

    xc, yc = xycols(cells)
    x = cells[xc].to_numpy(float)
    y = cells[yc].to_numpy(float)

    ref = part.reference_block.to_numpy()
    per = part.perturbed_block.to_numpy()

    if len(x) != len(ref):
        raise RuntimeError(
            f"coordinate/partition row mismatch: "
            f"{len(x)} coordinates versus {len(ref)} partitions"
        )

    r, p, mp, ch = match(ref, per)
    box = crop(x, y, ch)

    m = (
        (x >= box[0])
        & (x <= box[1])
        & (y >= box[2])
        & (y <= box[3])
    )

    cr, cpcol, ch = colors(ref, per, m)

    a0 = ax.inset_axes([0.035, 0.19, 0.18, 0.62])
    a0.scatter(
        x[m],
        y[m],
        s=4,
        c=".28",
        lw=0,
        rasterized=True,
    )
    a0.axis("off")
    a0.set_aspect("equal")

    ax.text(
        0.125,
        0.84,
        "fixed Level-0 tissue",
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.125,
        0.13,
        "same cells • same coordinates • same relational inputs",
        ha="center",
        fontsize=6.8,
        transform=ax.transAxes,
    )

    ax.add_patch(
        FancyArrowPatch(
            (0.22, 0.5),
            (0.275, 0.5),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1,
            color=".3",
        )
    )

    bx = 0.285

    ax.add_patch(
        FancyBboxPatch(
            (bx - 0.015, 0.25),
            0.175,
            0.5,
            transform=ax.transAxes,
            boxstyle="round,pad=.008,rounding_size=.01",
            fc="white",
            ec=".7",
            lw=0.8,
        )
    )

    ax.text(
        bx + 0.072,
        0.68,
        "controlled rule\nperturbation",
        ha="center",
        fontsize=8.1,
        fontweight="bold",
        transform=ax.transAxes,
    )

    yy = 0.54
    for fam in [
        "threshold",
        "rescue",
        "schedule",
        "frontier",
        "joint",
    ]:
        ax.add_patch(
            Rectangle(
                (bx + 0.025, yy - 0.012),
                0.014,
                0.014,
                transform=ax.transAxes,
                fc=FC[fam],
                ec="none",
            )
        )
        ax.text(
            bx + 0.047,
            yy,
            fam,
            fontsize=6.9,
            va="center",
            transform=ax.transAxes,
        )
        yy -= 0.055

    ax.text(
        bx + 0.072,
        0.29,
        r"$\theta_{\rm ref}\rightarrow\theta_p$",
        ha="center",
        fontsize=8.2,
        transform=ax.transAxes,
    )

    ax.add_patch(
        FancyArrowPatch(
            (0.44, 0.5),
            (0.485, 0.5),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1,
            color=".3",
        )
    )

    positions = [
        (0.49, "reference", cr),
        (0.655, "perturbed", cpcol),
    ]

    for xx, ttl, cc in positions:
        aa = ax.inset_axes([xx, 0.23, 0.15, 0.56])
        aa.scatter(
            x[m],
            y[m],
            s=5,
            c=cc[m],
            lw=0,
            rasterized=True,
        )
        aa.axis("off")
        aa.set_aspect("equal")

        ax.text(
            xx + 0.075,
            0.82,
            ttl,
            ha="center",
            fontsize=8.2,
            fontweight="bold",
            transform=ax.transAxes,
        )

    ad = ax.inset_axes([0.82, 0.23, 0.15, 0.56])
    ad.scatter(
        x[m],
        y[m],
        s=4,
        c=".88",
        lw=0,
        rasterized=True,
    )
    ad.scatter(
        x[m & ch],
        y[m & ch],
        s=6,
        c="#222222",
        lw=0,
        rasterized=True,
    )
    ad.axis("off")
    ad.set_aspect("equal")

    ax.text(
        0.895,
        0.82,
        "reassigned units",
        ha="center",
        fontsize=8.2,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.735,
        0.12,
        (
            f"matched u = {meta['target_removed_fraction']:.2f}"
            f"     VI = {meta['VI_bits_recomputed']:.3f} bits"
        ),
        ha="center",
        fontsize=8.5,
        fontweight="bold",
        transform=ax.transAxes,
    )

    ax.text(
        0.735,
        0.075,
        "block colors aligned by maximum membership overlap",
        ha="center",
        fontsize=6.7,
        color=".35",
        transform=ax.transAxes,
    )


def heat(ax, df, sample, norms):
    sub = df[
        (df["sample"] == sample)
        & df["case"].isin(CASES)
    ]

    ts = sorted(sub.target_removed_fraction.unique())
    W = len(ts)
    gap = 0.55

    for k, (metric, lab, cmap) in enumerate(METRICS):
        z = np.full((10, W), np.nan)

        for i, case in enumerate(CASES):
            q = sub[sub.case == case].set_index(
                "target_removed_fraction"
            )
            for j, t in enumerate(ts):
                if t in q.index:
                    z[i, j] = q.loc[t, metric]

        x0 = k * (W + gap)

        x_edges = np.arange(W + 1, dtype=float) + x0 - 0.5
        y_edges = np.arange(11, dtype=float) - 0.5

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
            -1.25,
            lab,
            ha="center",
            va="bottom",
            fontsize=8.4,
            fontweight="bold",
        )

        for j, t in enumerate(ts):
            ax.text(
                x0 + j,
                10.05,
                f"{t:.2f}",
                rotation=45,
                ha="right",
                va="top",
                fontsize=6.4,
            )

    ax.set_xlim(
        -0.5,
        3 * (W + gap) + W - 0.5,
    )
    ax.set_ylim(11.15, -1.85)
    ax.set_yticks(range(10))
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

    ax.tick_params(length=0)

    for spine in ax.spines.values():
        spine.set_visible(False)


def bottom_d(ax, brain, kidney):
    """
    Matched-state deviation distributions.

    Frozen matched-trajectory measurements only:
      VI_bits_vs_ref
      top1_absdiff_vs_ref
      gini_absdiff_vs_ref

    Healthy brain is filled; nondiseased kidney is open.
    """
    ax.axis("off")

    metrics = [
        ("VI_bits_vs_ref", "VI (bits)"),
        ("top1_absdiff_vs_ref", "|Δ top-1% mass|"),
        ("gini_absdiff_vs_ref", "|Δ Gini|"),
    ]

    fams = [
        "threshold",
        "rescue",
        "schedule",
        "frontier",
        "joint",
    ]

    datasets = [
        (
            brain,
            "healthy_reference",
            True,
            "healthy brain",
        ),
        (
            kidney,
            "nondiseased_kidney",
            False,
            "nondiseased kidney",
        ),
    ]

    axes = []
    lefts = [0.00, 0.345, 0.69]

    for k, (metric, title) in enumerate(metrics):
        a = ax.inset_axes(
            [lefts[k], 0.16, 0.29, 0.76]
        )
        axes.append(a)

        for fi, fam in enumerate(fams):
            cases = [
                c for c in CASES
                if FAMILY[c] == fam
            ]

            for oi, (
                df,
                sample,
                filled,
                organ,
            ) in enumerate(datasets):

                q = df[
                    (df["sample"] == sample)
                    & df["case"].isin(cases)
                ][metric].to_numpy(float)

                q = q[np.isfinite(q)]

                pos = fi + (-0.13 if oi == 0 else 0.13)

                if len(q):
                    jitter = np.linspace(
                        -0.045,
                        0.045,
                        len(q),
                    )

                    a.scatter(
                        np.full(len(q), pos) + jitter,
                        q,
                        s=7,
                        facecolors=(
                            FC[fam]
                            if filled
                            else "none"
                        ),
                        edgecolors=FC[fam],
                        linewidths=0.45,
                        alpha=0.32,
                        zorder=1,
                    )

                    bp = a.boxplot(
                        [q],
                        positions=[pos],
                        widths=0.20,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                        medianprops=dict(
                            color=FC[fam],
                            linewidth=1.0,
                        ),
                        whiskerprops=dict(
                            color=FC[fam],
                            linewidth=0.8,
                        ),
                        capprops=dict(
                            color=FC[fam],
                            linewidth=0.8,
                        ),
                        boxprops=dict(
                            edgecolor=FC[fam],
                            linewidth=1.0,
                        ),
                    )

                    patch = bp["boxes"][0]
                    patch.set_facecolor(
                        FC[fam] if filled else "white"
                    )
                    patch.set_alpha(
                        0.28 if filled else 1.0
                    )

        a.set_title(
            title,
            fontsize=9.2,
            fontweight="bold",
            pad=5,
        )

        a.set_xticks(
            np.arange(len(fams)),
            fams,
            fontsize=7.0,
        )
        a.tick_params(
            axis="x",
            rotation=0,
            length=0,
        )
        a.tick_params(
            axis="y",
            labelsize=7,
        )
        a.spines[["top", "right"]].set_visible(False)
        a.grid(False)

    ax.text(
        0.0,
        1.02,
        "Matched-state deviation distributions",
        transform=ax.transAxes,
        fontsize=11.0,
        fontweight="bold",
        va="bottom",
    )

    ax.add_patch(
        Rectangle(
            (0.16, 0.015),
            0.035,
            0.055,
            transform=ax.transAxes,
            fc=".75",
            ec="black",
            lw=0.8,
        )
    )
    ax.text(
        0.205,
        0.042,
        "healthy brain (filled)",
        transform=ax.transAxes,
        va="center",
        fontsize=7.3,
    )

    ax.add_patch(
        Rectangle(
            (0.49, 0.015),
            0.035,
            0.055,
            transform=ax.transAxes,
            fc="white",
            ec="black",
            lw=0.8,
        )
    )
    ax.text(
        0.535,
        0.042,
        "nondiseased kidney (open)",
        transform=ax.transAxes,
        va="center",
        fontsize=7.3,
    )


def bottom_e(ax, bt, kt):
    """
    Terminal displacement.

    x: terminal partition displacement, VI
    y: terminal hierarchy-extent displacement, |ΔN|/Nref

    Each point is one frozen construction-rule perturbation.
    Brain is filled; kidney is open.
    """
    rows = []

    for organ, tab, sample, marker, filled in [
        (
            "Healthy brain",
            bt,
            "healthy_reference",
            "o",
            True,
        ),
        (
            "Nondiseased kidney",
            kt,
            "nondiseased_kidney",
            "s",
            False,
        ),
    ]:
        q = tab[
            (tab["sample"] == sample)
            & tab["case"].isin(CASES)
        ].copy()

        for _, z in q.iterrows():
            rows.append(
                dict(
                    organ=organ,
                    case=z["case"],
                    x=float(
                        z["terminal_VI_bits_vs_ref"]
                    ),
                    y=float(
                        z[
                            "terminal_node_relerr_vs_ref"
                        ]
                    ),
                    stop=str(z["stop_reason"]),
                    marker=marker,
                    filled=filled,
                )
            )

    if len(rows) != 20:
        raise RuntimeError(
            "Expected exactly 20 terminal perturbations "
            f"(10 brain + 10 kidney), found {len(rows)}"
        )

    for z in rows:
        fam = FAMILY[z["case"]]

        ax.scatter(
            z["x"],
            z["y"],
            s=54,
            marker=z["marker"],
            facecolors=(
                FC[fam]
                if z["filled"]
                else "white"
            ),
            edgecolors=FC[fam],
            linewidths=1.25,
            zorder=3,
        )

    ax.set_xlabel(
        "Terminal partition displacement, VI (bits)",
        fontsize=8.5,
    )
    ax.set_ylabel(
        "Terminal hierarchy-extent displacement, |ΔN|/Nref",
        fontsize=8.5,
    )

    ax.tick_params(labelsize=7.2)
    ax.spines[["top", "right"]].set_visible(False)

    xmax = max(z["x"] for z in rows)
    ymax = max(z["y"] for z in rows)

    ax.set_xlim(
        -0.025 * max(xmax, 1.0),
        xmax * 1.08,
    )
    ax.set_ylim(
        -0.025 * max(ymax, 0.01),
        ymax * 1.13,
    )

    ax.set_title(
        "Terminal displacement",
        loc="left",
        fontsize=11.0,
        fontweight="bold",
        pad=7,
    )

    from matplotlib.lines import Line2D

    handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            label="Healthy brain (n = 10)",
        ),
        Line2D(
            [0], [0],
            marker="s",
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=6,
            label="Nondiseased kidney (n = 10)",
        ),
    ]

    for fam in [
        "threshold",
        "rescue",
        "schedule",
        "frontier",
        "joint",
    ]:
        handles.append(
            Line2D(
                [0], [0],
                marker="o",
                linestyle="none",
                markerfacecolor=FC[fam],
                markeredgecolor=FC[fam],
                markersize=5.5,
                label=fam,
            )
        )

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        fontsize=6.8,
        handletextpad=0.5,
        borderaxespad=0.2,
    )

    ok = sum(
        z["stop"]
        == "no_contextually_admissible_boundaries"
        for z in rows
    )

    ax.text(
        0.98,
        0.97,
        (
            f"{ok}/{len(rows)} perturbations terminated by\n"
            "no contextually admissible boundaries"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.2,
    )


def validate_inputs(data):
    required = {
        "brain_hrm_matched_trajectory.csv",
        "brain_hrm_terminal_summary.csv",
        "kidney_hrm_matched_trajectory.csv",
        "kidney_hrm_terminal_summary.csv",
        "brain_u055_exact_partitions.parquet",
        "brain_u055_partition_materialization.json",
        "brain_level0_cells.parquet",
    }

    missing = sorted(
        name
        for name in required
        if not (data / name).is_file()
    )

    if missing:
        raise FileNotFoundError(
            "Missing frozen Fig4 inputs: "
            + ", ".join(missing)
        )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing the seven frozen Fig4 inputs.",
    )
    ap.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Directory for standalone Fig4 panels.",
    )

    args = ap.parse_args()

    data = args.data_root.expanduser().resolve()
    out = args.output_root.expanduser().resolve()

    validate_inputs(data)
    out.mkdir(parents=True, exist_ok=True)

    brain = pd.read_csv(
        data / "brain_hrm_matched_trajectory.csv"
    )
    kidney = pd.read_csv(
        data / "kidney_hrm_matched_trajectory.csv"
    )
    bt = pd.read_csv(
        data / "brain_hrm_terminal_summary.csv"
    )
    kt = pd.read_csv(
        data / "kidney_hrm_terminal_summary.csv"
    )

    part = rp(
        data / "brain_u055_exact_partitions.parquet"
    )
    meta = json.loads(
        (
            data
            / "brain_u055_partition_materialization.json"
        ).read_text()
    )
    cells = rp(
        data / "brain_level0_cells.parquet"
    )

    norms = {}

    for metric, _, _ in METRICS:
        vals = np.r_[
            brain.loc[
                brain.case != "reference",
                metric,
            ].to_numpy(float),
            kidney.loc[
                kidney.case != "reference",
                metric,
            ].to_numpy(float),
        ]
        norms[metric] = Normalize(
            0,
            np.nanmax(vals),
        )

    fig, ax = plt.subplots(figsize=(13, 3.3))
    draw_panel_a(
        ax,
        cells,
        part,
        meta,
    )
    save(fig, out, "Fig4A")

    fig, ax = plt.subplots(figsize=(13, 3.25))
    heat(
        ax,
        brain,
        "healthy_reference",
        norms,
    )
    save(fig, out, "Fig4B")

    fig, ax = plt.subplots(figsize=(13, 3.25))
    heat(
        ax,
        kidney,
        "nondiseased_kidney",
        norms,
    )
    save(fig, out, "Fig4C")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bottom_d(
        ax,
        brain,
        kidney,
    )
    fig.tight_layout()
    save(fig, out, "Fig4D")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bottom_e(
        ax,
        bt,
        kt,
    )
    fig.tight_layout()
    save(fig, out, "Fig4E")

    manifest = {
        "figure": "MainFig4",
        "mode": "standalone_panels",
        "panels": [
            "Fig4A",
            "Fig4B",
            "Fig4C",
            "Fig4D",
            "Fig4E",
        ],
        "assembled_figure": False,
        "scientific_recalculation": False,
        "inputs": [
            "brain_hrm_matched_trajectory.csv",
            "brain_hrm_terminal_summary.csv",
            "kidney_hrm_matched_trajectory.csv",
            "kidney_hrm_terminal_summary.csv",
            "brain_u055_exact_partitions.parquet",
            "brain_u055_partition_materialization.json",
            "brain_level0_cells.parquet",
        ],
    }

    (
        out / "Figure4_standalone_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        )
        + "\n"
    )

    print(
        "WROTE",
        out / "Figure4_standalone_manifest.json",
    )


if __name__ == "__main__":
    main()
