#!/usr/bin/env python3

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy.optimize import linear_sum_assignment


ROOT = Path.home() / "Desktop" / "SUTRA"

V7 = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "structural_trajectory_v7"
)

V9 = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "relational_flow_v9"
)

OUT = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "domain_trajectories_v10"
)
OUT.mkdir(parents=True, exist_ok=True)

U = np.array([0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56])

# Primary definition. Robustness versions are exported too.
PRIMARY_Q = 0.80
ROBUST_Q = [0.75, 0.80, 0.85]

# Remove tiny connected islands.
MIN_DOMAIN_PIXELS = 18

# Matching controls.
# Primary relation is spatial overlap after a modest dilation.
DILATION_PIXELS = 3
MIN_DILATED_IOU = 0.035

# For domains with little direct overlap, allow nearby correspondence.
MAX_CENTROID_DISTANCE_FRACTION = 0.14

# Score components.
W_IOU = 0.65
W_DISTANCE = 0.25
W_DISTORTION = 0.10

# Retain secondary relations if they are reasonably close to the
# best relation from the same parent. This allows visible splits.
SECONDARY_SCORE_FRACTION = 0.72
MIN_EDGE_SCORE = 0.16

DPI = 600
EPS = 1e-12


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


def load_fields():
    p = V9 / "relational_flow_fields.npz"
    if not p.exists():
        raise FileNotFoundError(p)

    z = np.load(p, allow_pickle=False)

    xc = np.asarray(z["x_centers"], float)
    yc = np.asarray(z["y_centers"], float)

    D = {}
    for u in U:
        k = f"D_u{u:.2f}"
        if k not in z.files:
            raise RuntimeError(f"Missing field {k}")
        D[u] = np.asarray(z[k], float)

    return xc, yc, D


def load_cells():
    p = V7 / "prcc_cell_multiscale_gw.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(
        p,
        columns=["cell_index", "x", "y"]
    )

    good = (
        np.isfinite(df["x"].to_numpy(float)) &
        np.isfinite(df["y"].to_numpy(float))
    )

    return df.loc[good].copy()


def specimen_diagonal(xc, yc):
    return float(
        math.hypot(
            float(xc[-1] - xc[0]),
            float(yc[-1] - yc[0]),
        )
    )


def clean_binary(mask):
    """
    Connected high-distortion regions with small islands removed.

    No hole filling: internal tissue geometry is retained.
    """
    structure = np.ones((3, 3), dtype=int)
    lab, n = ndimage.label(mask, structure=structure)

    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(lab.ravel())

    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = counts[1:] >= MIN_DOMAIN_PIXELS

    return keep[lab]


def extract_domains(D, xc, yc, q):
    """
    Define domains independently at each hierarchy coordinate from
    the upper q-quantile of finite D(x,u).

    q is fixed across hierarchy coordinates.
    """
    domain_rows = []
    label_maps = {}
    masks_by_key = {}

    for u in U:
        A = D[u]
        finite = np.isfinite(A)

        if not np.any(finite):
            raise RuntimeError(f"No finite field at u={u}")

        threshold = float(np.quantile(A[finite], q))

        binary = finite & (A >= threshold)
        binary = clean_binary(binary)

        structure = np.ones((3, 3), dtype=int)
        lab, n = ndimage.label(binary, structure=structure)

        label_maps[u] = lab

        local_id = 0

        for lbl in range(1, n + 1):
            yy, xx = np.where(lab == lbl)

            if len(xx) < MIN_DOMAIN_PIXELS:
                continue

            local_id += 1

            vals = A[yy, xx]

            cx = float(np.mean(xc[xx]))
            cy = float(np.mean(yc[yy]))

            key = (float(u), int(local_id))

            m = np.zeros_like(binary, dtype=bool)
            m[yy, xx] = True
            masks_by_key[key] = m

            domain_rows.append({
                "u": float(u),
                "domain_id": int(local_id),
                "threshold": threshold,
                "n_pixels": int(len(xx)),
                "centroid_x": cx,
                "centroid_y": cy,
                "mean_distortion": float(np.mean(vals)),
                "median_distortion": float(np.median(vals)),
                "max_distortion": float(np.max(vals)),
                "integrated_distortion": float(np.sum(vals)),
            })

        print(
            f"q={q:.2f} u={u:.2f}: "
            f"threshold={threshold:.6g}, "
            f"domains={local_id}"
        )

    return (
        pd.DataFrame(domain_rows),
        label_maps,
        masks_by_key,
    )


def dilated_iou(a, b):
    structure = np.ones((3, 3), dtype=bool)

    ad = ndimage.binary_dilation(
        a,
        structure=structure,
        iterations=DILATION_PIXELS,
    )

    bd = ndimage.binary_dilation(
        b,
        structure=structure,
        iterations=DILATION_PIXELS,
    )

    inter = np.logical_and(ad, bd).sum()
    union = np.logical_or(ad, bd).sum()

    if union == 0:
        return 0.0

    return float(inter / union)


def relation_score(a, b, ma, mb, diag, distortion_scale):
    iou = dilated_iou(ma, mb)

    dx = float(a["centroid_x"] - b["centroid_x"])
    dy = float(a["centroid_y"] - b["centroid_y"])
    dist = math.hypot(dx, dy)
    dist_fraction = dist / max(diag, EPS)

    distance_similarity = max(
        0.0,
        1.0 -
        dist_fraction / MAX_CENTROID_DISTANCE_FRACTION,
    )

    dd = abs(
        float(a["mean_distortion"]) -
        float(b["mean_distortion"])
    )

    distortion_similarity = math.exp(
        -dd / max(distortion_scale, EPS)
    )

    score = (
        W_IOU * iou +
        W_DISTANCE * distance_similarity +
        W_DISTORTION * distortion_similarity
    )

    return {
        "score": float(score),
        "dilated_iou": float(iou),
        "centroid_distance": float(dist),
        "centroid_distance_fraction": float(dist_fraction),
        "distortion_similarity": float(distortion_similarity),
    }


def build_edges(domains, masks, xc, yc):
    diag = specimen_diagonal(xc, yc)

    allD = domains["mean_distortion"].to_numpy(float)
    distortion_scale = float(
        np.quantile(allD, 0.75) -
        np.quantile(allD, 0.25)
    )
    distortion_scale = max(distortion_scale, EPS)

    edge_rows = []

    for ua, ub in zip(U[:-1], U[1:]):
        A = domains[domains["u"] == ua].copy()
        B = domains[domains["u"] == ub].copy()

        A = A.reset_index(drop=True)
        B = B.reset_index(drop=True)

        if len(A) == 0 or len(B) == 0:
            continue

        S = np.zeros((len(A), len(B)), dtype=float)
        metadata = {}

        for i, a in A.iterrows():
            ka = (float(ua), int(a["domain_id"]))
            ma = masks[ka]

            for j, b in B.iterrows():
                kb = (float(ub), int(b["domain_id"]))
                mb = masks[kb]

                r = relation_score(
                    a, b, ma, mb,
                    diag, distortion_scale,
                )

                S[i, j] = r["score"]
                metadata[(i, j)] = r

        # Hungarian primary one-to-one backbone.
        rr, cc = linear_sum_assignment(-S)

        primary = set()

        for i, j in zip(rr, cc):
            r = metadata[(i, j)]

            # Require either some spatial overlap or genuinely close
            # centroids, in addition to score.
            spatial_ok = (
                r["dilated_iou"] >= MIN_DILATED_IOU or
                r["centroid_distance_fraction"]
                <= MAX_CENTROID_DISTANCE_FRACTION
            )

            if S[i, j] >= MIN_EDGE_SCORE and spatial_ok:
                primary.add((i, j))

        selected = set(primary)

        # Preserve plausible splits: secondary child relation close
        # to the best child relation for that parent.
        for i in range(len(A)):
            best = float(np.max(S[i]))

            if best <= 0:
                continue

            for j in range(len(B)):
                r = metadata[(i, j)]

                spatial_ok = (
                    r["dilated_iou"] >= MIN_DILATED_IOU or
                    r["centroid_distance_fraction"]
                    <= MAX_CENTROID_DISTANCE_FRACTION
                )

                if (
                    spatial_ok and
                    S[i, j] >= MIN_EDGE_SCORE and
                    S[i, j] >= SECONDARY_SCORE_FRACTION * best
                ):
                    selected.add((i, j))

        # Preserve plausible mergers by symmetric criterion.
        for j in range(len(B)):
            best = float(np.max(S[:, j]))

            if best <= 0:
                continue

            for i in range(len(A)):
                r = metadata[(i, j)]

                spatial_ok = (
                    r["dilated_iou"] >= MIN_DILATED_IOU or
                    r["centroid_distance_fraction"]
                    <= MAX_CENTROID_DISTANCE_FRACTION
                )

                if (
                    spatial_ok and
                    S[i, j] >= MIN_EDGE_SCORE and
                    S[i, j] >= SECONDARY_SCORE_FRACTION * best
                ):
                    selected.add((i, j))

        for i, j in sorted(selected):
            a = A.iloc[i]
            b = B.iloc[j]
            r = metadata[(i, j)]

            edge_rows.append({
                "u_from": float(ua),
                "domain_from": int(a["domain_id"]),
                "u_to": float(ub),
                "domain_to": int(b["domain_id"]),
                "primary_hungarian": bool((i, j) in primary),
                "score": float(S[i, j]),
                "dilated_iou": r["dilated_iou"],
                "centroid_distance": r["centroid_distance"],
                "centroid_distance_fraction":
                    r["centroid_distance_fraction"],
                "distortion_similarity":
                    r["distortion_similarity"],
                "x_from": float(a["centroid_x"]),
                "y_from": float(a["centroid_y"]),
                "x_to": float(b["centroid_x"]),
                "y_to": float(b["centroid_y"]),
                "distortion_from":
                    float(a["mean_distortion"]),
                "distortion_to":
                    float(b["mean_distortion"]),
                "delta_distortion":
                    float(
                        b["mean_distortion"] -
                        a["mean_distortion"]
                    ),
            })

    return pd.DataFrame(edge_rows)


def classify_topology(domains, edges):
    D = domains.copy()

    indeg = {}
    outdeg = {}

    for _, e in edges.iterrows():
        a = (
            float(e["u_from"]),
            int(e["domain_from"]),
        )
        b = (
            float(e["u_to"]),
            int(e["domain_to"]),
        )

        outdeg[a] = outdeg.get(a, 0) + 1
        indeg[b] = indeg.get(b, 0) + 1

    types = []

    for _, r in D.iterrows():
        k = (float(r["u"]), int(r["domain_id"]))
        ni = indeg.get(k, 0)
        no = outdeg.get(k, 0)

        if ni == 0 and no == 0:
            t = "isolated"
        elif ni == 0:
            t = "birth"
        elif no == 0:
            t = "termination"
        elif ni > 1 and no > 1:
            t = "merge_split"
        elif ni > 1:
            t = "merge"
        elif no > 1:
            t = "split"
        else:
            t = "continuation"

        types.append(t)

    D["trajectory_role"] = types
    return D


def plot_domain_trajectory(
    cells,
    domains,
    edges,
    xc,
    yc,
    stem,
):
    fig, ax = plt.subplots(figsize=(7.2, 7.2))

    # Real tissue as quiet anatomical support.
    ax.scatter(
        cells["x"],
        cells["y"],
        s=0.35,
        c="0.86",
        linewidths=0,
        rasterized=True,
        zorder=1,
    )

    # Global distortion range controls node intensity.
    vals = domains["mean_distortion"].to_numpy(float)
    vmin = float(np.quantile(vals, 0.02))
    vmax = float(np.quantile(vals, 0.98))

    # Edges first.
    maxscore = max(
        float(edges["score"].max())
        if len(edges) else 1.0,
        EPS,
    )

    for _, e in edges.iterrows():
        delta = float(e["delta_distortion"])

        # Direction encoded by arrow geometry; change sign encoded
        # by edge tone rather than claiming biological direction.
        color = (
            "#B2182B" if delta > 0
            else "#2166AC"
        )

        alpha = 0.20 + 0.62 * (
            float(e["score"]) / maxscore
        )

        lw = 0.45 + 2.0 * (
            float(e["score"]) / maxscore
        )

        ax.annotate(
            "",
            xy=(e["x_to"], e["y_to"]),
            xytext=(e["x_from"], e["y_from"]),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                alpha=alpha,
                lw=lw,
                mutation_scale=6.5,
                shrinkA=5,
                shrinkB=5,
            ),
            zorder=2,
        )

    # Domain nodes.
    sc = ax.scatter(
        domains["centroid_x"],
        domains["centroid_y"],
        c=domains["mean_distortion"],
        s=(
            12 +
            0.55 * np.sqrt(
                domains["n_pixels"].to_numpy(float)
            ) * 10
        ),
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.45,
        zorder=3,
    )

    ax.set_aspect("equal")
    ax.set_axis_off()

    cb = fig.colorbar(
        sc,
        ax=ax,
        fraction=0.035,
        pad=0.012,
    )
    cb.set_label("Mean relational displacement")

    fig.tight_layout(pad=0.1)
    save(fig, stem)


def plot_scale_lanes(domains, edges, stem):
    """
    Abstract trajectory view: hierarchy coordinate on x,
    relational domains on y according to their tissue y-centroid.

    Useful control showing that edges connect actual domains rather
    than being arbitrary streamline integration.
    """
    fig, ax = plt.subplots(figsize=(8.6, 5.2))

    vals = domains["mean_distortion"].to_numpy(float)
    vmin = float(np.quantile(vals, 0.02))
    vmax = float(np.quantile(vals, 0.98))

    for _, e in edges.iterrows():
        color = (
            "#B2182B"
            if e["delta_distortion"] > 0
            else "#2166AC"
        )

        ax.plot(
            [e["u_from"], e["u_to"]],
            [e["y_from"], e["y_to"]],
            color=color,
            alpha=0.35 + 0.5 * e["score"],
            lw=0.6 + 1.4 * e["score"],
            zorder=1,
        )

    sc = ax.scatter(
        domains["u"],
        domains["centroid_y"],
        c=domains["mean_distortion"],
        s=14 + 1.1 * np.sqrt(domains["n_pixels"]),
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.35,
        zorder=2,
    )

    ax.set_xticks(U)
    ax.set_xticklabels([f"{u:.2f}" for u in U])
    ax.set_xlabel("SUTRA hierarchy coordinate, $u$")
    ax.set_ylabel("Spatial position of relational domain")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cb = fig.colorbar(
        sc,
        ax=ax,
        fraction=0.035,
        pad=0.02,
    )
    cb.set_label("Mean relational displacement")

    fig.tight_layout()
    save(fig, stem)


def summarize(domains, edges, q):
    per_u = (
        domains.groupby("u")
        .agg(
            n_domains=("domain_id", "size"),
            total_domain_pixels=("n_pixels", "sum"),
            mean_domain_distortion=(
                "mean_distortion", "mean"
            ),
            median_domain_distortion=(
                "mean_distortion", "median"
            ),
        )
        .reset_index()
    )

    per_u["quantile"] = q

    if len(edges):
        per_transition = (
            edges.groupby(["u_from", "u_to"])
            .agg(
                n_relations=("score", "size"),
                n_primary=(
                    "primary_hungarian", "sum"
                ),
                mean_score=("score", "mean"),
                mean_delta_distortion=(
                    "delta_distortion", "mean"
                ),
                fraction_increasing=(
                    "delta_distortion",
                    lambda x: float(
                        np.mean(np.asarray(x) > 0)
                    ),
                ),
            )
            .reset_index()
        )
    else:
        per_transition = pd.DataFrame()

    return per_u, per_transition


def run_quantile(q, xc, yc, D, cells):
    print("\n" + "=" * 90)
    print("DOMAIN QUANTILE", q)
    print("=" * 90)

    domains, label_maps, masks = extract_domains(
        D, xc, yc, q
    )

    edges = build_edges(
        domains, masks, xc, yc
    )

    domains = classify_topology(domains, edges)

    tag = f"q{int(round(q * 100)):02d}"

    domains.to_csv(
        OUT / f"domains_{tag}.csv",
        index=False,
    )

    edges.to_csv(
        OUT / f"relations_{tag}.csv",
        index=False,
    )

    per_u, per_transition = summarize(
        domains, edges, q
    )

    per_u.to_csv(
        OUT / f"domain_summary_{tag}.csv",
        index=False,
    )

    per_transition.to_csv(
        OUT / f"relation_summary_{tag}.csv",
        index=False,
    )

    print("\nDomains:")
    print(per_u.to_string(index=False))

    print("\nRelations:")
    if len(per_transition):
        print(per_transition.to_string(index=False))
    else:
        print("NONE")

    if abs(q - PRIMARY_Q) < 1e-12:
        plot_domain_trajectory(
            cells,
            domains,
            edges,
            xc,
            yc,
            "relational_domain_trajectories",
        )

        plot_scale_lanes(
            domains,
            edges,
            "relational_domain_scale_lanes",
        )

    return domains, edges, per_u, per_transition


def robustness_table(results):
    rows = []

    for q, (_, edges, per_u, per_t) in results.items():
        for _, r in per_u.iterrows():
            rows.append({
                "quantile": q,
                "kind": "scale",
                "u_from": r["u"],
                "u_to": np.nan,
                "n": int(r["n_domains"]),
                "mean_delta": np.nan,
            })

        for _, r in per_t.iterrows():
            rows.append({
                "quantile": q,
                "kind": "transition",
                "u_from": r["u_from"],
                "u_to": r["u_to"],
                "n": int(r["n_relations"]),
                "mean_delta":
                    float(r["mean_delta_distortion"]),
            })

    R = pd.DataFrame(rows)
    R.to_csv(
        OUT / "quantile_robustness_summary.csv",
        index=False,
    )
    return R


def main():
    xc, yc, D = load_fields()
    cells = load_cells()

    results = {}

    for q in ROBUST_Q:
        results[q] = run_quantile(
            q, xc, yc, D, cells
        )

    R = robustness_table(results)

    provenance = {
        "version": "domain_trajectories_v10",
        "primary_quantile": PRIMARY_Q,
        "robustness_quantiles": ROBUST_Q,
        "domain_definition": (
            "Connected components of the upper fixed within-scale "
            "quantile of the frozen V9 PRCC relational-displacement "
            "field, with small spatial islands removed."
        ),
        "matching": (
            "Adjacent-scale domain correspondence using dilated "
            "spatial overlap, centroid proximity and mean-distortion "
            "similarity. Hungarian assignment defines a primary "
            "backbone; plausible secondary relations are retained "
            "to represent splits and mergers."
        ),
        "edge_interpretation": (
            "An edge is correspondence between spatial relational "
            "domains across adjacent SUTRA hierarchy coordinates. "
            "It is not cell movement, RNA velocity, lineage, "
            "biological time, causal flow, or disease progression."
        ),
        "red_edge": (
            "matched domain has greater mean relational displacement "
            "at the next hierarchy coordinate"
        ),
        "blue_edge": (
            "matched domain has lower mean relational displacement "
            "at the next hierarchy coordinate"
        ),
        "u_values": U.tolist(),
        "min_domain_pixels": MIN_DOMAIN_PIXELS,
        "dilation_pixels": DILATION_PIXELS,
        "weights": {
            "iou": W_IOU,
            "distance": W_DISTANCE,
            "distortion": W_DISTORTION,
        },
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\n" + "=" * 90)
    print("V10 COMPLETE")
    print("=" * 90)
    print(R.to_string(index=False))
    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
