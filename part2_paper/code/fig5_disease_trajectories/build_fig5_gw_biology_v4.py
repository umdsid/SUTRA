#!/usr/bin/env python3

from pathlib import Path
import json
import re
import math

import numpy as np
import pandas as pd
import h5py


ROOT = Path.home() / "Desktop" / "SUTRA"

GW = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "gw_multistart_v3"
)

LEDGER = (
    ROOT / "results" /
    "hierarchy_v0911_specimen_local_contextual_flow" /
    "ledger" / "prcc"
)

L0 = (
    ROOT / "results" /
    "hierarchy_level0_v070" /
    "prcc"
)

OUT = (
    ROOT / "results" /
    "Fig5_Disease_Trajectories" /
    "gw_biology_v4"
)

U_VALUES = [0.32, 0.40, 0.48, 0.56]
DISPLAY_U = 0.40
K = 512

# Descriptive ranking parameters, not inferential significance thresholds.
MIN_DETECTION = 0.01
MIN_CELLS_OBJECT = 2
TOP_EXPORT = 80
EPS = 1e-12


def read_steps():
    rows = []
    p = LEDGER / "steps.jsonl"

    with p.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    df = pd.DataFrame(rows)

    required = [
        "microstep",
        "hierarchy_coordinate_removed_fraction",
        "nodes_before",
        "nodes_after",
    ]

    for c in required:
        if c not in df.columns:
            raise RuntimeError(f"steps.jsonl missing {c}")

    return df


def exact_target_step(steps, u):
    """
    Select the productive hierarchy state already used by GW V2.

    GW V2 trajectory stores actual disease u.  We recover the row nearest
    to that exact value and reconstruct the state AFTER that microstep.
    """
    p = GW / "multistart_scale_summary.csv"
    ms = pd.read_csv(p)

    r = ms.iloc[
        np.argmin(np.abs(ms["u_target"].to_numpy(float) - u))
    ]

    u_actual = float(r["u_disease"])

    x = steps[
        "hierarchy_coordinate_removed_fraction"
    ].to_numpy(float)

    idx = int(np.argmin(np.abs(x - u_actual)))
    rr = steps.iloc[idx]

    if abs(float(rr["hierarchy_coordinate_removed_fraction"]) - u_actual) > 5e-6:
        raise RuntimeError(
            f"Could not bind u={u} to disease hierarchy state: "
            f"{rr['hierarchy_coordinate_removed_fraction']} vs {u_actual}"
        )

    return int(rr["microstep"]), u_actual, rr


def checkpoint_catalog():
    rows = []

    for p in sorted((LEDGER / "label_checkpoints").glob("labels_*.npz")):
        m = re.search(r"labels_(\d+)\.npz$", p.name)
        if m:
            rows.append((int(m.group(1)), p))

    if not rows:
        raise RuntimeError("No PRCC label checkpoints")

    return rows


def reconstruct_labels_after_step(target_step, n0):
    """
    Checkpoint S is state BEFORE microstep S.

    Start from greatest checkpoint <= target_step and replay accepted
    mergers from checkpoint step through target_step inclusive.

    Accepted mergers within a microstep are disjoint, so simultaneous
    relabeling is valid.
    """
    cps = checkpoint_catalog()
    eligible = [(s, p) for s, p in cps if s <= target_step]

    if not eligible:
        raise RuntimeError("No preceding checkpoint")

    cp_step, cp_path = eligible[-1]

    with np.load(cp_path, allow_pickle=False) as z:
        labels = np.asarray(z["labels"], dtype=np.int64).copy()

    if len(labels) != n0:
        raise RuntimeError(
            f"checkpoint length {len(labels)} != N0 {n0}"
        )

    merge_dir = LEDGER / "merge_events"

    for s in range(cp_step, target_step + 1):
        p = merge_dir / f"step_{s:06d}.parquet"

        if not p.exists():
            # terminal/nonproductive step is allowed.
            continue

        ev = pd.read_parquet(
            p,
            columns=["survivor_node", "removed_node"],
        )

        if len(ev) == 0:
            continue

        survivors = ev["survivor_node"].to_numpy(np.int64)
        removed = ev["removed_node"].to_numpy(np.int64)

        if len(np.intersect1d(survivors, removed)):
            raise RuntimeError(
                f"overlapping survivor/removed labels at step {s}"
            )

        # Each row is an accepted disjoint merger in the same pre-state.
        # Replace all removed labels by corresponding survivor labels.
        mapping = dict(zip(removed.tolist(), survivors.tolist()))

        mask = np.isin(labels, removed)
        if np.any(mask):
            vals = labels[mask]
            labels[mask] = np.fromiter(
                (mapping[int(x)] for x in vals),
                dtype=np.int64,
                count=len(vals),
            )

    return labels, cp_step, cp_path


def load_cells():
    cells = pd.read_parquet(L0 / "cells.parquet")

    required = [
        "cell_index",
        "matrix_column",
        "x",
        "y",
    ]
    for c in required:
        if c not in cells.columns:
            raise RuntimeError(f"cells.parquet missing {c}")

    cells = cells.sort_values("cell_index").reset_index(drop=True)

    n = len(cells)
    expected = np.arange(n, dtype=np.int64)

    if not np.array_equal(
        cells["cell_index"].to_numpy(np.int64), expected
    ):
        raise RuntimeError("cell_index is not exact 0..N0-1")

    if not np.array_equal(
        cells["matrix_column"].to_numpy(np.int64), expected
    ):
        raise RuntimeError(
            "matrix_column does not equal cell_index"
        )

    return cells


def load_feature_names():
    f = pd.read_parquet(L0 / "features.parquet")

    if "feature_index" not in f.columns:
        raise RuntimeError("features.parquet lacks feature_index")

    f = f.sort_values("feature_index").reset_index(drop=True)

    candidates = [
        "feature_name",
        "gene_name",
        "name",
        "gene",
        "feature_id",
        "id",
    ]

    col = next((x for x in candidates if x in f.columns), None)

    if col is None:
        raise RuntimeError(
            f"Cannot infer gene-name column from {list(f.columns)}"
        )

    names = f[col].astype(str).to_numpy()

    if len(names) != 541:
        raise RuntimeError(f"Expected 541 PRCC genes, found {len(names)}")

    return names, f


def source_matrix_path():
    manifest = json.loads(
        (L0 / "expression_manifest.json").read_text()
    )

    for key in [
        "source_matrix",
        "source",
        "matrix_path",
        "source_path",
    ]:
        if key in manifest:
            p = Path(manifest[key])
            if p.exists():
                return p

    raise RuntimeError(
        "Could not locate source expression matrix from manifest"
    )


def read_10x_expression():
    """
    Materialize 541 x N raw counts from the existing 10x-style H5.

    541 x 56,510 is small enough (~30.6M values) for a dense float32
    matrix during this downstream analysis.
    """
    p = source_matrix_path()
    print("Expression:", p)

    with h5py.File(p, "r") as h:
        g = h["matrix"]

        data = np.asarray(g["data"])
        indices = np.asarray(g["indices"], dtype=np.int64)
        indptr = np.asarray(g["indptr"], dtype=np.int64)
        shape = tuple(np.asarray(g["shape"], dtype=np.int64))

    if shape != (541, 56510):
        raise RuntimeError(f"Unexpected PRCC expression shape {shape}")

    # 10x H5 is CSC: columns are cells.
    X = np.zeros(shape, dtype=np.float32)

    for j in range(shape[1]):
        a, b = indptr[j], indptr[j + 1]
        X[indices[a:b], j] = data[a:b]

    return X


def normalize_log1p(X):
    lib = X.sum(axis=0, dtype=np.float64)

    scale = np.zeros_like(lib, dtype=np.float64)
    good = lib > 0
    scale[good] = 1e4 / lib[good]

    Y = np.log1p(
        X.astype(np.float64) * scale[None, :]
    ).astype(np.float32)

    return Y, lib


def rankdata_average(x):
    """
    Dependency-free average ranks.
    """
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=np.float64)

    xs = x[order]
    i = 0

    while i < len(x):
        j = i + 1
        while j < len(x) and xs[j] == xs[i]:
            j += 1

        # ranks are 1-based convention
        r = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = r
        i = j

    return ranks


def weighted_corr(x, y, w):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    good = (
        np.isfinite(x) &
        np.isfinite(y) &
        np.isfinite(w) &
        (w > 0)
    )

    x, y, w = x[good], y[good], w[good]

    if len(x) < 3:
        return np.nan

    w /= w.sum()

    mx = np.sum(w * x)
    my = np.sum(w * y)

    dx = x - mx
    dy = y - my

    vx = np.sum(w * dx * dx)
    vy = np.sum(w * dy * dy)

    if vx <= EPS or vy <= EPS:
        return np.nan

    return float(
        np.sum(w * dx * dy) / np.sqrt(vx * vy)
    )


def object_expression(Y, labels):
    """
    Mean normalized expression per SUTRA object.
    """
    uniq, inv, counts = np.unique(
        labels,
        return_inverse=True,
        return_counts=True,
    )

    ng = Y.shape[0]
    no = len(uniq)

    sums = np.zeros((ng, no), dtype=np.float64)

    # Only 541 genes, so bincount is efficient and memory-safe.
    for g in range(ng):
        sums[g] = np.bincount(
            inv,
            weights=Y[g],
            minlength=no,
        )

    means = sums / counts[None, :]

    return uniq, inv, counts, means


def project_gw_to_cells(npz_path, labels):
    with np.load(npz_path, allow_pickle=False) as z:
        owner = np.asarray(z["owner_disease"], dtype=np.int64)
        objlabels = np.asarray(
            z["dominant_object_labels_disease"],
            dtype=np.int64,
        )
        delta_landmark = np.asarray(
            z["disease_conditional_distortion"],
            dtype=np.float64,
        )

        if len(owner) != len(objlabels):
            raise RuntimeError("owner/object-label length mismatch")

        delta_obj = delta_landmark[owner]

        table = pd.DataFrame({
            "object_label": objlabels,
            "landmark": owner,
            "gw_distortion": delta_obj,
        })

        if table["object_label"].duplicated().any():
            raise RuntimeError("duplicate dominant object labels")

        lut = dict(
            zip(
                table["object_label"].astype(int),
                table["gw_distortion"].astype(float),
            )
        )

        cell_delta = np.full(
            len(labels),
            np.nan,
            dtype=np.float64,
        )

        for i, lab in enumerate(labels):
            if int(lab) in lut:
                cell_delta[i] = lut[int(lab)]

        coverage_cells = float(np.mean(np.isfinite(cell_delta)))

        return table, cell_delta, coverage_cells


def gene_associations(
    gene_names,
    Y,
    raw,
    labels,
    cell_delta,
    u,
):
    """
    Two complementary analyses:

    1. Cell-mass-weighted continuous association.
       Spearman-like correlation between cell expression and inherited
       GW distortion.

    2. Object-level association.
       Correlation between object mean expression and object GW
       distortion, weighted by sqrt(object mass). This prevents a single
       enormous object from completely dominating while retaining some
       information about object support.

    These are descriptive association statistics, not p-values.
    """
    finite = np.isfinite(cell_delta)
    idx = np.flatnonzero(finite)

    dcell = cell_delta[idx]
    rdcell = rankdata_average(dcell)

    uniq, inv, counts, obj_mean = object_expression(
        Y[:, idx],
        labels[idx],
    )

    # Each object has one inherited GW value.
    first = np.full(len(uniq), -1, dtype=np.int64)
    for i, oi in enumerate(inv):
        if first[oi] < 0:
            first[oi] = i

    dobj = dcell[first]
    rdobj = rankdata_average(dobj)

    sqrt_mass = np.sqrt(counts.astype(np.float64))

    rows = []

    for g, name in enumerate(gene_names):
        y = Y[g, idx].astype(np.float64)
        det = float(np.mean(raw[g, idx] > 0))

        if det < MIN_DETECTION:
            continue

        ry = rankdata_average(y)

        cell_rho = weighted_corr(
            rdcell,
            ry,
            np.ones(len(idx), dtype=np.float64),
        )

        oy = obj_mean[g]
        roy = rankdata_average(oy)

        object_rho = weighted_corr(
            rdobj,
            roy,
            sqrt_mass,
        )

        # Top vs bottom distortion quintiles are retained only as an
        # interpretable effect-size display, not as the primary test.
        q20, q80 = np.quantile(dcell, [0.20, 0.80])

        lo = dcell <= q20
        hi = dcell >= q80

        hi_mean = float(np.mean(y[hi]))
        lo_mean = float(np.mean(y[lo]))

        hi_det = float(np.mean(raw[g, idx[hi]] > 0))
        lo_det = float(np.mean(raw[g, idx[lo]] > 0))

        rows.append({
            "u": u,
            "gene": name,
            "detection": det,
            "cell_spearman": cell_rho,
            "object_spearman_sqrt_mass": object_rho,
            "high20_mean_logexpr": hi_mean,
            "low20_mean_logexpr": lo_mean,
            "high_minus_low_logexpr": hi_mean - lo_mean,
            "high20_detection": hi_det,
            "low20_detection": lo_det,
            "detection_difference": hi_det - lo_det,
            "n_cells": len(idx),
            "n_objects": len(uniq),
        })

    out = pd.DataFrame(rows)

    # Convergent ranking: require the cell and object analyses to point
    # in the same direction. Score magnitude is transparent and used
    # only for prioritization.
    same = (
        np.sign(out["cell_spearman"]) ==
        np.sign(out["object_spearman_sqrt_mass"])
    )

    out["direction_concordant"] = same

    out["convergent_score"] = np.where(
        same,
        np.sqrt(
            np.abs(out["cell_spearman"]) *
            np.abs(out["object_spearman_sqrt_mass"])
        ),
        0.0,
    )

    out = out.sort_values(
        ["convergent_score", "gene"],
        ascending=[False, True],
    )

    return out


def spatial_local_contrast(
    gene_names,
    Y,
    raw,
    cells,
    cell_delta,
    u,
):
    """
    Spatially local descriptive control.

    High-distortion cells (top quintile) are compared with cells from
    their surrounding tissue using a grid-based neighborhood scheme.

    This is intentionally simple and transparent. It does NOT claim to
    remove cell-type composition. Composition-aware interpretation is
    deferred to the cross-scale/object concordance analysis.
    """
    finite = np.isfinite(cell_delta)
    d = cell_delta[finite]

    threshold = np.quantile(d, 0.80)

    high = finite & (cell_delta >= threshold)

    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)

    # Adaptive spatial grid: ~25 bins along the shorter tissue axis.
    xr = np.nanmax(x) - np.nanmin(x)
    yr = np.nanmax(y) - np.nanmin(y)
    short = max(min(xr, yr), EPS)
    bw = short / 25.0

    gx = np.floor((x - np.nanmin(x)) / bw).astype(np.int64)
    gy = np.floor((y - np.nanmin(y)) / bw).astype(np.int64)

    high_bins = set(zip(gx[high], gy[high]))

    local = np.zeros(len(cells), dtype=bool)

    # include same and immediately adjacent grid bins
    expanded = set()
    for a, b in high_bins:
        for da in (-1, 0, 1):
            for db in (-1, 0, 1):
                expanded.add((a + da, b + db))

    for i, key in enumerate(zip(gx, gy)):
        if key in expanded:
            local[i] = True

    control = local & finite & (~high)

    rows = []

    for g, name in enumerate(gene_names):
        if np.mean(raw[g, finite] > 0) < MIN_DETECTION:
            continue

        yh = Y[g, high]
        yc = Y[g, control]

        if len(yh) == 0 or len(yc) == 0:
            continue

        rows.append({
            "u": u,
            "gene": name,
            "n_high": int(high.sum()),
            "n_local_control": int(control.sum()),
            "high_mean_logexpr": float(np.mean(yh)),
            "local_mean_logexpr": float(np.mean(yc)),
            "high_minus_local_logexpr": float(
                np.mean(yh) - np.mean(yc)
            ),
            "high_detection": float(
                np.mean(raw[g, high] > 0)
            ),
            "local_detection": float(
                np.mean(raw[g, control] > 0)
            ),
        })

    return pd.DataFrame(rows), high, control, bw


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "localization").mkdir(exist_ok=True)
    (OUT / "molecular").mkdir(exist_ok=True)

    print("=" * 100)
    print("SUTRA FIGURE 5 — GW BIOLOGY V4")
    print("=" * 100)

    cells = load_cells()
    n0 = len(cells)

    if n0 != 56510:
        raise RuntimeError(f"Expected PRCC N0=56510, got {n0}")

    genes, features = load_feature_names()
    features.to_csv(
        OUT / "prcc_features.csv",
        index=False,
    )

    raw = read_10x_expression()

    if raw.shape[1] != n0:
        raise RuntimeError("expression/cell count mismatch")

    Y, lib = normalize_log1p(raw)

    print(
        "Expression loaded:",
        raw.shape,
        "median library size:",
        float(np.median(lib)),
    )

    steps = read_steps()

    scale_summary = []
    assoc_all = []
    local_all = []

    for u in U_VALUES:
        print("\n" + "-" * 100)
        print("u =", u)

        step, u_actual, step_row = exact_target_step(
            steps, u
        )

        labels, cp_step, cp_path = reconstruct_labels_after_step(
            step, n0
        )

        expected_nodes = int(step_row["nodes_after"])
        observed_nodes = int(len(np.unique(labels)))

        print(
            "step", step,
            "actual u", u_actual,
            "checkpoint", cp_step,
            "objects", observed_nodes,
            "expected", expected_nodes,
        )

        if observed_nodes != expected_nodes:
            raise RuntimeError(
                f"Hierarchy replay mismatch at u={u}: "
                f"{observed_nodes} != {expected_nodes}"
            )

        gwp = GW / f"best_multistart_u_{u:.2f}_K{K}.npz"

        objmap, cell_delta, coverage = project_gw_to_cells(
            gwp,
            labels,
        )

        print(
            "dominant-component cell coverage:",
            coverage,
            "finite cells:",
            np.isfinite(cell_delta).sum(),
        )

        # Add object masses from exact cell membership.
        mass = pd.Series(labels).value_counts()
        objmap["cell_mass"] = (
            objmap["object_label"]
            .map(mass)
            .fillna(0)
            .astype(int)
        )

        objmap.to_csv(
            OUT / "localization" /
            f"prcc_object_distortion_u{u:.2f}.csv",
            index=False,
        )

        loc = cells[
            ["cell_index", "cell_id", "x", "y"]
        ].copy()

        loc["object_label"] = labels
        loc["gw_distortion"] = cell_delta

        # Keep the spatial file compact but exact.
        loc.to_parquet(
            OUT / "localization" /
            f"prcc_cell_distortion_u{u:.2f}.parquet",
            index=False,
        )

        assoc = gene_associations(
            genes,
            Y,
            raw,
            labels,
            cell_delta,
            u,
        )

        assoc.to_csv(
            OUT / "molecular" /
            f"gene_continuous_association_u{u:.2f}.csv",
            index=False,
        )

        assoc_all.append(assoc)

        local, high, control, bw = spatial_local_contrast(
            genes,
            Y,
            raw,
            cells,
            cell_delta,
            u,
        )

        local.to_csv(
            OUT / "molecular" /
            f"gene_spatial_local_control_u{u:.2f}.csv",
            index=False,
        )

        local_all.append(local)

        if abs(u - DISPLAY_U) < 1e-9:
            display = loc.copy()
            display["high_distortion_top20"] = high
            display["local_control"] = control

            display.to_parquet(
                OUT / "localization" /
                "prcc_display_u0.40.parquet",
                index=False,
            )

        with np.load(gwp, allow_pickle=False) as z:
            gw_best = float(z["best_gw_distance"])
            gw2_best = float(z["best_objective"])
            decomp = float(
                z["local_distortion_reconstruction_delta"]
            )

        scale_summary.append({
            "u_target": u,
            "u_disease": u_actual,
            "microstep": step,
            "checkpoint_start": cp_step,
            "n_objects_after": observed_nodes,
            "dominant_component_cell_coverage": coverage,
            "gw_distance": gw_best,
            "gw2_objective": gw2_best,
            "decomposition_error": decomp,
            "spatial_grid_width": bw,
        })

        print("\nTop convergent molecular associations:")
        print(
            assoc[
                [
                    "gene",
                    "cell_spearman",
                    "object_spearman_sqrt_mass",
                    "high_minus_low_logexpr",
                    "convergent_score",
                ]
            ].head(15).to_string(index=False)
        )

    scale_df = pd.DataFrame(scale_summary)
    scale_df.to_csv(
        OUT / "scale_summary.csv",
        index=False,
    )

    A = pd.concat(assoc_all, ignore_index=True)
    L = pd.concat(local_all, ignore_index=True)

    A.to_csv(
        OUT / "molecular" /
        "gene_continuous_association_all_scales.csv",
        index=False,
    )

    L.to_csv(
        OUT / "molecular" /
        "gene_spatial_local_control_all_scales.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Cross-scale molecular summary
    # ------------------------------------------------------------------

    merged = A.merge(
        L[
            [
                "u",
                "gene",
                "high_minus_local_logexpr",
            ]
        ],
        on=["u", "gene"],
        how="left",
    )

    # Directional agreement among:
    # cell continuous association,
    # object continuous association,
    # local spatial contrast.
    s1 = np.sign(merged["cell_spearman"])
    s2 = np.sign(merged["object_spearman_sqrt_mass"])
    s3 = np.sign(merged["high_minus_local_logexpr"])

    merged["three_way_direction_concordant"] = (
        (s1 == s2) & (s2 == s3) & (s1 != 0)
    )

    merged["three_way_score"] = np.where(
        merged["three_way_direction_concordant"],
        np.cbrt(
            np.abs(merged["cell_spearman"]) *
            np.abs(merged["object_spearman_sqrt_mass"]) *
            np.abs(merged["high_minus_local_logexpr"])
        ),
        0.0,
    )

    merged.to_csv(
        OUT / "molecular" /
        "gene_integrated_scale_associations.csv",
        index=False,
    )

    gene_summary = (
        merged.groupby("gene", as_index=False)
        .agg(
            n_scales=("u", "nunique"),
            n_three_way_concordant=(
                "three_way_direction_concordant",
                "sum",
            ),
            median_cell_spearman=(
                "cell_spearman",
                "median",
            ),
            median_object_spearman=(
                "object_spearman_sqrt_mass",
                "median",
            ),
            median_local_effect=(
                "high_minus_local_logexpr",
                "median",
            ),
            max_three_way_score=(
                "three_way_score",
                "max",
            ),
            median_three_way_score=(
                "three_way_score",
                "median",
            ),
        )
    )

    gene_summary = gene_summary.sort_values(
        [
            "n_three_way_concordant",
            "median_three_way_score",
            "max_three_way_score",
        ],
        ascending=False,
    )

    gene_summary.to_csv(
        OUT / "molecular" /
        "gene_cross_scale_summary.csv",
        index=False,
    )

    print("\n" + "=" * 100)
    print("CROSS-SCALE CANDIDATES")
    print("=" * 100)

    print(
        gene_summary.head(30).to_string(index=False)
    )

    provenance = {
        "version": "fig5_gw_biology_v4",
        "sample": "prcc",
        "gw_input": str(GW),
        "hierarchy_input": str(LEDGER),
        "level0_input": str(L0),
        "u_values": U_VALUES,
        "display_u": DISPLAY_U,
        "K": K,
        "expression_normalization":
            "cell library-size 1e4 followed by log1p",
        "interpretation": (
            "Descriptive molecular associations with PRCC-side "
            "GW relational distortion. Not causal driver inference "
            "and not population-level disease inference."
        ),
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nCOMPLETE")
    print(OUT)


if __name__ == "__main__":
    main()
