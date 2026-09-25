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

OUT = ROOT / "results" / "Fig5_Disease_Trajectories" / "structural_trajectory_v7"

U_VALUES = [0.08, 0.16, 0.24, 0.32, 0.40, 0.48, 0.56]

# Molecular analyses remain restricted to the scales already audited in V4.
MOLECULAR_U_VALUES = [0.32, 0.40, 0.48, 0.56]

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
    """
    Structural-only extension of the verified V4 GW→SUTRA→cell
    materialization.

    No expression matrix is loaded.
    No molecular associations are recomputed.
    """
    import json

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "localization").mkdir(exist_ok=True)

    cells = load_cells()
    n0 = len(cells)

    if n0 != 56510:
        raise RuntimeError(f"Unexpected PRCC N0: {n0}")

    steps = read_steps()

    scale_summary = []
    cell_wide = cells[
        ["cell_index", "cell_id", "x", "y"]
    ].copy()

    reference_ids = cell_wide["cell_id"].astype(str).to_numpy()
    reference_xy = cell_wide[["x", "y"]].to_numpy(float)

    for u in U_VALUES:
        print("\n" + "-" * 100)
        print("STRUCTURAL MATERIALIZATION u =", u)

        step, u_actual, step_row = exact_target_step(steps, u)

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

        if not gwp.exists():
            raise FileNotFoundError(gwp)

        objmap, cell_delta, coverage = project_gw_to_cells(
            gwp,
            labels,
        )

        finite = np.isfinite(cell_delta)

        print(
            "dominant-component cell coverage:",
            coverage,
            "finite cells:",
            int(finite.sum()),
        )

        # Exact object masses from Level-0 membership.
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

        # Strong invariant: all scale files must refer to the same
        # Level-0 cells in the same order and coordinates.
        ids = loc["cell_id"].astype(str).to_numpy()
        xy = loc[["x", "y"]].to_numpy(float)

        if not np.array_equal(ids, reference_ids):
            raise RuntimeError(
                f"Cell identity/order changed at u={u}"
            )

        if not np.allclose(
            xy, reference_xy,
            rtol=0.0, atol=1e-10,
            equal_nan=True,
        ):
            raise RuntimeError(
                f"Cell coordinates changed at u={u}"
            )

        loc["object_label"] = labels
        loc["gw_distortion"] = cell_delta

        loc.to_parquet(
            OUT / "localization" /
            f"prcc_cell_distortion_u{u:.2f}.parquet",
            index=False,
        )

        cell_wide[f"object_u{u:.2f}"] = labels
        cell_wide[f"gw_u{u:.2f}"] = cell_delta

        with np.load(gwp, allow_pickle=False) as z:
            gw_best = float(z["best_gw_distance"])
            gw2_best = float(z["best_objective"])
            decomp = float(
                z["local_distortion_reconstruction_delta"]
            )

            q = np.asarray(z["p_disease"], dtype=float)
            delta_landmark = np.asarray(
                z["disease_conditional_distortion"],
                dtype=float,
            )

            reconstructed_gw2 = float(
                np.sum(q * delta_landmark)
            )

        reconstruction_error = (
            reconstructed_gw2 - gw2_best
        )

        print(
            "GW =", gw_best,
            "GW^2 =", gw2_best,
            "sum(q*delta) =", reconstructed_gw2,
            "error =", reconstruction_error,
        )

        if abs(reconstruction_error) > 1e-8:
            raise RuntimeError(
                f"GW decomposition failed at u={u}: "
                f"{reconstruction_error}"
            )

        scale_summary.append({
            "u_target": float(u),
            "u_disease": float(u_actual),
            "microstep": int(step),
            "checkpoint_start": int(cp_step),
            "n_objects_after": int(observed_nodes),
            "dominant_component_cell_coverage":
                float(coverage),
            "n_finite_cells": int(finite.sum()),
            "gw_distance": gw_best,
            "gw2_objective": gw2_best,
            "landmark_reconstructed_gw2":
                reconstructed_gw2,
            "landmark_reconstruction_error":
                reconstruction_error,
            "stored_decomposition_error": decomp,
        })

    scale_df = pd.DataFrame(scale_summary)

    scale_df.to_csv(
        OUT / "scale_summary.csv",
        index=False,
    )

    cell_wide.to_parquet(
        OUT / "prcc_cell_multiscale_gw.parquet",
        index=False,
    )

    provenance = {
        "version": "structural_trajectory_v7",
        "sample": "prcc",
        "n_level0_cells": int(n0),
        "u_values": [float(x) for x in U_VALUES],
        "K": int(K),
        "structural_only": True,
        "expression_loaded": False,
        "molecular_analysis_recomputed": False,
        "description": (
            "Exact projection of frozen K=512 multistart "
            "GW conditional distortion through verified "
            "SUTRA hierarchy states to the same PRCC "
            "Level-0 cells at seven hierarchy coordinates."
        ),
        "interpretation_limit": (
            "Hierarchy coordinate is an organizational "
            "coordinate, not biological time."
        ),
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\n" + "=" * 100)
    print("STRUCTURAL V7 MATERIALIZATION COMPLETE")
    print("=" * 100)
    print(scale_df.to_string(index=False))
    print("\nWROTE:")
    print(OUT / "prcc_cell_multiscale_gw.parquet")


if __name__ == "__main__":
    main()
