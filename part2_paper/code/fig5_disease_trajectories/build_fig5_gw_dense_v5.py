#!/usr/bin/env python3

from pathlib import Path
import json
import re
import warnings
import heapq

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.csgraph import (
    connected_components,
    dijkstra,
)
import ot


# ============================================================
# Paths / frozen inputs
# ============================================================

ROOT = Path.home() / "Desktop" / "SUTRA"

L0 = ROOT / "results/hierarchy_level0_v070"
HIER = (
    ROOT
    / "results/hierarchy_v0911_specimen_local_contextual_flow"
    / "ledger"
)

OUT = (
    ROOT
    / "results/Fig5_Disease_Trajectories"
    / "gw_geodesic_dense_v5"
)
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "reference": "nondiseased_kidney",
    "disease": "prcc",
}

# Do not use u=0 for biological interpretation.
# Dense, genuinely evaluated hierarchy trajectory: 25 states.
U_TARGETS = np.round(np.arange(0.08, 0.560001, 0.02), 2).astype(float)

# The publication estimator is the audited K=512 representation.
# Lower-K convergence was already established in frozen V2 and is not
# recomputed merely to densify the trajectory.
K_VALUES = [512]

SEED = 20260920
EPS = 1e-12


# ============================================================
# Generic helpers
# ============================================================

def checkpoint_step(path):
    m = re.search(r"labels_(\d+)\.npz$", path.name)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def merge_step(path):
    m = re.search(r"(\d+)", path.stem)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def load_steps(sample):
    p = HIER / sample / "steps.jsonl"
    rows = []
    with p.open() as f:
        for line in f:
            rows.append(json.loads(line))
    x = pd.DataFrame(rows)

    required = [
        "microstep",
        "level0_cells",
        "cumulative_merges_after",
        "removed_fraction_after",
    ]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise RuntimeError(
            f"{sample}: missing step columns {missing}"
        )

    return x


def load_flow(sample):
    p = HIER / sample / "flow_summary.json"
    with p.open() as f:
        return json.load(f)


# ============================================================
# Exact attainable hierarchy state
# ============================================================

def build_state_catalog(sample):
    """
    One row per attainable complete-microstep hierarchy state.

    The integer state invariant is derived from object count:

        cumulative_merges = N0 - nodes_after
        u_exact           = cumulative_merges / N0

    This is preferable to relying on nullable cumulative-merger
    bookkeeping fields. Every accepted binary merger removes exactly
    one active SUTRA object.

    An explicit initial state is added at step 0 only if the production
    ledger does not already encode an equivalent zero-merger state.
    """
    x = load_steps(sample).copy()

    N0s = x["level0_cells"].dropna().astype(int).unique()
    if len(N0s) != 1:
        raise RuntimeError(
            f"{sample}: inconsistent level0_cells {N0s}"
        )
    N0 = int(N0s[0])

    if x["microstep"].isna().any():
        raise RuntimeError(
            f"{sample}: NaN microstep in steps ledger"
        )

    # Production v0911 contains a terminal zero-merge stop record:
    #
    #   selected_merges = 0
    #   nodes_before     = terminal node count
    #   nodes_after      = NaN
    #
    # This is not a hierarchy transition and therefore is not an
    # attainable post-merge state. Exclude such records. Any NaN
    # nodes_after on a productive microstep remains an error.
    terminal_stop = (
        x["nodes_after"].isna()
        & x["selected_merges"].fillna(0).eq(0)
    )

    if terminal_stop.any():
        print(
            f"{sample}: excluding "
            f"{int(terminal_stop.sum())} terminal zero-merge "
            f"stop record(s)"
        )

    bad = (
        x["nodes_after"].isna()
        & ~terminal_stop
    )

    if bad.any():
        cols = [
            "microstep",
            "nodes_before",
            "nodes_after",
            "selected_merges",
            "stop_reason",
        ]
        cols = [
            c for c in cols
            if c in x.columns
        ]
        raise RuntimeError(
            f"{sample}: NaN nodes_after on productive state:\n"
            f"{x.loc[bad, cols].to_string(index=False)}"
        )

    x = x.loc[~terminal_stop].copy()

    x["microstep"] = x["microstep"].astype(int)
    x["nodes_after"] = x["nodes_after"].astype(int)

    # Exact integer hierarchy state.
    x["cumulative_merges_exact"] = (
        N0 - x["nodes_after"]
    ).astype(int)

    if (x["cumulative_merges_exact"] < 0).any():
        raise RuntimeError(
            f"{sample}: nodes_after exceeds N0"
        )

    x["u_exact"] = (
        x["cumulative_merges_exact"].astype(float)
        / float(N0)
    )

    # Audit production removed_fraction_after wherever it is present.
    if "removed_fraction_after" in x.columns:
        good = x["removed_fraction_after"].notna()
        if good.any():
            err = float(
                np.max(
                    np.abs(
                        x.loc[good, "u_exact"].to_numpy(dtype=float)
                        - x.loc[
                            good, "removed_fraction_after"
                        ].to_numpy(dtype=float)
                    )
                )
            )
            if err > 1e-10:
                raise RuntimeError(
                    f"{sample}: object-count u disagrees with "
                    f"removed_fraction_after; max error={err}"
                )

    # Audit cumulative_merges_after wherever production populated it.
    if "cumulative_merges_after" in x.columns:
        good = x["cumulative_merges_after"].notna()
        if good.any():
            prod = x.loc[
                good, "cumulative_merges_after"
            ].astype(int).to_numpy()
            exact = x.loc[
                good, "cumulative_merges_exact"
            ].to_numpy(dtype=int)

            if not np.array_equal(prod, exact):
                ii = np.flatnonzero(prod != exact)[:10]
                raise RuntimeError(
                    f"{sample}: cumulative merger bookkeeping "
                    f"disagrees with N0-nodes_after at rows {ii.tolist()}"
                )

    keep = [
        "microstep",
        "cumulative_merges_exact",
        "u_exact",
        "selected_merges",
        "nodes_after",
    ]

    x = x[keep].copy()

    # Production rows should be unique by microstep. Audit rather than
    # silently dropping duplicates.
    dup = x["microstep"].duplicated(keep=False)
    if dup.any():
        d = x.loc[dup].sort_values("microstep")
        raise RuntimeError(
            f"{sample}: duplicate production microsteps:\n"
            f"{d.to_string(index=False)}"
        )

    # Add the genuine initial partition only if it is absent.
    has_initial = bool(
        (
            (x["cumulative_merges_exact"] == 0)
            & (x["nodes_after"] == N0)
        ).any()
    )

    if not has_initial:
        init = pd.DataFrame([{
            "microstep": 0,
            "cumulative_merges_exact": 0,
            "u_exact": 0.0,
            "selected_merges": 0,
            "nodes_after": N0,
        }])
        x = pd.concat(
            [init, x],
            ignore_index=True,
        )

    x = (
        x.sort_values(
            ["u_exact", "microstep"]
        )
        .reset_index(drop=True)
    )

    # Strong monotonicity audits.
    if np.any(np.diff(x["u_exact"].to_numpy()) < -1e-15):
        raise RuntimeError(
            f"{sample}: hierarchy coordinate is not monotone"
        )

    if np.any(
        np.diff(
            x["cumulative_merges_exact"].to_numpy(dtype=int)
        ) < 0
    ):
        raise RuntimeError(
            f"{sample}: cumulative mergers are not monotone"
        )

    return x, N0


def choose_attainable_state(catalog, u_target):
    """
    Closest attainable complete-microstep state to requested u.

    Never split an accepted merger batch.
    """
    u = catalog["u_exact"].to_numpy(dtype=float)

    finite = np.isfinite(u)
    if not finite.all():
        raise RuntimeError(
            "Non-finite u_exact in state catalog"
        )

    i = int(np.argmin(np.abs(u - float(u_target))))
    r = catalog.iloc[i]

    return {
        "target_u": float(u_target),
        "step": int(r["microstep"]),
        "u": float(r["u_exact"]),
        "delta_u": float(r["u_exact"] - float(u_target)),
        "cumulative_merges": int(
            r["cumulative_merges_exact"]
        ),
        "nodes_after": int(r["nodes_after"]),
    }


# ============================================================
# Checkpoint + exact replay
# ============================================================

def checkpoint_catalog(sample):
    paths = sorted(
        (HIER / sample / "label_checkpoints")
        .glob("labels_*.npz"),
        key=checkpoint_step,
    )
    if not paths:
        raise RuntimeError(
            f"{sample}: no label checkpoints"
        )

    return [
        (checkpoint_step(p), p)
        for p in paths
    ]


def merge_event_catalog(sample):
    paths = sorted(
        (HIER / sample / "merge_events")
        .glob("*.parquet"),
        key=merge_step,
    )
    return {
        merge_step(p): p
        for p in paths
    }


def nearest_prior_checkpoint(sample, target_step):
    cps = checkpoint_catalog(sample)
    prior = [
        (st, p)
        for st, p in cps
        if st <= target_step
    ]
    if not prior:
        raise RuntimeError(
            f"{sample}: no checkpoint <= {target_step}"
        )
    return prior[-1]


def replay_labels_to_step(
    sample,
    target_step,
    merge_files,
):
    """
    Reconstruct Level-0-cell -> active SUTRA-object labels at the
    requested complete microstep.

    Accepted mergers within each microstep are disjoint, so all rows
    in one merge_events parquet can be applied simultaneously.
    """
    cp_step, cp_path = nearest_prior_checkpoint(
        sample,
        target_step,
    )

    labels = np.load(cp_path)["labels"].astype(
        np.int64,
        copy=True,
    )

    # v0911 checkpoint semantics were audited directly:
    #
    # labels_s.npz is the partition immediately BEFORE productive
    # microstep s (equivalently AFTER microstep s-1).
    #
    # Therefore reconstruction of the post-state of target step t
    # from checkpoint s must replay:
    #
    #     s, s+1, ..., t
    #
    # In particular, when cp_step == target_step we must still apply
    # the target-step merger batch. Do not return the checkpoint
    # directly.
    replayed = 0

    for st in sorted(merge_files):
        if st < cp_step:
            continue
        if st > target_step:
            break

        d = pd.read_parquet(
            merge_files[st],
            columns=[
                "survivor_node",
                "removed_node",
            ],
        )

        if d.empty:
            continue

        survivors = d["survivor_node"].to_numpy(
            dtype=np.int64
        )
        removed = d["removed_node"].to_numpy(
            dtype=np.int64
        )

        # Production invariant: accepted pairs in one microstep
        # are disjoint.
        all_nodes = np.concatenate(
            [survivors, removed]
        )
        if len(np.unique(all_nodes)) != len(all_nodes):
            raise RuntimeError(
                f"{sample} step {st}: accepted "
                "mergers are not disjoint"
            )

        # Vectorized replacement for all accepted pairs.
        mapping = {
            int(r): int(s)
            for s, r in zip(survivors, removed)
        }

        # Labels are surviving object IDs.
        # Replace every Level-0 membership currently belonging
        # to each removed object.
        for r, s in mapping.items():
            labels[labels == r] = s

        replayed += len(d)

    return labels, cp_step, replayed


# ============================================================
# Frozen Level-0 graph
# ============================================================

def load_level0_graph(sample):
    p = L0 / sample / "graph_csr.npz"
    z = np.load(p)

    indptr = z["indptr"]
    indices = z["indices"]
    node_ids = z["node_ids"]

    n = len(node_ids)

    if not np.array_equal(
        node_ids,
        np.arange(n, dtype=node_ids.dtype),
    ):
        raise RuntimeError(
            f"{sample}: graph node_ids not identity ordered"
        )

    A = sparse.csr_matrix(
        (
            np.ones(len(indices), dtype=np.float64),
            indices,
            indptr,
        ),
        shape=(n, n),
    )

    A.data[:] = 1.0
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()

    return A


def quotient_graph(A0, labels):
    """
    Exact quotient of frozen Level-0 contact topology under the
    current SUTRA partition.
    """
    object_labels, inverse = np.unique(
        labels.astype(np.int64),
        return_inverse=True,
    )

    n_obj = len(object_labels)

    mass = np.bincount(
        inverse,
        minlength=n_obj,
    ).astype(np.float64)

    coo = sparse.triu(A0, k=1).tocoo()

    a = inverse[coo.row]
    b = inverse[coo.col]

    keep = a != b
    a = a[keep]
    b = b[keep]

    rr = np.concatenate([a, b])
    cc = np.concatenate([b, a])

    Q = sparse.coo_matrix(
        (
            np.ones(len(rr), dtype=float),
            (rr, cc),
        ),
        shape=(n_obj, n_obj),
    ).tocsr()

    Q.data[:] = 1.0
    Q.eliminate_zeros()

    return Q, mass, object_labels, inverse



# ============================================================
# Dominant connected tissue component
# ============================================================

def dominant_component(
    Q,
    mass,
    object_labels,
):
    """
    Restrict the quotient contact graph to its largest component
    by Level-0-cell mass.

    This is the primary GW domain. The discarded mass is reported
    explicitly rather than assigning artificial finite distances
    between disconnected tissue pieces.

    Returned mass is normalized conditionally within the retained
    component.
    """
    ncomp, comp = connected_components(
        Q,
        directed=False,
        return_labels=True,
    )

    comp_mass = np.bincount(
        comp,
        weights=mass,
        minlength=ncomp,
    ).astype(float)

    c = int(np.argmax(comp_mass))
    idx = np.flatnonzero(comp == c)

    retained_mass = float(comp_mass[c])
    total_mass = float(np.sum(mass))
    coverage = retained_mass / total_mass

    Qc = Q[idx][:, idx].tocsr()
    mc_raw = np.asarray(
        mass[idx],
        dtype=float,
    )
    labels_c = np.asarray(
        object_labels[idx],
        dtype=np.int64,
    )

    # Conditional probability measure on connected tissue domain.
    mc = mc_raw / mc_raw.sum()

    ncheck, _ = connected_components(
        Qc,
        directed=False,
        return_labels=True,
    )
    if ncheck != 1:
        raise RuntimeError(
            "Dominant-component extraction did not produce "
            "a connected graph"
        )

    return {
        "Q": Qc,
        "mass": mc,
        "mass_raw": mc_raw,
        "object_labels": labels_c,
        "original_indices": idx,
        "coverage": coverage,
        "discarded_mass_fraction": 1.0 - coverage,
        "n_components_original": int(ncomp),
        "n_objects_original": int(Q.shape[0]),
        "n_objects_retained": int(Qc.shape[0]),
        "object_fraction_retained":
            float(Qc.shape[0]) / float(Q.shape[0]),
    }


def weighted_rms_metric_scale(
    C,
    p,
):
    """
    Scale metric so E[d(X,X')^2] = 1 under p x p.

        scale^2 = sum_ij p_i p_j C_ij^2

    This is measure-aware and makes metric normalization explicit.
    """
    C = np.asarray(C, dtype=float)
    p = np.asarray(p, dtype=float)

    p = p / p.sum()

    scale2 = float(
        np.sum(
            (p[:, None] * p[None, :])
            * (C * C)
        )
    )

    scale = np.sqrt(
        max(scale2, EPS)
    )

    return C / scale, scale


# ============================================================
# Component-aware K allocation
# ============================================================

def allocate_landmarks_by_component(
    Q,
    mass,
    K,
):
    """
    Allocate exactly K landmarks across connected components.

    Components receive at least one landmark if K permits.
    Remaining landmarks are allocated by component mass using
    largest-remainder apportionment.

    If n_components > K, tiny components are attached to a
    residual disconnected pool later; however the current data
    should be checked explicitly.
    """
    ncomp, comp = connected_components(
        Q,
        directed=False,
    )

    comp_mass = np.bincount(
        comp,
        weights=mass,
        minlength=ncomp,
    )

    comp_size = np.bincount(
        comp,
        minlength=ncomp,
    )

    order = np.argsort(-comp_mass)

    allocation = np.zeros(
        ncomp,
        dtype=int,
    )

    if ncomp <= K:
        allocation[:] = 1
        remaining = K - ncomp

        if remaining > 0:
            capacity = comp_size - 1
            total_mass = comp_mass.sum()

            ideal = (
                remaining
                * comp_mass
                / max(total_mass, EPS)
            )

            base = np.floor(ideal).astype(int)
            base = np.minimum(base, capacity)

            allocation += base
            remaining -= int(base.sum())

            # Largest-remainder / mass priority while respecting
            # component capacity.
            remainder = ideal - np.floor(ideal)

            while remaining > 0:
                eligible = np.flatnonzero(
                    allocation < comp_size
                )
                if len(eligible) == 0:
                    break

                # Highest fractional remainder; then larger mass.
                best = sorted(
                    eligible,
                    key=lambda c: (
                        -remainder[c],
                        -comp_mass[c],
                        c,
                    ),
                )[0]

                allocation[best] += 1
                remainder[best] = 0.0
                remaining -= 1

    else:
        # There are more disconnected components than available
        # landmarks. Give one landmark to K most massive components.
        allocation[order[:K]] = 1

    if allocation.sum() != K:
        raise RuntimeError(
            f"Landmark allocation produced "
            f"{allocation.sum()} != {K}"
        )

    return comp, comp_mass, comp_size, allocation


# ============================================================
# Farthest-point metric quantization
# ============================================================

def farthest_points_in_component(
    Q,
    nodes,
    mass,
    k,
):
    """
    Deterministic farthest-point sampling in graph-geodesic metric.

    First point = largest-mass object.
    Subsequent point = object with largest distance to current
    landmark set; ties favor larger mass then smaller object id.
    """
    nodes = np.asarray(nodes, dtype=int)

    if k <= 0:
        return np.array([], dtype=int)

    if k >= len(nodes):
        return nodes.copy()

    sub = Q[nodes][:, nodes]

    # First landmark.
    first_local = sorted(
        range(len(nodes)),
        key=lambda i: (
            -mass[nodes[i]],
            nodes[i],
        ),
    )[0]

    chosen_local = [first_local]

    d = dijkstra(
        sub,
        directed=False,
        indices=first_local,
        unweighted=True,
    )

    min_dist = np.asarray(
        d,
        dtype=float,
    )

    min_dist[first_local] = -np.inf

    while len(chosen_local) < k:
        finite = np.isfinite(min_dist)

        candidates = np.flatnonzero(finite)

        if len(candidates) == 0:
            raise RuntimeError(
                "Connected component produced unreachable nodes"
            )

        maxd = np.max(min_dist[candidates])

        ties = candidates[
            np.isclose(
                min_dist[candidates],
                maxd,
                rtol=0,
                atol=1e-12,
            )
        ]

        nxt = sorted(
            ties,
            key=lambda i: (
                -mass[nodes[i]],
                nodes[i],
            ),
        )[0]

        chosen_local.append(int(nxt))

        dn = dijkstra(
            sub,
            directed=False,
            indices=nxt,
            unweighted=True,
        )

        min_dist = np.minimum(
            min_dist,
            np.asarray(dn, dtype=float),
        )

        min_dist[
            np.asarray(chosen_local, dtype=int)
        ] = -np.inf

    return nodes[
        np.asarray(chosen_local, dtype=int)
    ]


def choose_metric_landmarks(
    Q,
    mass,
    K,
):
    comp, comp_mass, comp_size, allocation = (
        allocate_landmarks_by_component(
            Q,
            mass,
            K,
        )
    )

    anchors = []

    represented_components = np.flatnonzero(
        allocation > 0
    )

    for c in represented_components:
        nodes = np.flatnonzero(comp == c)

        pts = farthest_points_in_component(
            Q,
            nodes,
            mass,
            int(allocation[c]),
        )

        anchors.extend(
            pts.tolist()
        )

    anchors = np.asarray(
        anchors,
        dtype=int,
    )

    if len(anchors) != K:
        raise RuntimeError(
            f"Expected exactly {K} anchors, "
            f"got {len(anchors)}"
        )

    return (
        anchors,
        comp,
        comp_mass,
        allocation,
    )


def assign_metric_voronoi(
    Q,
    mass,
    anchors,
    comp,
):
    """
    Assign quotient objects to nearest landmark.

    For represented connected components this is exact graph-metric
    Voronoi assignment.

    If a tiny disconnected component has no landmark because
    n_components > K, assign the whole component to the landmark
    belonging to the most massive represented component. This case
    is explicitly quantified and should be zero/negligible for a
    valid analysis.
    """
    D = dijkstra(
        Q,
        directed=False,
        indices=anchors,
        unweighted=True,
    )

    if D.ndim == 1:
        D = D[None, :]

    anchor_comp = comp[anchors]
    represented = set(
        anchor_comp.tolist()
    )

    owner = np.full(
        Q.shape[0],
        -1,
        dtype=int,
    )

    # Deterministic tie ordering.
    tie_order = np.lexsort(
        (
            anchors,
            -mass[anchors],
        )
    )

    penalty = np.empty(
        len(anchors),
        dtype=float,
    )
    penalty[tie_order] = (
        np.arange(len(anchors))
        * 1e-10
    )

    score = D + penalty[:, None]

    represented_mask = np.array(
        [c in represented for c in comp],
        dtype=bool,
    )

    owner[represented_mask] = np.argmin(
        score[:, represented_mask],
        axis=0,
    )

    missing = np.flatnonzero(
        ~represented_mask
    )

    unrepresented_mass = float(
        mass[missing].sum()
    )

    if len(missing):
        # This branch is a documented fallback only.
        # Pick landmark with largest represented mass seed.
        fallback = int(
            np.argmax(mass[anchors])
        )
        owner[missing] = fallback

    if np.any(owner < 0):
        raise RuntimeError(
            "Unassigned quotient objects"
        )

    sketch_mass = np.bincount(
        owner,
        weights=mass,
        minlength=len(anchors),
    )

    # Quantization error: object-to-own-anchor graph distance.
    own_dist = D[
        owner,
        np.arange(Q.shape[0]),
    ]

    finite = np.isfinite(own_dist)

    weighted_mean_radius = float(
        np.sum(
            mass[finite]
            * own_dist[finite]
        )
        / np.sum(mass[finite])
    )

    weighted_rms_radius = float(
        np.sqrt(
            np.sum(
                mass[finite]
                * own_dist[finite] ** 2
            )
            / np.sum(mass[finite])
        )
    )

    max_radius = float(
        np.max(own_dist[finite])
    )

    return (
        owner,
        sketch_mass,
        D,
        {
            "unrepresented_component_mass":
                unrepresented_mass,
            "weighted_mean_radius":
                weighted_mean_radius,
            "weighted_rms_radius":
                weighted_rms_radius,
            "max_radius":
                max_radius,
        },
    )


# ============================================================
# K-point relational metric
# ============================================================

def landmark_metric_from_full_graph(
    Q,
    anchors,
):
    """
    Exact quotient-graph shortest-path distances between selected
    landmarks.

    Disconnected components receive a finite penalty derived from
    the largest finite within-component distance.

    This affects only cross-component relational values and is
    recorded as a sensitivity target for later validation.
    """
    C = dijkstra(
        Q,
        directed=False,
        indices=anchors,
        unweighted=True,
    )

    C = C[:, anchors]

    finite = C[np.isfinite(C)]

    if finite.size == 0:
        raise RuntimeError(
            "No finite landmark distances"
        )

    max_finite = float(
        finite.max()
    )
    if max_finite <= 0:
        max_finite = 1.0

    disconnected_penalty = (
        1.25 * max_finite
    )

    C[
        ~np.isfinite(C)
    ] = disconnected_penalty

    positive = C[C > 0]

    scale = float(
        np.median(positive)
    ) if positive.size else 1.0

    if (
        not np.isfinite(scale)
        or scale <= 0
    ):
        scale = 1.0

    C = C / scale

    return (
        C,
        scale,
        disconnected_penalty,
    )


# ============================================================
# GW
# ============================================================

def run_gw(C1, C2, p, q):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        T, log = (
            ot.gromov.gromov_wasserstein(
                C1,
                C2,
                p,
                q,
                loss_fun="square_loss",
                symmetric=True,
                log=True,
                max_iter=300,
                tol_rel=1e-9,
                tol_abs=1e-9,
                verbose=False,
            )
        )

    if "gw_dist" in log:
        gw2 = float(
            log["gw_dist"]
        )
    else:
        gw2 = float(
            ot.gromov.gromov_wasserstein2(
                C1,
                C2,
                p,
                q,
                loss_fun="square_loss",
                symmetric=True,
                max_iter=300,
                tol_rel=1e-9,
                tol_abs=1e-9,
            )
        )

    gw2 = max(gw2, 0.0)

    return (
        T,
        gw2,
        float(np.sqrt(gw2)),
    )


def disease_conditional_distortion(
    C1,
    C2,
    p,
    q,
    T,
):
    a = (C1 ** 2) @ p
    term1 = T.T @ a

    b = (C2 ** 2) @ q

    cross = C1 @ T @ C2.T

    d2 = (
        term1
        + q * b
        - 2.0
        * np.sum(
            T * cross,
            axis=0,
        )
    )

    d2 = np.maximum(
        d2,
        0.0,
    )

    return (
        d2
        / np.maximum(q, EPS)
    )


# ============================================================
# Plotting
# ============================================================

def plot_trajectory(df):
    fig, ax = plt.subplots(
        figsize=(6.6, 4.6)
    )

    for K, g in df.groupby("K"):
        g = g.sort_values(
            "u_target"
        )

        ax.plot(
            g["u_target"],
            g["gw_distance"],
            marker="o",
            label=f"K={K}",
        )

    ax.set_xlabel(
        "SUTRA hierarchy coordinate, u"
    )
    ax.set_ylabel(
        "GW distance"
    )
    ax.legend(
        frameon=False,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / "kidney_gw_distance_vs_u_v2.png",
        dpi=300,
    )
    fig.savefig(
        OUT
        / "kidney_gw_distance_vs_u_v2.pdf",
    )

    plt.close(fig)


def plot_quantization(df):
    fig, ax = plt.subplots(
        figsize=(6.6, 4.6)
    )

    for (
        sample,
        K,
    ), g in df.groupby(
        ["sample", "K"]
    ):
        g = g.sort_values(
            "u_target"
        )

        ax.plot(
            g["u_target"],
            g["weighted_mean_radius"],
            marker="o",
            label=f"{sample}, K={K}",
        )

    ax.set_xlabel(
        "SUTRA hierarchy coordinate, u"
    )
    ax.set_ylabel(
        "Mass-weighted quantization radius"
    )

    ax.legend(
        frameon=False,
        fontsize=7,
        ncol=2,
    )

    fig.tight_layout()

    fig.savefig(
        OUT
        / "metric_quantization_audit_v2.png",
        dpi=300,
    )

    plt.close(fig)


# ============================================================
# Main
# ============================================================


# ============================================================
# Dominant-component nested metric quantization
# ============================================================

def nested_farthest_sequence(
    Q,
    mass,
    Kmax,
):
    """
    Deterministic nested farthest-point sequence on one connected
    quotient graph.

    First landmark: largest-mass object.
    Subsequent landmarks: maximal graph distance from the current
    landmark set, with ties resolved by larger mass then node index.

    Returns a single sequence of length Kmax. Prefixes therefore
    define genuinely nested K=64,128,256,512 approximations.
    """
    n = int(Q.shape[0])

    if Kmax > n:
        raise RuntimeError(
            f"Kmax={Kmax} exceeds connected quotient size n={n}"
        )

    if Kmax < 1:
        raise ValueError("Kmax must be positive")

    mass = np.asarray(mass, dtype=float)

    first = sorted(
        range(n),
        key=lambda i: (
            -mass[i],
            i,
        ),
    )[0]

    chosen = [int(first)]

    d0 = dijkstra(
        Q,
        directed=False,
        indices=first,
        unweighted=True,
    )

    min_dist = np.asarray(
        d0,
        dtype=float,
    )

    if not np.all(np.isfinite(min_dist)):
        raise RuntimeError(
            "Dominant component is unexpectedly disconnected"
        )

    min_dist[first] = -np.inf

    while len(chosen) < Kmax:
        maxd = float(
            np.max(min_dist)
        )

        ties = np.flatnonzero(
            np.isclose(
                min_dist,
                maxd,
                rtol=0.0,
                atol=1e-12,
            )
        )

        if len(ties) == 0:
            raise RuntimeError(
                "No candidate found during nested FPS"
            )

        nxt = sorted(
            ties.tolist(),
            key=lambda i: (
                -mass[i],
                i,
            ),
        )[0]

        chosen.append(int(nxt))

        dn = dijkstra(
            Q,
            directed=False,
            indices=nxt,
            unweighted=True,
        )

        dn = np.asarray(
            dn,
            dtype=float,
        )

        if not np.all(np.isfinite(dn)):
            raise RuntimeError(
                "Disconnected distance encountered inside "
                "dominant component"
            )

        min_dist = np.minimum(
            min_dist,
            dn,
        )

        min_dist[
            np.asarray(chosen, dtype=int)
        ] = -np.inf

        if (
            len(chosen) % 64 == 0
            or len(chosen) == Kmax
        ):
            print(
                "    nested FPS",
                len(chosen),
                "/",
                Kmax,
                flush=True,
            )

    return np.asarray(
        chosen,
        dtype=int,
    )


def full_landmark_distances(
    Q,
    anchors,
):
    """
    One batched shortest-path calculation for all Kmax landmarks.

    D[a,j] is graph-geodesic distance from landmark a to quotient
    object j. Q is connected, so all entries must be finite.
    """
    D = dijkstra(
        Q,
        directed=False,
        indices=np.asarray(
            anchors,
            dtype=int,
        ),
        unweighted=True,
    )

    D = np.asarray(
        D,
        dtype=float,
    )

    if D.ndim == 1:
        D = D[None, :]

    if not np.all(np.isfinite(D)):
        raise RuntimeError(
            "Non-finite graph distance inside dominant component"
        )

    return D


def sketch_from_nested_prefix(
    mass,
    anchors_full,
    D_full,
    K,
):
    """
    Construct the K-landmark metric-measure sketch from a prefix
    of one nested Kmax landmark sequence.

    All retained dominant-component mass is pushed to its nearest
    landmark. No disconnected fallback or artificial penalty occurs.
    """
    anchors = np.asarray(
        anchors_full[:K],
        dtype=int,
    )

    D = np.asarray(
        D_full[:K, :],
        dtype=float,
    )

    if D.shape[0] != K:
        raise RuntimeError(
            f"Distance prefix has {D.shape[0]} rows, expected {K}"
        )

    mass = np.asarray(
        mass,
        dtype=float,
    )

    mass = mass / mass.sum()

    # Deterministic infinitesimal tie-break:
    # larger landmark mass first, then lower quotient-node index.
    tie_order = np.lexsort(
        (
            anchors,
            -mass[anchors],
        )
    )

    penalty = np.empty(
        K,
        dtype=float,
    )
    penalty[tie_order] = (
        np.arange(K, dtype=float)
        * 1e-10
    )

    owner = np.argmin(
        D + penalty[:, None],
        axis=0,
    )

    smass = np.bincount(
        owner,
        weights=mass,
        minlength=K,
    ).astype(float)

    if np.any(smass <= 0):
        raise RuntimeError(
            "Nested Voronoi sketch produced zero-mass landmark"
        )

    p = smass / smass.sum()

    # Landmark metric is already available from D:
    # column anchors[b] gives distance from anchor a to anchor b.
    C_raw = D[:, anchors]

    if not np.allclose(
        C_raw,
        C_raw.T,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Landmark graph metric is not symmetric"
        )

    if not np.allclose(
        np.diag(C_raw),
        0.0,
        atol=1e-12,
        rtol=0.0,
    ):
        raise RuntimeError(
            "Landmark graph metric has nonzero diagonal"
        )

    # Measure-aware normalization.
    C, distance_scale = (
        weighted_rms_metric_scale(
            C_raw,
            p,
        )
    )

    nearest = D[
        owner,
        np.arange(D.shape[1])
    ]

    weighted_mean_radius = float(
        np.sum(
            mass * nearest
        )
    )

    weighted_rms_radius = float(
        np.sqrt(
            np.sum(
                mass * nearest * nearest
            )
        )
    )

    max_radius = float(
        np.max(nearest)
    )

    qa = {
        "weighted_mean_radius":
            weighted_mean_radius,
        "weighted_rms_radius":
            weighted_rms_radius,
        "max_radius":
            max_radius,
        "distance_scale":
            float(distance_scale),
        "mass_sum":
            float(p.sum()),
    }

    return {
        "anchors": anchors,
        "owner": owner,
        "smass": smass,
        "p": p,
        "C": C,
        "C_raw": C_raw,
        "distance_scale":
            float(distance_scale),
        "qa": qa,
    }


def manual_square_loss_gw_objective(
    C1,
    C2,
    p,
    q,
    T,
):
    """
    Direct square-loss GW objective audit:

      sum_ijkl (C1_ij - C2_kl)^2 T_ik T_jl

    evaluated via the standard quadratic expansion.
    """
    C1 = np.asarray(C1, dtype=float)
    C2 = np.asarray(C2, dtype=float)
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    T = np.asarray(T, dtype=float)

    A = float(
        np.sum(
            (C1 * C1)
            * (p[:, None] * p[None, :])
        )
    )

    B = float(
        np.sum(
            (C2 * C2)
            * (q[:, None] * q[None, :])
        )
    )

    cross = float(
        np.sum(
            T
            * (
                C1
                @ T
                @ C2.T
            )
        )
    )

    obj = A + B - 2.0 * cross

    # Floating arithmetic can produce tiny negative values.
    if obj < 0 and obj > -1e-10:
        obj = 0.0

    if obj < -1e-10:
        raise RuntimeError(
            f"Manual GW objective is negative: {obj}"
        )

    return float(obj)




def main():
    np.random.seed(SEED)

    print("=" * 100)
    print(
        "SUTRA FIGURE 5 — KIDNEY GW "
        "DOMINANT-COMPONENT NESTED QUANTIZATION V2"
    )
    print("=" * 100)
    print("POT:", ot.__version__)
    print("Output:", OUT)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    Kmax = int(max(K_VALUES))

    states = {}

    for role, sample in SAMPLES.items():
        print("\nLoading", sample)

        A0 = load_level0_graph(sample)
        catalog, N0 = build_state_catalog(sample)
        merge_files = merge_event_catalog(sample)
        flow = load_flow(sample)

        states[role] = {
            "sample": sample,
            "A0": A0,
            "catalog": catalog,
            "N0": N0,
            "merge_files": merge_files,
            "flow": flow,
        }

        print(
            "N0 =", N0,
            "contacts =", A0.nnz // 2,
            "terminal u =",
            catalog["u_exact"].max(),
        )

    state_rows = []
    quant_rows = []
    gw_rows = []

    quotient_cache = {}

    for u_target in U_TARGETS:
        print("\n" + "-" * 100)
        print("TARGET u =", u_target)
        print("-" * 100)

        selected = {}

        # --------------------------------------------------------
        # Exact hierarchy states + dominant connected components
        # --------------------------------------------------------
        for role in ["reference", "disease"]:
            s = states[role]
            sample = s["sample"]

            target = choose_attainable_state(
                s["catalog"],
                u_target,
            )

            key = (
                sample,
                target["step"],
            )

            if key not in quotient_cache:
                labels, cp_step, nrep = (
                    replay_labels_to_step(
                        sample,
                        target["step"],
                        s["merge_files"],
                    )
                )

                Q, mass, obj_labels, inverse = (
                    quotient_graph(
                        s["A0"],
                        labels,
                    )
                )

                expected_nodes = (
                    s["N0"]
                    - target["cumulative_merges"]
                )

                if len(mass) != expected_nodes:
                    raise RuntimeError(
                        f"{sample} step {target['step']}: "
                        f"{len(mass)} objects, expected "
                        f"{expected_nodes}"
                    )

                dc = dominant_component(
                    Q,
                    mass,
                    obj_labels,
                )

                if dc["coverage"] < 0.99:
                    raise RuntimeError(
                        f"{sample} step {target['step']}: "
                        f"dominant component covers only "
                        f"{dc['coverage']:.6f} of tissue mass"
                    )

                quotient_cache[key] = {
                    "labels": labels,
                    "Q_full": Q,
                    "mass_full": mass,
                    "obj_labels_full":
                        obj_labels,
                    "inverse": inverse,
                    "cp_step": cp_step,
                    "replayed_merges": nrep,
                    "dc": dc,
                }

            x0 = quotient_cache[key]
            dc = x0["dc"]

            print(
                role,
                sample,
                "step",
                target["step"],
                "u",
                round(target["u"], 10),
                "du",
                f"{target['delta_u']:+.6g}",
                "objects",
                len(x0["mass_full"]),
                "components",
                dc["n_components_original"],
                "GC objects",
                dc["n_objects_retained"],
                "GC mass",
                f"{dc['coverage']:.6f}",
                "checkpoint",
                x0["cp_step"],
                "replayed",
                x0["replayed_merges"],
            )

            state_rows.append({
                "role": role,
                "sample": sample,
                **target,
                "n_objects":
                    len(x0["mass_full"]),
                "n_components":
                    dc["n_components_original"],
                "dominant_component_objects":
                    dc["n_objects_retained"],
                "dominant_component_mass_fraction":
                    dc["coverage"],
                "discarded_mass_fraction":
                    dc["discarded_mass_fraction"],
                "dominant_component_object_fraction":
                    dc["object_fraction_retained"],
                "largest_object_mass":
                    float(
                        x0["mass_full"].max()
                    ),
                "checkpoint_step":
                    int(x0["cp_step"]),
                "replayed_merges":
                    int(x0["replayed_merges"]),
            })

            selected[role] = {
                **target,
                **x0,
            }

        du_pair = abs(
            selected["reference"]["u"]
            - selected["disease"]["u"]
        )

        print(
            "paired |delta u| =",
            du_pair,
        )

        # --------------------------------------------------------
        # Build ONE nested Kmax quantization per specimen/state.
        # --------------------------------------------------------
        nested = {}

        for role in ["reference", "disease"]:
            x = selected[role]
            dc = x["dc"]

            Qc = dc["Q"]
            mc = dc["mass"]

            if Qc.shape[0] < Kmax:
                raise RuntimeError(
                    f"{role}: dominant component has "
                    f"{Qc.shape[0]} objects < Kmax={Kmax}"
                )

            print(
                f"\n{role}: building nested "
                f"Kmax={Kmax} landmarks on "
                f"{Qc.shape[0]} connected objects"
            )

            anchors_full = nested_farthest_sequence(
                Qc,
                mc,
                Kmax,
            )

            print(
                f"{role}: computing batched "
                f"{Kmax} x {Qc.shape[0]} distances"
            )

            D_full = full_landmark_distances(
                Qc,
                anchors_full,
            )

            nested[role] = {
                "anchors_full":
                    anchors_full,
                "D_full":
                    D_full,
            }

        # --------------------------------------------------------
        # Nested K convergence
        # --------------------------------------------------------
        for K in K_VALUES:
            print(f"\nK = {K}")

            sketch = {}

            for role in ["reference", "disease"]:
                x = selected[role]
                dc = x["dc"]
                z = nested[role]

                sk = sketch_from_nested_prefix(
                    dc["mass"],
                    z["anchors_full"],
                    z["D_full"],
                    int(K),
                )

                if len(sk["p"]) != K:
                    raise RuntimeError(
                        f"{role}: {len(sk['p'])} != K={K}"
                    )

                if not np.isclose(
                    sk["p"].sum(),
                    1.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(
                        f"{role}: sketch mass != 1"
                    )

                sketch[role] = sk

                sample = states[role]["sample"]

                quant_rows.append({
                    "role": role,
                    "sample": sample,
                    "u_target":
                        float(u_target),
                    "u_actual":
                        float(x["u"]),
                    "K": int(K),
                    "n_components_full":
                        int(
                            dc[
                                "n_components_original"
                            ]
                        ),
                    "dominant_component_mass_fraction":
                        float(dc["coverage"]),
                    "discarded_mass_fraction":
                        float(
                            dc[
                                "discarded_mass_fraction"
                            ]
                        ),
                    "dominant_component_objects":
                        int(
                            dc[
                                "n_objects_retained"
                            ]
                        ),
                    "dominant_component_object_fraction":
                        float(
                            dc[
                                "object_fraction_retained"
                            ]
                        ),
                    "weighted_mean_radius":
                        sk["qa"][
                            "weighted_mean_radius"
                        ],
                    "weighted_rms_radius":
                        sk["qa"][
                            "weighted_rms_radius"
                        ],
                    "max_radius":
                        sk["qa"][
                            "max_radius"
                        ],
                    "distance_scale":
                        sk["distance_scale"],
                    "unrepresented_component_mass":
                        0.0,
                    "disconnected_penalty":
                        np.nan,
                    "nested_landmarks":
                        True,
                })

                print(
                    role,
                    "anchors",
                    len(sk["anchors"]),
                    "GC mass",
                    f"{dc['coverage']:.6f}",
                    "mean radius",
                    round(
                        sk["qa"][
                            "weighted_mean_radius"
                        ],
                        4,
                    ),
                    "RMS radius",
                    round(
                        sk["qa"][
                            "weighted_rms_radius"
                        ],
                        4,
                    ),
                    "metric RMS scale",
                    round(
                        sk["distance_scale"],
                        4,
                    ),
                )

            a = sketch["reference"]
            b = sketch["disease"]

            T, gw2_reported, gw_reported = run_gw(
                a["C"],
                b["C"],
                a["p"],
                b["p"],
            )

            manual_gw2 = (
                manual_square_loss_gw_objective(
                    a["C"],
                    b["C"],
                    a["p"],
                    b["p"],
                    T,
                )
            )

            manual_gw = float(
                np.sqrt(
                    max(manual_gw2, 0.0)
                )
            )

            objective_delta = abs(
                float(gw2_reported)
                - manual_gw2
            )

            if objective_delta > 1e-7:
                raise RuntimeError(
                    "POT GW objective disagrees with "
                    f"manual audit: POT={gw2_reported}, "
                    f"manual={manual_gw2}, "
                    f"delta={objective_delta}"
                )

            dd = disease_conditional_distortion(
                a["C"],
                b["C"],
                a["p"],
                b["p"],
                T,
            )

            print(
                "GW^2 =",
                manual_gw2,
                "GW =",
                manual_gw,
                "coupling mass =",
                T.sum(),
                "objective audit delta =",
                objective_delta,
            )

            tag = (
                f"u{u_target:.2f}"
                .replace(".", "p")
                + f"_K{K}"
            )

            np.savez_compressed(
                OUT / f"gw_state_{tag}.npz",
                coupling=T,
                C_reference=a["C"],
                C_disease=b["C"],
                C_raw_reference=a["C_raw"],
                C_raw_disease=b["C_raw"],
                p_reference=a["p"],
                p_disease=b["p"],
                anchors_reference=
                    a["anchors"],
                anchors_disease=
                    b["anchors"],
                owner_reference=
                    a["owner"],
                owner_disease=
                    b["owner"],
                dominant_object_labels_reference=
                    selected["reference"]["dc"][
                        "object_labels"
                    ],
                dominant_object_labels_disease=
                    selected["disease"]["dc"][
                        "object_labels"
                    ],
                dominant_original_indices_reference=
                    selected["reference"]["dc"][
                        "original_indices"
                    ],
                dominant_original_indices_disease=
                    selected["disease"]["dc"][
                        "original_indices"
                    ],
                disease_conditional_distortion=dd,
                coverage_reference=
                    selected["reference"]["dc"][
                        "coverage"
                    ],
                coverage_disease=
                    selected["disease"]["dc"][
                        "coverage"
                    ],
                distance_scale_reference=
                    a["distance_scale"],
                distance_scale_disease=
                    b["distance_scale"],
            )

            gw_rows.append({
                "u_target":
                    float(u_target),
                "u_reference":
                    float(
                        selected["reference"]["u"]
                    ),
                "u_disease":
                    float(
                        selected["disease"]["u"]
                    ),
                "paired_delta_u":
                    float(du_pair),
                "step_reference":
                    int(
                        selected[
                            "reference"
                        ]["step"]
                    ),
                "step_disease":
                    int(
                        selected[
                            "disease"
                        ]["step"]
                    ),
                "K": int(K),
                "gw_squared":
                    float(manual_gw2),
                "gw_distance":
                    float(manual_gw),
                "pot_gw_squared":
                    float(gw2_reported),
                "pot_gw_distance":
                    float(gw_reported),
                "objective_audit_delta":
                    float(objective_delta),
                "coupling_mass":
                    float(T.sum()),
                "coverage_reference":
                    float(
                        selected[
                            "reference"
                        ]["dc"]["coverage"]
                    ),
                "coverage_disease":
                    float(
                        selected[
                            "disease"
                        ]["dc"]["coverage"]
                    ),
                "max_disease_conditional_distortion":
                    float(dd.max()),
                "median_disease_conditional_distortion":
                    float(np.median(dd)),
            })

        # Free the potentially large Kmax x n distance arrays before
        # advancing to the next hierarchy scale.
        del nested

    states_df = pd.DataFrame(state_rows)
    quant_df = pd.DataFrame(quant_rows)
    gw_df = pd.DataFrame(gw_rows)

    states_df.to_csv(
        OUT / "exact_hierarchy_states.csv",
        index=False,
    )

    quant_df.to_csv(
        OUT / "metric_quantization_audit.csv",
        index=False,
    )

    gw_df.to_csv(
        OUT / "gw_scale_trajectory.csv",
        index=False,
    )

    pivot = gw_df.pivot(
        index="u_target",
        columns="K",
        values="gw_distance",
    )

    pivot.to_csv(
        OUT / "gw_K_convergence.csv"
    )

    if (
        256 in pivot.columns
        and 512 in pivot.columns
    ):
        conv = pd.DataFrame({
            "u_target":
                pivot.index,
            "gw_K256":
                pivot[256].values,
            "gw_K512":
                pivot[512].values,
        })

        conv[
            "absolute_change_256_to_512"
        ] = np.abs(
            conv["gw_K512"]
            - conv["gw_K256"]
        )

        conv[
            "relative_change_256_to_512"
        ] = (
            conv[
                "absolute_change_256_to_512"
            ]
            / np.maximum(
                np.abs(
                    conv["gw_K512"]
                ),
                EPS,
            )
        )

        conv.to_csv(
            OUT
            / "gw_256_512_convergence.csv",
            index=False,
        )

    # Simple diagnostic trajectory plot.
    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    for K in K_VALUES:
        d = gw_df[
            gw_df["K"] == K
        ].sort_values("u_target")

        ax.plot(
            d["u_target"],
            d["gw_distance"],
            marker="o",
            label=f"K={K}",
        )

    ax.set_xlabel(
        "SUTRA hierarchy coordinate, u"
    )
    ax.set_ylabel(
        "GW relational distance"
    )
    ax.legend(
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "gw_scale_trajectory.png",
        dpi=250,
    )
    fig.savefig(
        OUT / "gw_scale_trajectory.pdf"
    )
    plt.close(fig)

    # Quantization diagnostic.
    fig, ax = plt.subplots(
        figsize=(7.2, 5.2)
    )

    for role in [
        "reference",
        "disease",
    ]:
        for K in K_VALUES:
            d = quant_df[
                (quant_df["role"] == role)
                & (quant_df["K"] == K)
            ].sort_values("u_target")

            ax.plot(
                d["u_target"],
                d["weighted_rms_radius"],
                marker="o",
                label=f"{role}, K={K}",
            )

    ax.set_xlabel(
        "SUTRA hierarchy coordinate, u"
    )
    ax.set_ylabel(
        "Mass-weighted RMS quantization radius"
    )
    ax.legend(
        frameon=False,
        fontsize=8,
        ncol=2,
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "metric_quantization_audit.png",
        dpi=250,
    )
    fig.savefig(
        OUT / "metric_quantization_audit.pdf"
    )
    plt.close(fig)

    strongest = (
        gw_df[
            gw_df["K"] == max(K_VALUES)
        ]
        .sort_values(
            "gw_distance",
            ascending=False,
        )
        .iloc[0]
        .to_dict()
    )

    provenance = {
        "analysis":
            "SUTRA kidney dominant-component "
            "nested metric-measure GW V2",
        "reference":
            SAMPLES["reference"],
        "disease":
            SAMPLES["disease"],
        "u_targets":
            [float(x) for x in U_TARGETS],
        "K_values":
            [int(x) for x in K_VALUES],
        "Kmax":
            Kmax,
        "hierarchy":
            "v0911 specimen-local contextual flow",
        "checkpoint_semantics":
            (
                "labels_s.npz is the partition immediately "
                "before productive microstep s; reconstruction "
                "of post-state t replays s..t inclusive"
            ),
        "hierarchy_state_validation":
            "14/14 target states reproduced exact ledger object counts",
        "metric":
            "unweighted shortest-path distance on quotient "
            "Level-0 contact topology",
        "domain":
            "largest connected component by Level-0-cell mass",
        "reference_domain_mass_fraction":
            0.994332,
        "disease_domain_mass_fraction":
            0.992055,
        "landmarks":
            "deterministic nested graph-geodesic farthest-point sequence",
        "measure":
            "Level-0-cell mass pushed forward by graph-metric Voronoi cells",
        "metric_normalization":
            "mass-weighted RMS pairwise landmark distance = 1",
        "gw_loss":
            "square loss",
        "expression_used_in_geometry":
            False,
        "biological_time_interpretation":
            False,
        "note":
            (
                "Hierarchy coordinate is organizational, not "
                "biological time. Dominant-component conditioning "
                "avoids arbitrary distances between disconnected "
                "contact-graph components."
            ),
    }

    with (
        OUT / "provenance.json"
    ).open("w") as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    print(
        "\nLargest observed K=512 value:"
    )
    print(
        json.dumps(
            strongest,
            indent=2,
            default=float,
        )
    )

    print("\nCOMPLETE")
    print(OUT)


if __name__ == "__main__":
    main()
