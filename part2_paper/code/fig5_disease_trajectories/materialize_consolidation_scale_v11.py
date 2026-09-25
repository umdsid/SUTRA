#!/usr/bin/env python3

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path.home() / "Desktop" / "SUTRA"

LEVEL0 = ROOT / "results" / "hierarchy_level0_v070"
HIER = ROOT / "results" / "hierarchy_v0911_specimen_local_contextual_flow"

OUT = (
    ROOT / "results" / "Fig5_Disease_Trajectories" /
    "consolidation_scale_v11"
)
OUT.mkdir(parents=True, exist_ok=True)

# PRCC is Fig5. Healthy brain + kidney immediately prepare Fig6.
SAMPLES = [
    "prcc",
    "healthy_reference",
    "nondiseased_kidney",
]

EPS = 1e-12


def load_cells(sample):
    p = LEVEL0 / sample / "cells.parquet"
    if not p.exists():
        raise FileNotFoundError(p)

    df = pd.read_parquet(p)

    # Preserve only what downstream needs.
    cols = ["cell_id", "x", "y"]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"{sample}: missing cells column {c}")

    return df[cols].copy()


def load_graph(sample, n):
    """
    Load SUTRA's custom structural CSR container.

    graph_csr.npz is NOT scipy.sparse.save_npz format. It stores:
      indptr
      indices
      interface_ids
      node_ids

    For consolidation-scale analysis we need structural adjacency
    only; interface_ids are therefore not used as edge weights.
    """
    p = LEVEL0 / sample / "graph_csr.npz"
    if not p.exists():
        raise FileNotFoundError(p)

    z = np.load(p, allow_pickle=False)

    required = {
        "indptr",
        "indices",
        "interface_ids",
        "node_ids",
    }

    missing = required.difference(z.files)

    if missing:
        raise RuntimeError(
            f"{sample}: graph container missing {sorted(missing)}; "
            f"found {z.files}"
        )

    indptr = np.asarray(z["indptr"], dtype=np.int64)
    indices = np.asarray(z["indices"], dtype=np.int64)
    node_ids = np.asarray(z["node_ids"], dtype=np.int64)

    if len(indptr) != n + 1:
        raise RuntimeError(
            f"{sample}: len(indptr)={len(indptr)} != n+1={n+1}"
        )

    if len(node_ids) != n:
        raise RuntimeError(
            f"{sample}: len(node_ids)={len(node_ids)} != n={n}"
        )

    # The current Level-0 materialization uses contiguous node IDs.
    # Do not silently assume this if a future materialization changes.
    expected = np.arange(n, dtype=np.int64)

    if not np.array_equal(node_ids, expected):
        raise RuntimeError(
            f"{sample}: node_ids are not contiguous 0..N-1; "
            "explicit node-ID remapping is required."
        )

    if indptr[0] != 0:
        raise RuntimeError(
            f"{sample}: CSR indptr must begin at zero"
        )

    if indptr[-1] != len(indices):
        raise RuntimeError(
            f"{sample}: indptr[-1]={indptr[-1]} "
            f"!= len(indices)={len(indices)}"
        )

    if len(indices):
        if indices.min() < 0 or indices.max() >= n:
            raise RuntimeError(
                f"{sample}: CSR indices outside [0,{n})"
            )

    data = np.ones(len(indices), dtype=np.uint8)

    G = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n, n),
    )

    # Structural graph only.
    G.setdiag(0)
    G.eliminate_zeros()

    # Defensively enforce undirected adjacency. For the production
    # graphs this should not change the edge set.
    nnz_before = int(G.nnz)

    G = G.maximum(G.T).tocsr()

    nnz_after = int(G.nnz)

    print(
        "custom CSR:",
        f"directed entries={nnz_before:,}",
        f"symmetrized entries={nnz_after:,}",
        f"undirected edges={nnz_after // 2:,}",
    )

    return G


def checkpoint_step(path):
    m = re.search(r"labels_(\d+)\.npz$", path.name)
    if not m:
        raise RuntimeError(f"Cannot parse checkpoint {path}")
    return int(m.group(1))


def load_checkpoints(sample, n):
    d = HIER / "ledger" / sample / "label_checkpoints"

    if not d.exists():
        raise FileNotFoundError(d)

    files = sorted(
        d.glob("labels_*.npz"),
        key=checkpoint_step,
    )

    if not files:
        raise RuntimeError(f"{sample}: no label checkpoints")

    out = []

    for p in files:
        z = np.load(p, allow_pickle=False)

        if "labels" not in z.files:
            raise RuntimeError(f"{p}: missing labels")

        labels = np.asarray(z["labels"])

        if len(labels) != n:
            raise RuntimeError(
                f"{sample}: checkpoint {p.name} "
                f"length {len(labels)} != {n}"
            )

        out.append((checkpoint_step(p), labels))

    return out


def load_steps(sample):
    p = HIER / "ledger" / sample / "steps.jsonl"
    if not p.exists():
        raise FileNotFoundError(p)

    rows = []

    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise RuntimeError(f"{sample}: empty steps ledger")

    # Discover coordinate field conservatively.
    candidates = [
        "hierarchy_coordinate_removed_fraction",
        "removed_fraction_after",
        "u",
        "u_after",
        "hierarchy_coordinate",
        "removed_fraction",
    ]

    ufield = None
    for c in candidates:
        if c in rows[0]:
            ufield = c
            break

    if ufield is None:
        print("First step keys:", sorted(rows[0]))
        raise RuntimeError(
            f"{sample}: could not identify hierarchy coordinate field"
        )

    step_to_u = {}

    for i, r in enumerate(rows):
        # Prefer explicit microstep if present.
        s = int(r.get("microstep", i + 1))
        step_to_u[s] = float(r[ufield])

    return rows, step_to_u, ufield


def checkpoint_u(step, step_to_u):
    """
    Coordinate associated with a checkpoint.

    Checkpoints encode a hierarchy state. We use the nearest
    available ledger coordinate <= checkpoint step where possible.
    This is for coarse bracketing only; exact consolidation is
    subsequently refined from merger ancestry.
    """
    keys = np.array(sorted(step_to_u), dtype=int)

    q = keys[keys <= step]

    if len(q):
        return float(step_to_u[int(q[-1])])

    return 0.0


def upper_triangle_edges(G):
    T = sparse.triu(G, k=1).tocoo()

    return (
        T.row.astype(np.int64),
        T.col.astype(np.int64),
    )


def bracket_edge_consolidation(
    ii,
    jj,
    checkpoints,
    step_to_u,
):
    """
    Find first stored checkpoint where each Level-0 adjacency pair
    has the same hierarchy label.

    This produces a checkpoint-bracketed consolidation scale.
    It is already scientifically valid as an interval/upper bound,
    but we export bracketing information explicitly rather than
    pretending checkpoint resolution is exact.
    """
    m = len(ii)

    first_step = np.full(m, -1, dtype=np.int64)
    previous_step = np.full(m, -1, dtype=np.int64)

    unresolved = np.ones(m, dtype=bool)
    prev_cp_step = 0

    for cp_step, labels in checkpoints:
        idx = np.flatnonzero(unresolved)

        if len(idx) == 0:
            break

        same = (
            labels[ii[idx]] ==
            labels[jj[idx]]
        )

        hit = idx[same]

        if len(hit):
            first_step[hit] = cp_step
            previous_step[hit] = prev_cp_step
            unresolved[hit] = False

        prev_cp_step = cp_step

        print(
            f"  checkpoint {cp_step}: "
            f"new={len(hit):,} "
            f"unresolved={int(unresolved.sum()):,}"
        )

    lower_u = np.full(m, np.nan, dtype=float)
    upper_u = np.full(m, np.nan, dtype=float)

    resolved = first_step >= 0

    for s in np.unique(previous_step[resolved]):
        mask = resolved & (previous_step == s)
        lower_u[mask] = (
            0.0 if s <= 0
            else checkpoint_u(int(s), step_to_u)
        )

    for s in np.unique(first_step[resolved]):
        mask = resolved & (first_step == s)
        upper_u[mask] = checkpoint_u(
            int(s), step_to_u
        )

    return (
        first_step,
        previous_step,
        lower_u,
        upper_u,
        unresolved,
    )


def cell_summary(
    n,
    ii,
    jj,
    lower_u,
    upper_u,
):
    """
    Summarize consolidation of Level-0 boundaries incident on each
    cell. These are graph-local outcomes and will later support both
    visualization and predictive tests.
    """
    finite = np.isfinite(upper_u)

    count = np.zeros(n, dtype=np.int64)
    sum_u = np.zeros(n, dtype=float)
    max_u = np.full(n, np.nan, dtype=float)
    min_u = np.full(n, np.nan, dtype=float)

    for a, b, u in zip(
        ii[finite],
        jj[finite],
        upper_u[finite],
    ):
        for c in (a, b):
            count[c] += 1
            sum_u[c] += u

            if np.isnan(max_u[c]) or u > max_u[c]:
                max_u[c] = u

            if np.isnan(min_u[c]) or u < min_u[c]:
                min_u[c] = u

    mean_u = np.full(n, np.nan, dtype=float)
    good = count > 0
    mean_u[good] = sum_u[good] / count[good]

    return pd.DataFrame({
        "cell_index": np.arange(n, dtype=np.int64),
        "n_resolved_incident_boundaries": count,
        "mean_boundary_consolidation_u": mean_u,
        "min_boundary_consolidation_u": min_u,
        "max_boundary_consolidation_u": max_u,
    })


def process(sample):
    print("\n" + "=" * 90)
    print(sample)
    print("=" * 90)

    cells = load_cells(sample)
    n = len(cells)

    print("N0:", n)

    G = load_graph(sample, n)

    ii, jj = upper_triangle_edges(G)

    print("Level-0 undirected adjacencies:", len(ii))

    checkpoints = load_checkpoints(sample, n)
    print("checkpoints:", len(checkpoints))

    steps, step_to_u, ufield = load_steps(sample)
    print("hierarchy coordinate field:", ufield)

    (
        first_step,
        previous_step,
        lower_u,
        upper_u,
        unresolved,
    ) = bracket_edge_consolidation(
        ii,
        jj,
        checkpoints,
        step_to_u,
    )

    edge = pd.DataFrame({
        "cell_i": ii,
        "cell_j": jj,
        "previous_checkpoint_step": previous_step,
        "first_same_checkpoint_step": first_step,
        "consolidation_u_lower": lower_u,
        "consolidation_u_upper": upper_u,
        "resolved": ~unresolved,
    })

    # Physical edge midpoint and length for Fig5 rendering.
    x = cells["x"].to_numpy(float)
    y = cells["y"].to_numpy(float)

    edge["mid_x"] = 0.5 * (x[ii] + x[jj])
    edge["mid_y"] = 0.5 * (y[ii] + y[jj])

    edge["edge_length"] = np.hypot(
        x[ii] - x[jj],
        y[ii] - y[jj],
    )

    sample_out = OUT / sample
    sample_out.mkdir(parents=True, exist_ok=True)

    edge.to_parquet(
        sample_out / "level0_edge_consolidation.parquet",
        index=False,
    )

    cs = cell_summary(
        n,
        ii,
        jj,
        lower_u,
        upper_u,
    )

    cs = pd.concat(
        [
            cells.reset_index(drop=True),
            cs.drop(columns=["cell_index"]),
        ],
        axis=1,
    )

    cs.insert(
        0,
        "cell_index",
        np.arange(n, dtype=np.int64),
    )

    cs.to_parquet(
        sample_out / "cell_consolidation_summary.parquet",
        index=False,
    )

    resolved = ~unresolved

    summary = {
        "sample": sample,
        "n_level0_cells": int(n),
        "n_level0_undirected_edges": int(len(ii)),
        "n_checkpoints": int(len(checkpoints)),
        "hierarchy_coordinate_field": ufield,
        "resolved_edges": int(resolved.sum()),
        "unresolved_edges": int(unresolved.sum()),
        "resolved_fraction": float(resolved.mean()),
        "median_upper_u": (
            float(np.nanmedian(upper_u))
            if np.any(resolved) else None
        ),
        "median_bracket_width": (
            float(
                np.nanmedian(
                    upper_u[resolved] -
                    lower_u[resolved]
                )
            )
            if np.any(resolved) else None
        ),
        "interpretation": (
            "consolidation_u_lower/upper bracket the first stored "
            "hierarchy state in which the two endpoints of a "
            "Level-0 adjacency belong to the same SUTRA object. "
            "This is organizational scale, not biological time."
        ),
    }

    (
        sample_out / "summary.json"
    ).write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))

    return summary


def main():
    summaries = []

    for sample in SAMPLES:
        summaries.append(process(sample))

    pd.DataFrame(summaries).to_csv(
        OUT / "all_sample_summary.csv",
        index=False,
    )

    provenance = {
        "version": "consolidation_scale_v11",
        "source_hierarchy": str(HIER),
        "source_level0": str(LEVEL0),
        "samples": SAMPLES,
        "hierarchy_recomputed": False,
        "gw_recomputed": False,
        "expression_recomputed": False,
        "primary_object": (
            "Level-0 adjacency consolidation scale: first hierarchy "
            "state at which adjacent Level-0 cells share a SUTRA "
            "object."
        ),
        "current_resolution": (
            "checkpoint-bracketed. Exact event-level refinement can "
            "be performed only if required after inspecting bracket "
            "widths."
        ),
        "not_biological_time": True,
    }

    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2)
    )

    print("\nWROTE:", OUT)


if __name__ == "__main__":
    main()
