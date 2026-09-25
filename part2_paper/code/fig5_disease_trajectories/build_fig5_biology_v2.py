#!/usr/bin/env python3

from pathlib import Path
import json
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


ROOT = Path.home() / "Desktop" / "SUTRA"

LEDGER = (
    ROOT / "results" /
    "hierarchy_v0911_specimen_local_contextual_flow" /
    "ledger"
)

LEVEL0 = ROOT / "results" / "hierarchy_level0_v070"

V1 = (
    ROOT / "results" /
    "Fig5_Disease_Trajectories" /
    "structural_recruitment_v1"
)

OUT = (
    ROOT / "results" /
    "Fig5_Disease_Trajectories" /
    "biology_v2"
)

MEMBERS = OUT / "event_membership"
EXPR = OUT / "expression"
SPATIAL = OUT / "spatial"
FIGS = OUT / "figures"

for d in [OUT, MEMBERS, EXPR, SPATIAL, FIGS]:
    d.mkdir(parents=True, exist_ok=True)


SAMPLES = [
    "healthy_reference",
    "alzheimers",
    "nondiseased_kidney",
    "prcc",
]

DISPLAY = {
    "healthy_reference": "Healthy brain",
    "alzheimers": "Alzheimer's",
    "nondiseased_kidney": "Nondiseased kidney",
    "prcc": "PRCC",
}

# Keep enough events to learn the biology without exploding storage.
N_EVENTS = 30


# =============================================================================
# DATA
# =============================================================================

def cells(sample):
    return pq.read_table(
        LEVEL0 / sample / "cells.parquet"
    ).to_pandas()


def features(sample):
    return pq.read_table(
        LEVEL0 / sample / "features.parquet"
    ).to_pandas()


def expression_manifest(sample):
    with open(LEVEL0 / sample / "expression_manifest.json") as f:
        return json.load(f)


def selected_events(sample):
    p = (
        V1 / "event_selection" /
        f"{sample}_structural_events.parquet"
    )

    x = pd.read_parquet(p)

    return (
        x.sort_values(
            "structural_event_magnitude",
            ascending=False
        )
        .head(N_EVENTS)
        .reset_index(drop=True)
    )


# =============================================================================
# HIERARCHY REPLAY
# =============================================================================

def checkpoints(sample):
    d = LEDGER / sample / "label_checkpoints"

    ans = {}

    for p in d.glob("labels_*.npz"):
        s = int(p.stem.split("_")[1])
        ans[s] = p

    return ans


def load_checkpoint(path):
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["labels"], dtype=np.int64)


def nearest_checkpoint_before(sample, target_step):
    cps = checkpoints(sample)

    valid = [s for s in cps if s <= target_step]

    if not valid:
        raise RuntimeError(
            f"No checkpoint <= {target_step} for {sample}"
        )

    s = max(valid)

    return s, cps[s]


def replay_to_pre_event(sample, target_step):
    """
    Return Level-0 -> active-node labels immediately BEFORE target_step.

    labels_k represents the state entering microstep k.
    """

    cp_step, cp_path = nearest_checkpoint_before(
        sample, target_step
    )

    labels = load_checkpoint(cp_path)

    if cp_step == target_step:
        return labels

    merge_dir = LEDGER / sample / "merge_events"

    for step in range(cp_step, target_step):
        p = merge_dir / f"step_{step:06d}.parquet"

        if not p.exists():
            raise RuntimeError(
                f"Missing productive step during replay: {p}"
            )

        d = pq.read_table(
            p,
            columns=["survivor_node", "removed_node"]
        ).to_pandas()

        # Accepted mergers are node-disjoint.
        # All removed labels are replaced simultaneously by survivors.
        for survivor, removed in zip(
            d["survivor_node"].to_numpy(np.int64),
            d["removed_node"].to_numpy(np.int64),
        ):
            labels[labels == removed] = survivor

    return labels


# =============================================================================
# EXPRESSION
# =============================================================================

def decode_strings(x):
    out = []

    for v in x:
        if isinstance(v, bytes):
            out.append(v.decode("utf-8"))
        else:
            out.append(str(v))

    return np.asarray(out, dtype=object)


def load_expression(sample):
    """
    Materialize genes x cells dense matrix.

    541 genes x <= ~100k cells is modest enough here.
    """

    manifest = expression_manifest(sample)

    # Frozen Level-0 expression-pointer contract.
    expected_schema = "strata.level0.expression_pointer.v1"

    if manifest.get("schema_version") != expected_schema:
        raise RuntimeError(
            f"{sample}: unexpected expression manifest schema "
            f"{manifest.get('schema_version')!r}; "
            f"expected {expected_schema!r}"
        )

    source = Path(manifest["source_matrix"]).expanduser()

    if not source.exists():
        raise FileNotFoundError(
            f"{sample}: expression matrix does not exist: {source}"
        )

    expected_shape = tuple(
        int(x)
        for x in manifest["matrix_shape_features_by_cells"]
    )

    if len(expected_shape) != 2:
        raise RuntimeError(
            f"{sample}: invalid manifest matrix shape "
            f"{expected_shape}"
        )

    print("expression source:", source)
    print("manifest shape:", expected_shape)
    print("manifest nnz:", manifest.get("nnz"))

    with h5py.File(source, "r") as h:
        g = h["matrix"]

        data = np.asarray(g["data"])
        indices = np.asarray(g["indices"], dtype=np.int64)
        indptr = np.asarray(g["indptr"], dtype=np.int64)
        shape = tuple(np.asarray(g["shape"], dtype=int))

        fg = g["features"]

        if "name" in fg:
            gene_names = decode_strings(fg["name"][:])
        elif "gene_names" in fg:
            gene_names = decode_strings(fg["gene_names"][:])
        elif "id" in fg:
            gene_names = decode_strings(fg["id"][:])
        else:
            raise RuntimeError(
                f"Cannot find feature names in {source}"
            )

    ng, nc = shape

    if shape != expected_shape:
        raise RuntimeError(
            f"{sample}: H5 shape {shape} != "
            f"manifest shape {expected_shape}"
        )

    if len(data) != int(manifest["nnz"]):
        raise RuntimeError(
            f"{sample}: H5 nnz {len(data)} != "
            f"manifest nnz {manifest['nnz']}"
        )

    if len(gene_names) != ng:
        raise RuntimeError(
            f"{sample}: {len(gene_names)} feature names "
            f"for {ng} matrix rows"
        )

    X = np.zeros((ng, nc), dtype=np.float32)

    # 10x CSC: each indptr interval is one cell/column.
    for j in range(nc):
        a = indptr[j]
        b = indptr[j + 1]

        rows = indices[a:b]
        X[rows, j] = data[a:b]

    print(
        f"resolved expression: "
        f"{ng:,} genes x {nc:,} cells; "
        f"nnz={len(data):,}"
    )

    return X, gene_names


def lognorm(X):
    """
    Cell-wise library-size normalization followed by log1p.

    This is used only for descriptive event contrasts.
    Raw detection fractions are retained separately.
    """

    lib = X.sum(axis=0, keepdims=True)

    scale = np.divide(
        1e4,
        lib,
        out=np.zeros_like(lib, dtype=np.float32),
        where=lib > 0,
    )

    return np.log1p(X * scale)


# =============================================================================
# LOCAL SPATIAL CONTROL
# =============================================================================

def spatial_control_indices(
    cdf,
    incoming_idx,
    recipient_idx,
    multiplier=2.0,
):
    """
    Define a local annular/background population around the event.

    Radius is data-derived from the spatial extent of participating cells.
    No scipy dependency.
    """

    event_idx = np.concatenate(
        [incoming_idx, recipient_idx]
    )

    xy = cdf[["x", "y"]].to_numpy(float)
    event_xy = xy[event_idx]

    center = np.nanmean(event_xy, axis=0)

    dist_event = np.sqrt(
        np.sum((event_xy - center) ** 2, axis=1)
    )

    # Robust event spatial scale.
    r = np.nanquantile(dist_event, 0.90)

    if not np.isfinite(r) or r <= 0:
        r = np.nanmax(dist_event)

    if not np.isfinite(r) or r <= 0:
        return np.array([], dtype=np.int64), center, np.nan

    dist_all = np.sqrt(
        np.sum((xy - center) ** 2, axis=1)
    )

    participant = np.zeros(len(cdf), dtype=bool)
    participant[event_idx] = True

    # Local cells surrounding the event footprint.
    local = (
        (~participant)
        & (dist_all <= multiplier * r)
    )

    idx = np.flatnonzero(local)

    return idx, center, float(r)


# =============================================================================
# GENE SUMMARY
# =============================================================================

def group_stats(Xraw, Xlog, idx):
    ng = Xraw.shape[0]

    if len(idx) == 0:
        return {
            "mean": np.full(ng, np.nan),
            "detect": np.full(ng, np.nan),
        }

    return {
        "mean": Xlog[:, idx].mean(axis=1),
        "detect": (Xraw[:, idx] > 0).mean(axis=1),
    }


def event_gene_table(
    sample,
    event,
    Xraw,
    Xlog,
    gene_names,
    incoming_idx,
    recipient_idx,
    local_idx,
):
    I = group_stats(
        Xraw, Xlog, incoming_idx
    )

    R = group_stats(
        Xraw, Xlog, recipient_idx
    )

    L = group_stats(
        Xraw, Xlog, local_idx
    )

    out = pd.DataFrame({
        "sample": sample,
        "step": int(event["step"]),
        "gene": gene_names,

        "n_incoming": len(incoming_idx),
        "n_recipient": len(recipient_idx),
        "n_local": len(local_idx),

        "incoming_mean": I["mean"],
        "recipient_mean": R["mean"],
        "local_mean": L["mean"],

        "incoming_detect": I["detect"],
        "recipient_detect": R["detect"],
        "local_detect": L["detect"],
    })

    out["delta_incoming_recipient"] = (
        out["incoming_mean"]
        - out["recipient_mean"]
    )

    out["delta_incoming_local"] = (
        out["incoming_mean"]
        - out["local_mean"]
    )

    out["detect_delta_incoming_recipient"] = (
        out["incoming_detect"]
        - out["recipient_detect"]
    )

    out["detect_delta_incoming_local"] = (
        out["incoming_detect"]
        - out["local_detect"]
    )

    return out


# =============================================================================
# SPATIAL EVENT MAP
# =============================================================================

def plot_event(
    sample,
    event,
    cdf,
    incoming_idx,
    recipient_idx,
    local_idx,
):
    xy = cdf[["x", "y"]].to_numpy(float)

    fig, ax = plt.subplots(figsize=(6.5, 6.5))

    # Whole tissue context
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=1,
        alpha=0.08,
        rasterized=True,
        label="Other cells",
    )

    if len(local_idx):
        ax.scatter(
            xy[local_idx, 0],
            xy[local_idx, 1],
            s=3,
            alpha=0.25,
            rasterized=True,
            label="Local tissue",
        )

    ax.scatter(
        xy[recipient_idx, 0],
        xy[recipient_idx, 1],
        s=7,
        alpha=0.75,
        rasterized=True,
        label="Recipient collective",
    )

    ax.scatter(
        xy[incoming_idx, 0],
        xy[incoming_idx, 1],
        s=9,
        alpha=0.85,
        rasterized=True,
        label="Incoming collective",
    )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    ax.legend(
        frameon=False,
        markerscale=2,
        loc="best",
    )

    fig.tight_layout()

    stem = (
        f"{sample}_step_{int(event['step']):06d}"
    )

    fig.savefig(
        FIGS / f"{stem}_spatial.png",
        dpi=300,
        bbox_inches="tight",
    )

    fig.savefig(
        FIGS / f"{stem}_spatial.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# SAMPLE
# =============================================================================

def process_sample(sample):
    print("\n" + "=" * 90)
    print(sample)
    print("=" * 90)

    cdf = cells(sample).sort_values(
        "cell_index"
    ).reset_index(drop=True)

    assert np.array_equal(
        cdf["cell_index"].to_numpy(),
        np.arange(len(cdf)),
    )

    events = selected_events(sample)

    print("events:", len(events))
    print("loading expression...")

    Xraw, genes = load_expression(sample)

    assert Xraw.shape[1] == len(cdf)

    print("expression:", Xraw.shape)

    Xlog = lognorm(Xraw)

    membership_records = []
    gene_tables = []

    for rank, (_, event) in enumerate(
        events.iterrows(), start=1
    ):
        step = int(event["step"])

        labels = replay_to_pre_event(
            sample, step
        )

        recipient = int(event["recipient_node"])
        incoming = int(event["incoming_node"])

        recipient_idx = np.flatnonzero(
            labels == recipient
        )

        incoming_idx = np.flatnonzero(
            labels == incoming
        )

        # Exact agreement with event ledger.
        if len(recipient_idx) != int(
            event["recipient_mass"]
        ):
            raise RuntimeError(
                f"{sample} step {step}: recipient mass "
                f"{len(recipient_idx)} != "
                f"{int(event['recipient_mass'])}"
            )

        if len(incoming_idx) != int(
            event["incoming_mass"]
        ):
            raise RuntimeError(
                f"{sample} step {step}: incoming mass "
                f"{len(incoming_idx)} != "
                f"{int(event['incoming_mass'])}"
            )

        local_idx, center, radius = (
            spatial_control_indices(
                cdf,
                incoming_idx,
                recipient_idx,
            )
        )

        print(
            f"{rank:02d}/{len(events)} "
            f"step={step:4d} "
            f"I={len(incoming_idx):5d} "
            f"R={len(recipient_idx):5d} "
            f"L={len(local_idx):5d}"
        )

        # Save exact memberships compactly.
        np.savez_compressed(
            MEMBERS /
            f"{sample}_step_{step:06d}.npz",
            incoming=incoming_idx,
            recipient=recipient_idx,
            local=local_idx,
        )

        membership_records.append({
            "sample": sample,
            "rank": rank,
            "step": step,
            "u_before": float(event["u_before"]),
            "incoming_node": incoming,
            "recipient_node": recipient,
            "incoming_mass": len(incoming_idx),
            "recipient_mass": len(recipient_idx),
            "post_mass": (
                len(incoming_idx)
                + len(recipient_idx)
            ),
            "local_n": len(local_idx),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "event_radius_q90": radius,
            "structural_event_magnitude": float(
                event["structural_event_magnitude"]
            ),
            "composite_merge_cost": float(
                event["composite_merge_cost"]
            ),
            "molecular_distance": float(
                event["molecular_distance"]
            ),
            "n_boundary_edges": int(
                event["n_boundary_edges"]
            ),
        })

        gt = event_gene_table(
            sample,
            event,
            Xraw,
            Xlog,
            genes,
            incoming_idx,
            recipient_idx,
            local_idx,
        )

        gt["event_rank"] = rank
        gt["u_before"] = float(
            event["u_before"]
        )
        gt["structural_event_magnitude"] = float(
            event["structural_event_magnitude"]
        )

        gene_tables.append(gt)

        # Spatial maps for strongest 6 events only.
        if rank <= 6:
            plot_event(
                sample,
                event,
                cdf,
                incoming_idx,
                recipient_idx,
                local_idx,
            )

    membership = pd.DataFrame(
        membership_records
    )

    membership.to_csv(
        SPATIAL /
        f"{sample}_event_membership_summary.csv",
        index=False,
    )

    genes_all = pd.concat(
        gene_tables,
        ignore_index=True,
    )

    genes_all.to_parquet(
        EXPR /
        f"{sample}_event_gene_effects.parquet",
        index=False,
    )

    # Aggregate evidence across nominated events.
    agg = (
        genes_all
        .groupby("gene", as_index=False)
        .agg(
            n_events=("step", "nunique"),

            median_delta_IR=(
                "delta_incoming_recipient",
                "median",
            ),
            median_delta_IL=(
                "delta_incoming_local",
                "median",
            ),

            mean_delta_IR=(
                "delta_incoming_recipient",
                "mean",
            ),
            mean_delta_IL=(
                "delta_incoming_local",
                "mean",
            ),

            median_detect_delta_IR=(
                "detect_delta_incoming_recipient",
                "median",
            ),
            median_detect_delta_IL=(
                "detect_delta_incoming_local",
                "median",
            ),
        )
    )

    # Convergence rather than opaque composite score.
    agg["same_direction"] = (
        np.sign(agg["median_delta_IR"])
        ==
        np.sign(agg["median_delta_IL"])
    )

    agg["min_abs_effect"] = np.minimum(
        np.abs(agg["median_delta_IR"]),
        np.abs(agg["median_delta_IL"]),
    )

    agg = agg.sort_values(
        [
            "same_direction",
            "min_abs_effect",
        ],
        ascending=[False, False],
    )

    agg.to_csv(
        EXPR /
        f"{sample}_gene_recruitment_summary.csv",
        index=False,
    )

    print("\nTop convergent genes:")
    print(
        agg.head(20)[[
            "gene",
            "median_delta_IR",
            "median_delta_IL",
            "median_detect_delta_IR",
            "median_detect_delta_IL",
            "same_direction",
        ]].to_string(index=False)
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print(
        "SUTRA FIGURE 5 — EXACT STRUCTURAL "
        "RECRUITMENT BIOLOGY V2"
    )
    print("=" * 90)

    for sample in SAMPLES:
        process_sample(sample)

    provenance = {
        "version": "biology_v2",
        "hierarchy": str(LEDGER),
        "structural_selection": str(V1),
        "samples": SAMPLES,
        "n_events_per_sample": N_EVENTS,
        "expression_selection_blind": True,
        "interpretation": (
            "Descriptive molecular associations with "
            "SUTRA structural recruitment events; hierarchy "
            "coordinate is not biological time."
        ),
    }

    with open(
        OUT / "provenance.json", "w"
    ) as f:
        json.dump(
            provenance,
            f,
            indent=2,
        )

    print("\n" + "=" * 90)
    print("COMPLETE")
    print("=" * 90)
    print(OUT)


if __name__ == "__main__":
    main()
