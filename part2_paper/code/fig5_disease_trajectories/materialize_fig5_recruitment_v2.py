#!/usr/bin/env python3

from pathlib import Path
import json
import math
import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path.home() / "Desktop" / "SUTRA"
RES = ROOT / "results"

OUT = (
    RES /
    "Fig5_Disease_Trajectories" /
    "recruitment_v2"
)
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    "healthy_reference",
    "alzheimers",
    "gbm_reference_addon",
    "nondiseased_kidney",
    "prcc",
]

META = {
    "healthy_reference": ("brain", "reference"),
    "alzheimers": ("brain", "disease"),
    "gbm_reference_addon": ("brain", "gbm_related"),
    "nondiseased_kidney": ("kidney", "reference"),
    "prcc": ("kidney", "disease"),
}

# Full expression profiling is done only for the strongest
# structurally informative events, but EVERY merger goes into
# the event census.
TOP_EVENTS_PER_SAMPLE = 40

# Avoid interpreting tiny two-cell mergers as biological events.
MIN_SMALL_OBJECT = 20
MIN_POST_MASS = 100

EPS = 0.25


def decode(x):
    if isinstance(x, bytes):
        return x.decode()
    return str(x)


def log2_effect(a, b):
    return np.log2(
        (a + EPS) /
        (b + EPS)
    )


def bbox_distance_mask(cells, idx, radius_factor=1.5):
    """
    Spatial local background around the union of the two pre-event
    objects. Uses an expanded bounding box. This is intentionally
    transparent and not a cell-type matching procedure.
    """
    xy = cells.loc[idx, ["x", "y"]].to_numpy(float)

    xmin, ymin = np.nanmin(xy, axis=0)
    xmax, ymax = np.nanmax(xy, axis=0)

    dx = max(xmax - xmin, 1.0)
    dy = max(ymax - ymin, 1.0)

    xmin -= radius_factor * dx
    xmax += radius_factor * dx
    ymin -= radius_factor * dy
    ymax += radius_factor * dy

    return (
        (cells["x"].to_numpy() >= xmin) &
        (cells["x"].to_numpy() <= xmax) &
        (cells["y"].to_numpy() >= ymin) &
        (cells["y"].to_numpy() <= ymax)
    )


def read_expression(sample, cells, features):
    manifest_path = (
        RES /
        "hierarchy_level0_v070" /
        sample /
        "expression_manifest.json"
    )

    with open(manifest_path) as f:
        manifest = json.load(f)

    h5_path = Path(manifest["source_matrix"])

    with h5py.File(h5_path, "r") as h:
        m = h["matrix"]

        shape = tuple(
            int(x)
            for x in np.asarray(m["shape"])
        )

        if shape != (
            len(features),
            len(cells),
        ):
            raise RuntimeError(
                f"{sample}: matrix shape {shape} != "
                f"({len(features)}, {len(cells)})"
            )

        h5_barcodes = np.array(
            [decode(x) for x in m["barcodes"][:]],
            dtype=object,
        )

        cell_ids = cells["cell_id"].astype(str).to_numpy()

        barcode_match = (
            len(h5_barcodes) == len(cell_ids) and
            np.array_equal(h5_barcodes, cell_ids)
        )

        print(
            f"{sample}: exact barcode equality = "
            f"{barcode_match}"
        )

        if not barcode_match:
            # Helpful diagnostic but DO NOT silently reorder.
            n = min(len(h5_barcodes), len(cell_ids))
            frac = np.mean(
                h5_barcodes[:n] ==
                cell_ids[:n]
            )
            raise RuntimeError(
                f"{sample}: barcode ordering failed; "
                f"position-wise agreement={frac:.6f}"
            )

        h5_names = np.array(
            [decode(x) for x in m["features"]["name"][:]],
            dtype=object,
        )

        expected_names = (
            features
            .sort_values("feature_index")
            ["feature_name"]
            .astype(str)
            .to_numpy()
        )

        if not np.array_equal(
            h5_names,
            expected_names,
        ):
            raise RuntimeError(
                f"{sample}: feature ordering mismatch"
            )

        # 10x CSC matrix: features x cells.
        data = np.asarray(m["data"])
        indices = np.asarray(m["indices"])
        indptr = np.asarray(m["indptr"])

    return (
        data,
        indices,
        indptr,
        h5_names,
        manifest,
    )


def group_stats(
    cell_indices,
    n_features,
    data,
    indices,
    indptr,
):
    """
    Mean expression and detection fraction for a group of cells,
    directly from 10x CSC without scipy.
    """
    cell_indices = np.asarray(
        cell_indices,
        dtype=np.int64,
    )

    sums = np.zeros(
        n_features,
        dtype=np.float64,
    )

    detect = np.zeros(
        n_features,
        dtype=np.int64,
    )

    for c in cell_indices:
        lo = indptr[c]
        hi = indptr[c + 1]

        feats = indices[lo:hi]
        vals = data[lo:hi]

        np.add.at(
            sums,
            feats,
            vals,
        )

        # Sparse 10x entries are nonzero.
        np.add.at(
            detect,
            feats,
            1,
        )

    n = len(cell_indices)

    if n == 0:
        return (
            np.full(n_features, np.nan),
            np.full(n_features, np.nan),
        )

    return (
        sums / n,
        detect / n,
    )


def replay_events(sample, cells):
    p = (
        RES /
        "hierarchy_v073_full" /
        sample /
        "merge_trajectory.parquet"
    )

    merges = pd.read_parquet(p).copy()

    N0 = len(cells)

    # Each active node maps to exact Level-0 membership.
    members = {
        i: [i]
        for i in range(N0)
    }

    rows = []
    event_members = {}

    cumulative = 0

    for event_id, r in merges.iterrows():

        survivor = int(r["survivor_node"])
        removed = int(r["removed_node"])

        if survivor not in members:
            raise RuntimeError(
                f"{sample} event {event_id}: "
                f"survivor {survivor} inactive"
            )

        if removed not in members:
            raise RuntimeError(
                f"{sample} event {event_id}: "
                f"removed {removed} inactive"
            )

        A = members[survivor]
        B = members[removed]

        # "recipient" = larger pre-existing object.
        # "incoming"  = smaller pre-existing object.
        # This is a descriptive orientation, not biological causality.
        if len(A) >= len(B):
            recipient = A
            incoming = B
            recipient_node = survivor
            incoming_node = removed
        else:
            recipient = B
            incoming = A
            recipient_node = removed
            incoming_node = survivor

        post = A + B

        cumulative += 1
        u = cumulative / N0

        m_rec = len(recipient)
        m_inc = len(incoming)
        m_post = len(post)

        # Structural event score only.
        # Favors substantial incoming objects and large resulting
        # collectives without using gene expression.
        structural_score = (
            math.sqrt(m_inc) *
            math.log1p(m_post) *
            (m_inc / m_post)
        )

        row = {
            "sample": sample,
            "event_id": int(event_id),
            "level": int(r["level"]),
            "u": u,
            "recipient_node": recipient_node,
            "incoming_node": incoming_node,
            "survivor_node": survivor,
            "removed_node": removed,
            "recipient_mass": m_rec,
            "incoming_mass": m_inc,
            "post_mass": m_post,
            "incoming_fraction": m_inc / m_post,
            "structural_score": structural_score,
            "n_boundary_edges": int(r["n_boundary_edges"]),
            "molecular_z": float(r["molecular_z"]),
            "mechanics_support_fraction":
                float(r["mechanics_support_fraction"]),
            "abs_tension_z": float(r["abs_tension_z"]),
            "abs_delta_p_z": float(r["abs_delta_p_z"]),
            "comm_strength": float(r["comm_strength"]),
            "comm_reciprocity": float(r["comm_reciprocity"]),
            "ordering_merit": float(r["ordering_merit"]),
        }

        rows.append(row)

        event_members[int(event_id)] = {
            "recipient": np.asarray(
                recipient,
                dtype=np.int64,
            ),
            "incoming": np.asarray(
                incoming,
                dtype=np.int64,
            ),
            "post": np.asarray(
                post,
                dtype=np.int64,
            ),
        }

        # Actual hierarchy update follows survivor/removed semantics.
        members[survivor] = post
        del members[removed]

    return (
        pd.DataFrame(rows),
        event_members,
    )


all_event_census = []
all_gene_tables = []
all_selected_events = []
alignment_report = []

for sample in SAMPLES:
    print()
    print("=" * 88)
    print(sample)
    print("=" * 88)

    l0 = (
        RES /
        "hierarchy_level0_v070" /
        sample
    )

    cells = (
        pq.read_table(
            l0 / "cells.parquet"
        )
        .to_pandas()
        .sort_values("cell_index")
        .reset_index(drop=True)
    )

    features = (
        pq.read_table(
            l0 / "features.parquet"
        )
        .to_pandas()
        .sort_values("feature_index")
        .reset_index(drop=True)
    )

    # Hard invariants.
    expected_idx = np.arange(len(cells))

    if not np.array_equal(
        cells["cell_index"].to_numpy(),
        expected_idx,
    ):
        raise RuntimeError(
            f"{sample}: cell_index is not 0..N-1"
        )

    if not np.array_equal(
        cells["matrix_column"].to_numpy(),
        expected_idx,
    ):
        raise RuntimeError(
            f"{sample}: matrix_column != Level-0 cell_index"
        )

    data, indices, indptr, genes, manifest = (
        read_expression(
            sample,
            cells,
            features,
        )
    )

    alignment_report.append({
        "sample": sample,
        "n_cells": len(cells),
        "n_features": len(features),
        "nnz": int(len(data)),
        "barcode_exact": True,
        "feature_exact": True,
        "source_matrix":
            manifest["source_matrix"],
    })

    census, memberships = replay_events(
        sample,
        cells,
    )

    organ, state = META[sample]
    census["organ"] = organ
    census["state"] = state

    all_event_census.append(census)

    eligible = census[
        (census["incoming_mass"] >= MIN_SMALL_OBJECT) &
        (census["post_mass"] >= MIN_POST_MASS)
    ].copy()

    eligible = eligible.sort_values(
        [
            "structural_score",
            "post_mass",
            "incoming_mass",
        ],
        ascending=False,
    )

    selected = eligible.head(
        TOP_EVENTS_PER_SAMPLE
    ).copy()

    print(
        f"mergers={len(census):,}; "
        f"eligible={len(eligible):,}; "
        f"profiled={len(selected):,}"
    )

    all_selected_events.append(selected)

    n_features = len(genes)

    for rank, (_, ev) in enumerate(
        selected.iterrows(),
        start=1,
    ):
        event_id = int(ev["event_id"])
        mem = memberships[event_id]

        rec = mem["recipient"]
        inc = mem["incoming"]
        post = mem["post"]

        event_mask = np.zeros(
            len(cells),
            dtype=bool,
        )
        event_mask[post] = True

        local_mask = bbox_distance_mask(
            cells,
            post,
            radius_factor=1.5,
        )

        local_mask &= ~event_mask

        local = np.flatnonzero(
            local_mask
        )

        # If spatial neighborhood is too small, retain it as NaN
        # rather than silently switching definitions.
        local_valid = len(local) >= 20

        rec_mean, rec_det = group_stats(
            rec,
            n_features,
            data,
            indices,
            indptr,
        )

        inc_mean, inc_det = group_stats(
            inc,
            n_features,
            data,
            indices,
            indptr,
        )

        post_mean, post_det = group_stats(
            post,
            n_features,
            data,
            indices,
            indptr,
        )

        if local_valid:
            loc_mean, loc_det = group_stats(
                local,
                n_features,
                data,
                indices,
                indptr,
            )
        else:
            loc_mean = np.full(
                n_features,
                np.nan,
            )
            loc_det = np.full(
                n_features,
                np.nan,
            )

        # Whole-specimen background, excluding event cells.
        bg = np.flatnonzero(
            ~event_mask
        )

        bg_mean, bg_det = group_stats(
            bg,
            n_features,
            data,
            indices,
            indptr,
        )

        tab = pd.DataFrame({
            "sample": sample,
            "organ": organ,
            "state": state,
            "event_rank": rank,
            "event_id": event_id,
            "level": int(ev["level"]),
            "u": float(ev["u"]),
            "recipient_mass":
                int(ev["recipient_mass"]),
            "incoming_mass":
                int(ev["incoming_mass"]),
            "post_mass":
                int(ev["post_mass"]),
            "incoming_fraction":
                float(ev["incoming_fraction"]),
            "structural_score":
                float(ev["structural_score"]),
            "gene": genes,
            "recipient_mean": rec_mean,
            "incoming_mean": inc_mean,
            "post_mean": post_mean,
            "local_mean": loc_mean,
            "background_mean": bg_mean,
            "recipient_detection": rec_det,
            "incoming_detection": inc_det,
            "post_detection": post_det,
            "local_detection": loc_det,
            "background_detection": bg_det,
        })

        tab[
            "log2_incoming_vs_recipient"
        ] = log2_effect(
            inc_mean,
            rec_mean,
        )

        tab[
            "log2_incoming_vs_local"
        ] = log2_effect(
            inc_mean,
            loc_mean,
        )

        tab[
            "log2_incoming_vs_background"
        ] = log2_effect(
            inc_mean,
            bg_mean,
        )

        tab[
            "detection_delta_incoming_recipient"
        ] = (
            inc_det -
            rec_det
        )

        tab[
            "detection_delta_incoming_local"
        ] = (
            inc_det -
            loc_det
        )

        all_gene_tables.append(tab)

        # Save exact membership only for the strongest 10 events
        # per specimen to keep package compact.
        if rank <= 10:
            spatial = cells.loc[
                post,
                [
                    "cell_index",
                    "cell_id",
                    "x",
                    "y",
                    "area",
                    "patch_id",
                ]
            ].copy()

            spatial["role"] = "recipient"
            spatial.loc[
                spatial["cell_index"].isin(inc),
                "role"
            ] = "incoming"

            spatial["sample"] = sample
            spatial["event_rank"] = rank
            spatial["event_id"] = event_id

            spatial.to_csv(
                OUT /
                f"{sample}_event{rank:02d}_spatial_membership.csv",
                index=False,
            )


EVENTS = pd.concat(
    all_event_census,
    ignore_index=True,
)

SELECTED = pd.concat(
    all_selected_events,
    ignore_index=True,
)

GENES = pd.concat(
    all_gene_tables,
    ignore_index=True,
)

ALIGN = pd.DataFrame(
    alignment_report
)

EVENTS.to_csv(
    OUT / "all_merger_event_census.csv",
    index=False,
)

SELECTED.to_csv(
    OUT / "profiled_structural_events.csv",
    index=False,
)

GENES.to_csv(
    OUT / "all_profiled_event_gene_statistics.csv",
    index=False,
)

ALIGN.to_csv(
    OUT / "expression_alignment_audit.csv",
    index=False,
)


# ------------------------------------------------------------
# Candidate summary.
#
# IMPORTANT: descriptive ranking, not causal driver inference.
# Requires:
#   - reasonable detection in incoming object
#   - consistency of incoming-vs-recipient and incoming-vs-local
#   - recurrence across profiled events
# ------------------------------------------------------------

usable = GENES[
    (GENES["incoming_detection"] >= 0.10)
].copy()

usable["same_direction_local"] = (
    np.sign(
        usable["log2_incoming_vs_recipient"]
    ) ==
    np.sign(
        usable["log2_incoming_vs_local"]
    )
)

usable["event_molecular_strength"] = (
    np.minimum(
        np.abs(
            usable["log2_incoming_vs_recipient"]
        ),
        np.abs(
            usable["log2_incoming_vs_local"]
        ),
    )
)

usable.loc[
    ~usable["same_direction_local"],
    "event_molecular_strength"
] = 0.0


candidate_rows = []

for (sample, gene), z in usable.groupby(
    ["sample", "gene"],
    sort=False,
):
    z = z.sort_values("u")

    finite = np.isfinite(
        z["event_molecular_strength"]
    )

    zz = z.loc[finite]

    if len(zz) == 0:
        continue

    # Structural weighting rewards expression effects observed
    # at more substantial recruitment events.
    w = np.sqrt(
        zz["incoming_mass"].to_numpy(float)
    )

    s = (
        zz["event_molecular_strength"]
        .to_numpy(float)
    )

    weighted = float(
        np.sum(w * s) /
        np.sum(w)
    )

    positive_fraction = float(
        np.mean(
            zz["log2_incoming_vs_recipient"] > 0
        )
    )

    negative_fraction = float(
        np.mean(
            zz["log2_incoming_vs_recipient"] < 0
        )
    )

    directional_consistency = max(
        positive_fraction,
        negative_fraction,
    )

    candidate_rows.append({
        "sample": sample,
        "gene": gene,
        "n_profiled_events":
            len(zz),
        "weighted_structural_association":
            weighted,
        "directional_consistency":
            directional_consistency,
        "median_abs_incoming_vs_recipient":
            float(
                np.nanmedian(
                    np.abs(
                        zz[
                            "log2_incoming_vs_recipient"
                        ]
                    )
                )
            ),
        "median_abs_incoming_vs_local":
            float(
                np.nanmedian(
                    np.abs(
                        zz[
                            "log2_incoming_vs_local"
                        ]
                    )
                )
            ),
        "median_incoming_detection":
            float(
                np.nanmedian(
                    zz["incoming_detection"]
                )
            ),
    })


CAND = pd.DataFrame(
    candidate_rows
)

CAND["candidate_score"] = (
    CAND[
        "weighted_structural_association"
    ] *
    CAND[
        "directional_consistency"
    ]
)

CAND = CAND.sort_values(
    [
        "sample",
        "candidate_score",
    ],
    ascending=[
        True,
        False,
    ],
)

CAND.to_csv(
    OUT /
    "descriptive_structural_gene_candidates.csv",
    index=False,
)


# Top 25 per specimen for immediate inspection.
TOP = (
    CAND
    .groupby(
        "sample",
        group_keys=False,
    )
    .head(25)
)

TOP.to_csv(
    OUT /
    "top25_structural_gene_candidates.csv",
    index=False,
)


manifest = {
    "version":
        "SUTRA.Fig5.recruitment_v2",
    "purpose":
        (
            "Exact hierarchy ancestry reconstruction and "
            "descriptive molecular profiling of major "
            "architectural recruitment events."
        ),
    "important_semantics": {
        "recipient":
            "larger pre-event collective",
        "incoming":
            "smaller pre-event collective",
        "driver_claim":
            (
                "NOT established. Candidate scores are "
                "descriptive structural associations."
            ),
        "local_control":
            (
                "cells in expanded event bounding box "
                "excluding event membership; not yet "
                "composition matched"
            ),
        "u":
            (
                "cumulative selected merger count / "
                "Level-0 cell count for v073 replay"
            ),
    },
    "top_events_per_sample":
        TOP_EVENTS_PER_SAMPLE,
    "min_incoming_mass":
        MIN_SMALL_OBJECT,
    "min_post_mass":
        MIN_POST_MASS,
}

with open(
    OUT / "manifest.json",
    "w",
) as f:
    json.dump(
        manifest,
        f,
        indent=2,
    )


print()
print("=" * 88)
print("SUTRA FIG5 RECRUITMENT V2 COMPLETE")
print("=" * 88)
print(OUT)
print()
print("TOP CANDIDATES")
print(
    TOP[
        [
            "sample",
            "gene",
            "candidate_score",
            "directional_consistency",
            "median_abs_incoming_vs_recipient",
            "median_abs_incoming_vs_local",
        ]
    ]
    .groupby("sample")
    .head(10)
    .to_string(index=False)
)
