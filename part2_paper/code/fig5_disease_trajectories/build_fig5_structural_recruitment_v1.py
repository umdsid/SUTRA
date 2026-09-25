#!/usr/bin/env python3

from pathlib import Path
import json
import math
import hashlib
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


# =============================================================================
# PATHS
# =============================================================================

ROOT = Path.home() / "Desktop" / "SUTRA"

V0911 = (
    ROOT / "results" /
    "hierarchy_v0911_specimen_local_contextual_flow" /
    "ledger"
)

LEVEL0 = ROOT / "results" / "hierarchy_level0_v070"

OUT = (
    ROOT / "results" /
    "Fig5_Disease_Trajectories" /
    "structural_recruitment_v1"
)

EVENT_OUT = OUT / "event_ledger"
SELECT_OUT = OUT / "event_selection"
AUDIT_OUT = OUT / "checkpoints_audit"
FIG_OUT = OUT / "figures"

for d in [OUT, EVENT_OUT, SELECT_OUT, AUDIT_OUT, FIG_OUT]:
    d.mkdir(parents=True, exist_ok=True)


SAMPLES = [
    "healthy_reference",
    "alzheimers",
    "gbm_reference_addon",
    "nondiseased_kidney",
    "prcc",
]

DISPLAY = {
    "healthy_reference": "Healthy brain",
    "alzheimers": "Alzheimer's",
    "gbm_reference_addon": "GBM-related",
    "nondiseased_kidney": "Nondiseased kidney",
    "prcc": "PRCC",
}

ORGANS = {
    "healthy_reference": "brain",
    "alzheimers": "brain",
    "gbm_reference_addon": "brain",
    "nondiseased_kidney": "kidney",
    "prcc": "kidney",
}

PRIMARY = {
    "brain": ("healthy_reference", "alzheimers"),
    "kidney": ("nondiseased_kidney", "prcc"),
}


# =============================================================================
# HELPERS
# =============================================================================

def read_summary(sample):
    p = V0911 / sample / "flow_summary.json"
    with open(p) as f:
        return json.load(f)


def read_cells(sample):
    p = LEVEL0 / sample / "cells.parquet"
    cols = [
        "cell_index",
        "matrix_column",
        "x",
        "y",
        "patch_id",
    ]
    return pq.read_table(p, columns=cols).to_pandas()


def checkpoint_files(sample):
    d = V0911 / sample / "label_checkpoints"
    return sorted(d.glob("labels_*.npz"))


def checkpoint_number(path):
    return int(path.stem.split("_")[1])


def load_labels(path):
    with np.load(path, allow_pickle=False) as z:
        return np.asarray(z["labels"], dtype=np.int64)


def event_files(sample):
    return sorted(
        (V0911 / sample / "merge_events").glob("step_*.parquet")
    )


def step_number(path):
    return int(path.stem.split("_")[1])


def sha256(path, block=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# =============================================================================
# CHECKPOINT VALIDATION
# =============================================================================

def audit_checkpoints(sample, summary):
    files = checkpoint_files(sample)

    records = []

    for p in files:
        s = checkpoint_number(p)
        labels = load_labels(p)

        records.append({
            "sample": sample,
            "checkpoint": s,
            "n_cells": len(labels),
            "n_objects": int(np.unique(labels).size),
            "min_label": int(labels.min()),
            "max_label": int(labels.max()),
        })

    df = pd.DataFrame(records)

    # Initial invariant
    init = df.loc[df["checkpoint"] == 0]
    assert len(init) == 1
    assert int(init.iloc[0]["n_cells"]) == summary["level0_cells"]
    assert int(init.iloc[0]["n_objects"]) == summary["level0_cells"]

    # Final invariant
    final_checkpoint = int(df["checkpoint"].max())
    final = df.loc[df["checkpoint"] == final_checkpoint].iloc[0]

    assert int(final["n_cells"]) == summary["level0_cells"]
    assert int(final["n_objects"]) == summary["final_nodes"]

    out = AUDIT_OUT / f"{sample}_checkpoint_audit.csv"
    df.to_csv(out, index=False)

    return df


# =============================================================================
# EXACT EVENT REPLAY
#
# Active object masses are enough to characterize all 168k accepted events.
# Because accepted mergers within a microstep are node-disjoint, all masses
# are measured from the same pre-step state and then updated simultaneously.
# =============================================================================

KEEP_NATIVE = [
    "microstep",
    "super_i",
    "super_j",
    "survivor_node",
    "removed_node",

    "n_boundary_edges",

    "molecular_distance",
    "mechanics_support_fraction",
    "abs_tension_z",
    "abs_delta_p_z",

    "comm_ij",
    "comm_ji",
    "comm_support",
    "comm_supported",
    "comm_directionality",
    "comm_asymmetry",
    "comm_reciprocity_descriptive",

    "geometry_state_resolved",
    "alpha_cost",
    "directional_correction",
    "geometry_pair_cost",
    "geometry_reversal_ratio",

    "molecular_local",
    "molecular_context",
    "molecular_effective",

    "mechanics_local",
    "mechanics_context",
    "mechanics_effective",

    "communication_support_local",
    "communication_support_context",
    "communication_support_effective",

    "geometry_local",
    "geometry_context",
    "geometry_effective",

    "topology_local",
    "topology_context",
    "topology_effective",

    "context_hops",
    "composite_merge_cost",
    "evidence_coverage",

    "molecular_penalty",
    "mechanics_penalty",
    "communication_penalty",
    "geometry_penalty",
    "topology_penalty",

    "merge_cost_threshold",
    "merge_allowed",
    "admissible",
    "ordering_merit",
]


def materialize_event_ledger(sample, summary):
    N0 = int(summary["level0_cells"])

    # Every Level-0 cell starts as its own active object.
    mass = np.ones(N0, dtype=np.int64)

    # Track birth step of current object.
    birth = np.zeros(N0, dtype=np.int64)

    records = []

    files = event_files(sample)

    total_seen = 0

    for p in files:
        step = step_number(p)

        pf = pq.ParquetFile(p)
        available = set(pf.schema_arrow.names)

        cols = [c for c in KEEP_NATIVE if c in available]

        df = pq.read_table(p, columns=cols).to_pandas()

        assert len(df) > 0

        survivors = df["survivor_node"].to_numpy(dtype=np.int64)
        removed = df["removed_node"].to_numpy(dtype=np.int64)

        # Previously verified globally, but preserve invariant in production.
        endpoints = np.concatenate([survivors, removed])
        assert np.unique(endpoints).size == endpoints.size
        assert np.all(survivors != removed)

        mi = mass[df["super_i"].to_numpy(dtype=np.int64)]
        mj = mass[df["super_j"].to_numpy(dtype=np.int64)]

        ms = mass[survivors]
        mr = mass[removed]

        # Both endpoint descriptions should refer to same pre-step objects.
        # We retain both native i/j and survivor/removed semantics.
        assert np.all(ms > 0)
        assert np.all(mr > 0)

        post = ms + mr

        # Define recipient/incoming only by pre-event mass.
        # Ties remain explicitly flagged.
        recipient = np.where(ms >= mr, survivors, removed)
        incoming = np.where(ms >= mr, removed, survivors)

        recipient_mass = np.maximum(ms, mr)
        incoming_mass = np.minimum(ms, mr)

        tie = ms == mr

        u_before = total_seen / N0
        u_after = (total_seen + len(df)) / N0

        for k in range(len(df)):
            row = df.iloc[k].to_dict()

            row.update({
                "sample": sample,
                "organ": ORGANS[sample],
                "display_name": DISPLAY[sample],

                "step": step,

                "cumulative_merges_before": total_seen,
                "cumulative_merges_after": total_seen + len(df),

                "u_before": u_before,
                "u_after": u_after,

                "super_i_mass_pre": int(mi[k]),
                "super_j_mass_pre": int(mj[k]),

                "survivor_mass_pre": int(ms[k]),
                "removed_mass_pre": int(mr[k]),

                "recipient_node": int(recipient[k]),
                "incoming_node": int(incoming[k]),

                "recipient_mass": int(recipient_mass[k]),
                "incoming_mass": int(incoming_mass[k]),
                "post_mass": int(post[k]),

                "mass_tie": bool(tie[k]),

                "incoming_fraction": (
                    float(incoming_mass[k] / post[k])
                ),

                "absolute_recruitment": int(incoming_mass[k]),

                "recipient_birth_step": int(
                    birth[int(recipient[k])]
                ),
                "incoming_birth_step": int(
                    birth[int(incoming[k])]
                ),
            })

            records.append(row)

        # Simultaneous update.
        # Native hierarchy says removed_node disappears into survivor_node.
        new_mass = ms + mr

        mass[survivors] = new_mass
        mass[removed] = 0

        birth[survivors] = step + 1
        birth[removed] = -1

        total_seen += len(df)

    assert total_seen == int(summary["total_merges"])

    active = mass > 0

    assert int(active.sum()) == int(summary["final_nodes"])
    assert int(mass.sum()) == N0

    out = pd.DataFrame(records)

    # Event-level derived descriptors.
    out["log10_post_mass"] = np.log10(out["post_mass"].astype(float))
    out["log10_incoming_mass"] = np.log10(
        out["incoming_mass"].astype(float)
    )

    out["recruitment_jump_fraction_N0"] = (
        out["incoming_mass"] / N0
    )

    out["balanced_recruitment"] = (
        2.0 * out["incoming_fraction"]
    )

    out["structural_event_magnitude"] = (
        out["recruitment_jump_fraction_N0"]
        * np.sqrt(out["balanced_recruitment"].clip(lower=0))
    )

    path = EVENT_OUT / f"{sample}_accepted_events.parquet"
    out.to_parquet(path, index=False)

    final_mass = pd.DataFrame({
        "node": np.flatnonzero(active),
        "mass": mass[active],
    }).sort_values("mass", ascending=False)

    final_mass.to_csv(
        EVENT_OUT / f"{sample}_final_object_masses.csv",
        index=False,
    )

    return out


# =============================================================================
# EXPRESSION-BLIND EVENT SELECTION
# =============================================================================

def add_event_ranks(df):
    x = df.copy()

    x["rank_abs_recruitment"] = (
        x["incoming_mass"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    x["rank_post_mass"] = (
        x["post_mass"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    x["rank_structural_magnitude"] = (
        x["structural_event_magnitude"]
        .rank(method="min", ascending=False)
        .astype(int)
    )

    return x


def select_structural_events(df, n_each=40):
    """
    Selection is deliberately expression-blind.

    Capture:
      1. largest incoming collectives
      2. largest post-event collectives
      3. largest normalized recruitment events
      4. balanced large mergers

    Union them, then deduplicate.
    """

    x = add_event_ranks(df)

    picks = []

    picks.append(
        x.nlargest(n_each, "incoming_mass")
    )

    picks.append(
        x.nlargest(n_each, "post_mass")
    )

    picks.append(
        x.nlargest(n_each, "structural_event_magnitude")
    )

    balanced = x[
        (x["incoming_fraction"] >= 0.20) &
        (x["post_mass"] >= 20)
    ]

    if len(balanced):
        picks.append(
            balanced.nlargest(n_each, "post_mass")
        )

    out = pd.concat(picks, ignore_index=True)

    out = out.drop_duplicates(
        subset=[
            "sample",
            "step",
            "survivor_node",
            "removed_node",
        ]
    )

    out = out.sort_values(
        [
            "structural_event_magnitude",
            "incoming_mass",
            "post_mass",
        ],
        ascending=False,
    )

    return out


# =============================================================================
# REFERENCE ↔ DISEASE EVENT MATCHING
#
# Match structurally, not molecularly.
# This prevents gene expression from selecting its own comparator.
# =============================================================================

MATCH_FEATURES = [
    "u_before",
    "log10_post_mass",
    "incoming_fraction",
    "log10_incoming_mass",
]


def robust_standardize(A, B):
    both = pd.concat(
        [A[MATCH_FEATURES], B[MATCH_FEATURES]],
        ignore_index=True,
    ).astype(float)

    med = both.median()
    mad = (both - med).abs().median()

    # Avoid zero scale.
    scale = mad.replace(0, np.nan)
    scale = scale.fillna(both.std().replace(0, 1.0))
    scale = scale.fillna(1.0)

    ZA = (A[MATCH_FEATURES].astype(float) - med) / scale
    ZB = (B[MATCH_FEATURES].astype(float) - med) / scale

    return ZA.to_numpy(), ZB.to_numpy()


def greedy_match(reference, disease):
    """
    One-to-one nearest structural matching without expression features.
    """

    if len(reference) == 0 or len(disease) == 0:
        return pd.DataFrame()

    Zr, Zd = robust_standardize(reference, disease)

    used = set()
    records = []

    # Match strongest disease events first.
    order = np.argsort(
        -disease["structural_event_magnitude"].to_numpy()
    )

    for di in order:
        dvec = Zd[di]

        dist = np.sqrt(
            np.sum((Zr - dvec[None, :]) ** 2, axis=1)
        )

        for ri in np.argsort(dist):
            if int(ri) not in used:
                used.add(int(ri))

                dr = disease.iloc[int(di)]
                rr = reference.iloc[int(ri)]

                records.append({
                    "organ": dr["organ"],

                    "disease_sample": dr["sample"],
                    "disease_step": int(dr["step"]),
                    "disease_survivor_node": int(
                        dr["survivor_node"]
                    ),
                    "disease_removed_node": int(
                        dr["removed_node"]
                    ),

                    "reference_sample": rr["sample"],
                    "reference_step": int(rr["step"]),
                    "reference_survivor_node": int(
                        rr["survivor_node"]
                    ),
                    "reference_removed_node": int(
                        rr["removed_node"]
                    ),

                    "structural_match_distance": float(dist[ri]),

                    "disease_u": float(dr["u_before"]),
                    "reference_u": float(rr["u_before"]),

                    "disease_post_mass": int(dr["post_mass"]),
                    "reference_post_mass": int(rr["post_mass"]),

                    "disease_incoming_mass": int(
                        dr["incoming_mass"]
                    ),
                    "reference_incoming_mass": int(
                        rr["incoming_mass"]
                    ),

                    "disease_incoming_fraction": float(
                        dr["incoming_fraction"]
                    ),
                    "reference_incoming_fraction": float(
                        rr["incoming_fraction"]
                    ),
                })
                break

    return pd.DataFrame(records)


# =============================================================================
# FIGURE — STRUCTURAL PREFLIGHT
# =============================================================================

def make_preflight_figure(event_tables, selected_tables, matches):
    fig = plt.figure(figsize=(14, 10))

    gs = fig.add_gridspec(
        3, 2,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.42,
        wspace=0.28,
    )

    comparisons = [
        ("brain", "healthy_reference", "alzheimers"),
        ("kidney", "nondiseased_kidney", "prcc"),
    ]

    # A/B: hierarchy structural-event cloud
    for col, (organ, ref, dis) in enumerate(comparisons):
        ax = fig.add_subplot(gs[0, col])

        for sample in [ref, dis]:
            d = event_tables[sample]

            ax.scatter(
                d["u_before"],
                d["incoming_mass"],
                s=5,
                alpha=0.20,
                label=DISPLAY[sample],
            )

        ax.set_yscale("log")
        ax.set_xlabel("Hierarchy coordinate, $u$")
        ax.set_ylabel("Incoming collective size")
        ax.legend(frameon=False)

        ax.text(
            -0.12, 1.06,
            "A" if col == 0 else "B",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    # C/D: largest structurally nominated events
    for col, (organ, ref, dis) in enumerate(comparisons):
        ax = fig.add_subplot(gs[1, col])

        frames = []

        for sample in [ref, dis]:
            q = selected_tables[sample].copy()
            q = q.nlargest(25, "structural_event_magnitude")
            q["condition"] = DISPLAY[sample]
            frames.append(q)

        q = pd.concat(frames)

        for condition, z in q.groupby("condition"):
            ax.scatter(
                z["post_mass"],
                z["incoming_fraction"],
                s=28,
                alpha=0.75,
                label=condition,
            )

        ax.set_xscale("log")
        ax.set_xlabel("Post-merger collective size")
        ax.set_ylabel("Incoming fraction")
        ax.legend(frameon=False)

        ax.text(
            -0.12, 1.06,
            "C" if col == 0 else "D",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    # E/F: structural matching
    for col, (organ, ref, dis) in enumerate(comparisons):
        ax = fig.add_subplot(gs[2, col])

        m = matches[organ]

        if len(m):
            ax.scatter(
                m["reference_post_mass"],
                m["disease_post_mass"],
                s=24,
                alpha=0.7,
            )

            vals = np.concatenate([
                m["reference_post_mass"].to_numpy(float),
                m["disease_post_mass"].to_numpy(float),
            ])

            lo = max(1, np.nanmin(vals))
            hi = np.nanmax(vals)

            ax.plot(
                [lo, hi],
                [lo, hi],
                linestyle="--",
                linewidth=1,
            )

            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel(f"{DISPLAY[ref]} matched event size")
        ax.set_ylabel(f"{DISPLAY[dis]} event size")

        ax.text(
            -0.12, 1.06,
            "E" if col == 0 else "F",
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top",
        )

    fig.savefig(
        FIG_OUT / "Fig5_structural_preflight_v1.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        FIG_OUT / "Fig5_structural_preflight_v1.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 90)
    print("SUTRA FIGURE 5 — STRUCTURAL RECRUITMENT V1")
    print("=" * 90)

    summaries = {}
    event_tables = {}
    selected_tables = {}

    provenance = {
        "analysis": "Fig5 structural recruitment v1",
        "hierarchy_source": str(V0911),
        "level0_source": str(LEVEL0),
        "selection_uses_expression": False,
        "samples": {},
    }

    for sample in SAMPLES:
        print("\n" + "#" * 90)
        print(sample)
        print("#" * 90)

        summary = read_summary(sample)
        summaries[sample] = summary

        print(
            "N0:",
            summary["level0_cells"],
            "merges:",
            summary["total_merges"],
            "final:",
            summary["final_nodes"],
            "u:",
            summary["removed_fraction"],
        )

        # Validate Level-0 indexing.
        cells = read_cells(sample)

        N0 = int(summary["level0_cells"])

        assert len(cells) == N0

        assert np.array_equal(
            cells["cell_index"].to_numpy(),
            np.arange(N0),
        )

        assert np.array_equal(
            cells["matrix_column"].to_numpy(),
            np.arange(N0),
        )

        print("Level-0 cell ordering: PASS")

        # Validate checkpoint semantics.
        cp = audit_checkpoints(sample, summary)

        print(
            "Checkpoints:",
            len(cp),
            "final objects:",
            int(cp.iloc[-1]["n_objects"]),
            "PASS",
        )

        # Build accepted-event ledger.
        events = materialize_event_ledger(sample, summary)

        event_tables[sample] = events

        print(
            "Accepted events materialized:",
            f"{len(events):,}",
        )

        print(
            "Largest incoming:",
            int(events["incoming_mass"].max()),
            "largest post:",
            int(events["post_mass"].max()),
        )

        # Expression-blind structural event nomination.
        selected = select_structural_events(events)

        selected_tables[sample] = selected

        selected.to_parquet(
            SELECT_OUT / f"{sample}_structural_events.parquet",
            index=False,
        )

        selected.to_csv(
            SELECT_OUT / f"{sample}_structural_events.csv",
            index=False,
        )

        print(
            "Structurally nominated events:",
            len(selected),
        )

        provenance["samples"][sample] = {
            "level0_cells": int(summary["level0_cells"]),
            "total_merges": int(summary["total_merges"]),
            "final_nodes": int(summary["final_nodes"]),
            "removed_fraction": float(summary["removed_fraction"]),
            "stop_reason": summary["stop_reason"],
            "natural_exhaustion": bool(summary["natural_exhaustion"]),
            "specimen_local_calibration": bool(
                summary["specimen_local_calibration"]
            ),
            "event_ledger": str(
                EVENT_OUT / f"{sample}_accepted_events.parquet"
            ),
        }

    # -------------------------------------------------------------------------
    # Organ-matched structural comparisons
    # -------------------------------------------------------------------------

    matches = {}

    for organ, (ref, dis) in PRIMARY.items():
        print("\n" + "=" * 90)
        print("MATCHING:", organ, ref, "<->", dis)
        print("=" * 90)

        r = selected_tables[ref]
        d = selected_tables[dis]

        m = greedy_match(r, d)

        matches[organ] = m

        m.to_csv(
            SELECT_OUT / f"{organ}_reference_disease_matches.csv",
            index=False,
        )

        print("matched events:", len(m))

        if len(m):
            print(
                "median structural distance:",
                float(m["structural_match_distance"].median())
            )

    # -------------------------------------------------------------------------
    # Compact cross-specimen summary
    # -------------------------------------------------------------------------

    summary_rows = []

    for sample in SAMPLES:
        e = event_tables[sample]

        summary_rows.append({
            "sample": sample,
            "display_name": DISPLAY[sample],
            "organ": ORGANS[sample],
            "N0": summaries[sample]["level0_cells"],
            "total_merges": len(e),
            "endpoint_u": summaries[sample]["removed_fraction"],
            "final_nodes": summaries[sample]["final_nodes"],
            "max_incoming_mass": int(e["incoming_mass"].max()),
            "max_post_mass": int(e["post_mass"].max()),
            "median_incoming_fraction": float(
                e["incoming_fraction"].median()
            ),
            "q99_incoming_mass": float(
                e["incoming_mass"].quantile(0.99)
            ),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(
        OUT / "cross_specimen_structural_summary.csv",
        index=False,
    )

    # -------------------------------------------------------------------------
    # Preflight figure
    # -------------------------------------------------------------------------

    make_preflight_figure(
        event_tables,
        selected_tables,
        matches,
    )

    # -------------------------------------------------------------------------
    # Provenance
    # -------------------------------------------------------------------------

    provenance["primary_comparisons"] = PRIMARY
    provenance["gbm_in_primary_figure"] = False
    provenance["reason_gbm_excluded"] = (
        "Biological provenance must be verified before interpreting "
        "gbm_reference_addon as a disease specimen."
    )

    with open(OUT / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print("\n" + "=" * 90)
    print("COMPLETE")
    print("=" * 90)
    print("Output:")
    print(OUT)
    print("\nPreflight figure:")
    print(FIG_OUT / "Fig5_structural_preflight_v1.pdf")
    print("\nNext stage after inspection:")
    print(
        "Exact Level-0 membership + expression profiling for "
        "structurally nominated events."
    )


if __name__ == "__main__":
    main()
