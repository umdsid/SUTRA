#!/usr/bin/env python3

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "Desktop" / "SUTRA" / "results" / "Fig5_Disease_Trajectories"
V2 = ROOT / "biology_v2"
OUT = ROOT / "disease_shift_v3"
OUT.mkdir(parents=True, exist_ok=True)

EXPR = V2 / "expression"

PAIRS = {
    "brain": ("healthy_reference", "alzheimers"),
    "kidney": ("nondiseased_kidney", "prcc"),
}


def load_summary(sample):
    p = EXPR / f"{sample}_gene_recruitment_summary.csv"
    d = pd.read_csv(p)

    assert len(d) == 541
    assert d["gene"].is_unique

    return d


def build_shift_table(organ, ref, disease):
    a = load_summary(ref).reset_index(drop=True)
    b = load_summary(disease).reset_index(drop=True)

    # --------------------------------------------------------------
    # Cross-specimen feature alignment
    # --------------------------------------------------------------
    #
    # Each specimen has 541 measured features, but the feature order
    # is specimen-specific. Gene symbols are unique within each of the
    # four v2 tables, so gene is the valid cross-specimen key.
    #
    # We explicitly audit uniqueness and panel overlap before merging.

    if a["gene"].isna().any() or b["gene"].isna().any():
        raise RuntimeError(
            f"{organ}: missing gene labels in recruitment summary"
        )

    if not a["gene"].is_unique:
        dup = (
            a.loc[a["gene"].duplicated(False), "gene"]
            .astype(str)
            .tolist()
        )
        raise RuntimeError(
            f"{organ}: duplicate reference gene labels: "
            f"{dup[:20]}"
        )

    if not b["gene"].is_unique:
        dup = (
            b.loc[b["gene"].duplicated(False), "gene"]
            .astype(str)
            .tolist()
        )
        raise RuntimeError(
            f"{organ}: duplicate disease gene labels: "
            f"{dup[:20]}"
        )

    ref_genes = set(a["gene"].astype(str))
    disease_genes = set(b["gene"].astype(str))

    common = ref_genes & disease_genes
    only_ref = ref_genes - disease_genes
    only_disease = disease_genes - ref_genes

    print(
        f"{organ}: reference features={len(ref_genes)}, "
        f"disease features={len(disease_genes)}, "
        f"common={len(common)}, "
        f"reference-only={len(only_ref)}, "
        f"disease-only={len(only_disease)}"
    )

    if only_ref:
        print(
            f"{organ}: reference-only genes:",
            sorted(only_ref),
        )

    if only_disease:
        print(
            f"{organ}: disease-only genes:",
            sorted(only_disease),
        )

    if len(common) == 0:
        raise RuntimeError(
            f"{organ}: no common genes between specimens"
        )

    d = a.merge(
        b,
        on="gene",
        how="inner",
        validate="one_to_one",
        suffixes=("_ref", "_disease"),
        sort=False,
    )

    if len(d) != len(common):
        raise RuntimeError(
            f"{organ}: merged {len(d)} rows but expected "
            f"{len(common)} common genes"
        )

    for contrast in ["IR", "IL"]:
        d[f"shift_{contrast}"] = (
            d[f"median_delta_{contrast}_disease"]
            - d[f"median_delta_{contrast}_ref"]
        )

    # A shift is concordant only when both structural contrasts
    # move in the same nonzero direction.
    d["shift_concordant"] = (
        np.sign(d["shift_IR"])
        == np.sign(d["shift_IL"])
    ) & (np.sign(d["shift_IR"]) != 0)

    d["shift_direction"] = np.where(
        d["shift_concordant"],
        np.where(d["shift_IR"] > 0, "increased", "decreased"),
        "discordant",
    )

    # Require both IR and IL to reverse sign for a full
    # structural-recruitment reversal.
    d["full_reversal"] = (
        np.sign(d["median_delta_IR_ref"])
        != np.sign(d["median_delta_IR_disease"])
    ) & (
        np.sign(d["median_delta_IL_ref"])
        != np.sign(d["median_delta_IL_disease"])
    )

    # Conservative effect magnitude:
    # the weaker of the two disease-reference changes.
    # This is NOT a biological score; it is only useful for
    # sorting concordant changes.
    d["min_abs_shift"] = np.minimum(
        np.abs(d["shift_IR"]),
        np.abs(d["shift_IL"]),
    )

    # Strength of recruitment in each state, requiring agreement
    # between recipient and local comparisons.
    d["ref_min_abs_effect"] = np.minimum(
        np.abs(d["median_delta_IR_ref"]),
        np.abs(d["median_delta_IL_ref"]),
    )

    d["disease_min_abs_effect"] = np.minimum(
        np.abs(d["median_delta_IR_disease"]),
        np.abs(d["median_delta_IL_disease"]),
    )

    # Descriptive structural classes.
    #
    # Threshold intentionally modest and explicit. These are
    # visualization classes, NOT significance calls.
    eps = 0.05

    ref_pos = (
        (d["median_delta_IR_ref"] >= eps)
        & (d["median_delta_IL_ref"] >= eps)
    )
    ref_neg = (
        (d["median_delta_IR_ref"] <= -eps)
        & (d["median_delta_IL_ref"] <= -eps)
    )
    dis_pos = (
        (d["median_delta_IR_disease"] >= eps)
        & (d["median_delta_IL_disease"] >= eps)
    )
    dis_neg = (
        (d["median_delta_IR_disease"] <= -eps)
        & (d["median_delta_IL_disease"] <= -eps)
    )

    ref_active = ref_pos | ref_neg
    dis_active = dis_pos | dis_neg

    cls = np.full(len(d), "mixed/weak", dtype=object)

    cls[(~ref_active) & dis_active] = "gained"
    cls[ref_active & (~dis_active)] = "lost"

    same_active = (
        (ref_pos & dis_pos)
        | (ref_neg & dis_neg)
    )
    cls[same_active] = "conserved"

    reversed_active = (
        (ref_pos & dis_neg)
        | (ref_neg & dis_pos)
    )
    cls[reversed_active] = "reversed"

    d["structural_class"] = cls
    d.insert(0, "organ", organ)
    d.insert(1, "reference", ref)
    d.insert(2, "disease", disease)

    return d


def print_audit(d, organ):
    print("\n" + "=" * 90)
    print(organ.upper())
    print("=" * 90)

    print("genes:", len(d))
    print(
        "concordant shifts:",
        int(d["shift_concordant"].sum()),
    )
    print(
        "full sign reversals:",
        int(d["full_reversal"].sum()),
    )

    print("\nStructural classes:")
    print(
        d["structural_class"]
        .value_counts()
        .to_string()
    )

    q = (
        d[d["shift_concordant"]]
        .sort_values(
            "min_abs_shift",
            ascending=False,
        )
        .head(25)
    )

    cols = [
        "gene",
        "median_delta_IR_ref",
        "median_delta_IR_disease",
        "shift_IR",
        "median_delta_IL_ref",
        "median_delta_IL_disease",
        "shift_IL",
        "structural_class",
        "full_reversal",
    ]

    print("\nLargest concordant changes:")
    print(q[cols].to_string(index=False))


def plot_shift_plane(d, organ):
    fig, ax = plt.subplots(figsize=(7.2, 6.6))

    # Background genes.
    bg = d[~d["shift_concordant"]]
    ax.scatter(
        bg["shift_IR"],
        bg["shift_IL"],
        s=14,
        alpha=0.25,
        linewidths=0,
    )

    fg = d[d["shift_concordant"]]
    ax.scatter(
        fg["shift_IR"],
        fg["shift_IL"],
        s=24,
        alpha=0.72,
        linewidths=0,
    )

    ax.axhline(0, linewidth=0.8)
    ax.axvline(0, linewidth=0.8)

    lim = np.nanmax(
        np.abs(
            np.concatenate([
                d["shift_IR"].to_numpy(),
                d["shift_IL"].to_numpy(),
            ])
        )
    )
    lim *= 1.08

    ax.plot(
        [-lim, lim],
        [-lim, lim],
        linestyle="--",
        linewidth=0.8,
        alpha=0.5,
    )

    # Label only the strongest concordant shifts.
    lab = (
        fg.sort_values(
            "min_abs_shift",
            ascending=False,
        )
        .head(15)
    )

    for _, r in lab.iterrows():
        ax.annotate(
            r["gene"],
            (r["shift_IR"], r["shift_IL"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7.5,
        )

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel(
        "Disease − reference change\n"
        "incoming vs recipient"
    )
    ax.set_ylabel(
        "Disease − reference change\n"
        "incoming vs local tissue"
    )

    ax.set_title(
        f"{organ.capitalize()}: change in structural recruitment"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / f"{organ}_structural_shift_plane_v3.pdf"
    )
    fig.savefig(
        OUT / f"{organ}_structural_shift_plane_v3.png",
        dpi=300,
    )

    plt.close(fig)


def plot_class_effects(d, organ):
    keep = d[
        d["structural_class"].isin(
            ["gained", "lost", "reversed"]
        )
    ].copy()

    keep = keep.sort_values(
        "min_abs_shift",
        ascending=False,
    ).head(24)

    if keep.empty:
        return

    keep = keep.iloc[::-1]

    y = np.arange(len(keep))

    fig, ax = plt.subplots(
        figsize=(8.0, max(5.0, 0.28 * len(keep) + 1.8))
    )

    ax.hlines(
        y,
        keep["median_delta_IR_ref"],
        keep["median_delta_IR_disease"],
        linewidth=1.0,
        alpha=0.65,
    )

    ax.scatter(
        keep["median_delta_IR_ref"],
        y,
        s=28,
        label="Reference",
    )

    ax.scatter(
        keep["median_delta_IR_disease"],
        y,
        s=28,
        label="Disease",
    )

    ax.axvline(0, linewidth=0.8)

    ax.set_yticks(y)
    ax.set_yticklabels(keep["gene"])

    ax.set_xlabel(
        "Median structural recruitment effect\n"
        "(incoming − recipient)"
    )

    ax.set_title(
        f"{organ.capitalize()}: gained, lost and reversed recruitment"
    )

    ax.legend(
        frameon=False,
        loc="best",
    )

    fig.tight_layout()

    fig.savefig(
        OUT / f"{organ}_structural_classes_v3.pdf"
    )
    fig.savefig(
        OUT / f"{organ}_structural_classes_v3.png",
        dpi=300,
    )

    plt.close(fig)


def main():
    print("=" * 90)
    print("SUTRA FIGURE 5 — DISEASE-SPECIFIC STRUCTURAL RECRUITMENT V3")
    print("=" * 90)

    all_tables = []

    for organ, (ref, disease) in PAIRS.items():
        d = build_shift_table(
            organ,
            ref,
            disease,
        )

        d.to_csv(
            OUT / f"{organ}_gene_structural_shift_v3.csv",
            index=False,
        )

        print_audit(d, organ)
        plot_shift_plane(d, organ)
        plot_class_effects(d, organ)

        all_tables.append(d)

    all_d = pd.concat(
        all_tables,
        ignore_index=True,
    )

    all_d.to_csv(
        OUT / "all_gene_structural_shift_v3.csv",
        index=False,
    )

    provenance = {
        "analysis": "Fig5 disease-specific structural recruitment v3",
        "input": str(V2),
        "hierarchy_replayed": False,
        "expression_reloaded": False,
        "gene_count_per_comparison": "common measured genes; audited per comparison",
        "comparisons": {
            k: {
                "reference": v[0],
                "disease": v[1],
            }
            for k, v in PAIRS.items()
        },
        "interpretation": (
            "Descriptive specimen-level change in molecular association "
            "with SUTRA structural recruitment. Hierarchy position is "
            "organizational scale, not biological time. Results do not "
            "establish causal gene drivers or population-level disease effects."
        ),
    }

    with open(
        OUT / "provenance.json",
        "w",
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
