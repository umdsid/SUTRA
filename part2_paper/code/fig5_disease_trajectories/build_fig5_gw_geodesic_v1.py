#!/usr/bin/env python3

from pathlib import Path
import json
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.csgraph import dijkstra
import ot


ROOT = Path.home() / "Desktop" / "SUTRA"

L0 = ROOT / "results/hierarchy_level0_v070"
HIER = ROOT / "results/hierarchy_v0911_specimen_local_contextual_flow/ledger"

OUT = (
    ROOT
    / "results/Fig5_Disease_Trajectories/gw_geodesic_v1"
)
OUT.mkdir(parents=True, exist_ok=True)

SAMPLES = {
    "reference": "nondiseased_kidney",
    "disease": "prcc",
}

# Common hierarchy coordinate = fraction of Level-0 cells removed by mergers.
U_TARGETS = np.array([
    0.00, 0.08, 0.16, 0.24,
    0.32, 0.40, 0.48, 0.56,
])

K_VALUES = [64, 128, 256]

# Fixed seed only for POT initialization if needed.
SEED = 20260920

EPS = 1e-12


def checkpoint_step(path):
    m = re.search(r"labels_(\d+)\.npz$", path.name)
    if not m:
        raise ValueError(path)
    return int(m.group(1))


def load_steps(sample):
    path = HIER / sample / "steps.jsonl"
    rows = []
    with path.open() as f:
        for line in f:
            x = json.loads(line)
            rows.append(x)
    return pd.DataFrame(rows)


def infer_u_column(df):
    # Production v0911 hierarchy coordinate.
    # This is the fraction of Level-0 objects removed by accepted mergers.
    candidates = [
        "hierarchy_coordinate_removed_fraction",
        "removed_fraction_after",
        "removed_fraction",
        "u",
        "fraction_removed",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not identify hierarchy coordinate. "
        f"Columns={list(df.columns)}"
    )


def infer_step_column(df):
    candidates = [
        "microstep",
        "step",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise RuntimeError(
        f"Could not identify step column. "
        f"Columns={list(df.columns)}"
    )


def checkpoint_catalog(sample):
    steps = load_steps(sample)
    # Checkpoint labels encode the hierarchy state after the
    # corresponding productive microstep.
    uc = (
        "removed_fraction_after"
        if "removed_fraction_after" in steps.columns
        else infer_u_column(steps)
    )
    sc = infer_step_column(steps)

    # step -> u
    step_to_u = {
        int(r[sc]): float(r[uc])
        for _, r in steps.iterrows()
        if pd.notna(r[uc])
    }

    cps = sorted(
        (HIER / sample / "label_checkpoints").glob("labels_*.npz"),
        key=checkpoint_step,
    )

    rows = []
    for p in cps:
        st = checkpoint_step(p)

        if st == 0:
            u = 0.0
        elif st in step_to_u:
            u = step_to_u[st]
        else:
            # Nearest recorded microstep only for assigning checkpoint u.
            keys = np.asarray(list(step_to_u), dtype=int)
            nearest = int(keys[np.argmin(np.abs(keys - st))])
            u = step_to_u[nearest]

        rows.append({
            "step": st,
            "u": u,
            "path": str(p),
        })

    return pd.DataFrame(rows)


def choose_checkpoint(cat, u):
    i = int(np.argmin(np.abs(cat["u"].to_numpy() - u)))
    return cat.iloc[i]


def load_level0_graph(sample):
    p = L0 / sample / "graph_csr.npz"
    z = np.load(p)

    indptr = z["indptr"]
    indices = z["indices"]

    n = len(z["node_ids"])

    # Native contact topology. Every finite interface occurs twice in CSR.
    data = np.ones(len(indices), dtype=np.float64)

    A = sparse.csr_matrix(
        (data, indices, indptr),
        shape=(n, n),
    )

    # Force binary symmetric contact representation.
    A.data[:] = 1.0
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()

    return A


def quotient_graph(A0, labels):
    """
    Collapse Level-0 contact graph onto current SUTRA objects.

    Returns
    -------
    Q : sparse binary adjacency among current objects
    mass : Level-0 cell count in each object
    object_labels : original surviving labels
    inverse : Level-0 cell -> quotient object index
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
            np.ones(len(rr), dtype=np.float64),
            (rr, cc),
        ),
        shape=(n_obj, n_obj),
    ).tocsr()

    Q.data[:] = 1.0
    Q.eliminate_zeros()

    return Q, mass, object_labels, inverse


def mass_spectrum(mass):
    x = np.sort(mass)[::-1]
    cs = np.cumsum(x) / x.sum()

    out = {}
    for f in [0.50, 0.80, 0.90, 0.95, 0.99]:
        out[f] = int(np.searchsorted(cs, f) + 1)

    return out


def choose_anchors(Q, mass, K):
    """
    Deterministic structural sketch.

    Begin with largest-mass SUTRA objects. If a connected component has
    no mass-selected anchor, give that component its largest object.

    The resulting anchor count can therefore exceed K slightly.
    """
    n = len(mass)

    order = np.lexsort((
        np.arange(n),
        -mass,
    ))

    anchors = list(order[:min(K, n)])

    ncomp, comp = sparse.csgraph.connected_components(
        Q,
        directed=False,
    )

    anchor_set = set(anchors)

    for c in range(ncomp):
        nodes = np.flatnonzero(comp == c)
        if not any(i in anchor_set for i in nodes):
            best = nodes[
                np.argmax(mass[nodes])
            ]
            anchors.append(int(best))
            anchor_set.add(int(best))

    anchors = np.asarray(
        sorted(
            set(anchors),
            key=lambda i: (-mass[i], i),
        ),
        dtype=int,
    )

    return anchors


def assign_to_anchors(Q, mass, anchors):
    """
    Assign every quotient object to its nearest anchor in contact-graph
    shortest-path distance.

    Ties are broken toward the anchor with larger mass, then smaller id.
    """
    # Unweighted graph distance from each anchor to every object.
    D = dijkstra(
        Q,
        directed=False,
        indices=anchors,
        unweighted=True,
    )

    if D.ndim == 1:
        D = D[None, :]

    # Resolve ties deterministically.
    anchor_mass = mass[anchors]
    rank = np.lexsort((
        anchors,
        -anchor_mass,
    ))

    # Tiny deterministic penalty establishes tie preference.
    penalty = np.empty(len(anchors), dtype=float)
    penalty[rank] = np.arange(len(anchors)) * 1e-9

    score = D + penalty[:, None]
    owner = np.argmin(score, axis=0)

    # Every component should contain an anchor.
    mind = np.min(D, axis=0)
    if not np.all(np.isfinite(mind)):
        bad = int(np.sum(~np.isfinite(mind)))
        raise RuntimeError(
            f"{bad} quotient objects not reachable from an anchor"
        )

    sketch_mass = np.bincount(
        owner,
        weights=mass,
        minlength=len(anchors),
    )

    return owner, sketch_mass, D


def sketch_relational_metric(Q, mass, anchors, owner, sketch_mass):
    """
    Construct anchor-level quotient topology after assigning every
    SUTRA object to an anchor.

    Edges exist whenever original quotient objects assigned to different
    anchors are in native contact.

    Metric = shortest-path distance on this sketched contact graph.
    """
    coo = sparse.triu(Q, k=1).tocoo()

    a = owner[coo.row]
    b = owner[coo.col]

    keep = a != b
    a = a[keep]
    b = b[keep]

    rr = np.concatenate([a, b])
    cc = np.concatenate([b, a])

    S = sparse.coo_matrix(
        (
            np.ones(len(rr), dtype=float),
            (rr, cc),
        ),
        shape=(len(anchors), len(anchors)),
    ).tocsr()

    S.data[:] = 1.0
    S.eliminate_zeros()

    # A disconnected sketch is permitted because tissue can have separate
    # connected components. GW requires finite relational values, so use a
    # finite between-component penalty after computing within-component
    # shortest paths.
    C = dijkstra(
        S,
        directed=False,
        unweighted=True,
    )

    finite = C[np.isfinite(C)]

    if finite.size == 0:
        raise RuntimeError("No finite sketch distances")

    diameter = max(float(finite.max()), 1.0)

    # Explicit disconnected-component penalty.
    C[~np.isfinite(C)] = 1.25 * diameter

    # Normalize scale so comparisons are about relational organization
    # rather than raw graph diameter.
    positive = C[C > 0]
    scale = np.median(positive) if positive.size else 1.0

    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    C = C / scale

    p = sketch_mass / sketch_mass.sum()

    return C, p, S


def run_gw(C1, C2, p, q):
    """
    Squared-loss GW.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        T, log = ot.gromov.gromov_wasserstein(
            C1,
            C2,
            p,
            q,
            loss_fun="square_loss",
            symmetric=True,
            log=True,
            max_iter=200,
            tol_rel=1e-9,
            tol_abs=1e-9,
            verbose=False,
        )

    if "gw_dist" in log:
        gw2 = float(log["gw_dist"])
    else:
        gw2 = float(
            ot.gromov.gromov_wasserstein2(
                C1,
                C2,
                p,
                q,
                loss_fun="square_loss",
                symmetric=True,
                max_iter=200,
                tol_rel=1e-9,
                tol_abs=1e-9,
            )
        )

    gw = np.sqrt(max(gw2, 0.0))

    return T, gw2, gw


def object_distortion(C1, C2, p, q, T):
    """
    Endpoint-local distortion contributions induced by the GW coupling.

    Disease-side result d2[j] is the expected squared relational
    distortion associated with disease landmark j.

    This is descriptive attribution of GW cost, not a causal quantity.
    """
    # Efficient expansion of:
    # sum_i,k,l (C1[i,k]-C2[j,l])^2 T[i,j] T[k,l]

    # First term.
    a = (C1 ** 2) @ p
    term1 = T.T @ a

    # Second term.
    b = (C2 ** 2) @ q

    # Cross term:
    cross = C1 @ T @ C2.T

    d2 = term1 + q * b - 2.0 * np.sum(T * cross, axis=0)

    # Numerical cleanup.
    d2 = np.maximum(d2, 0.0)

    # Conditional cost given disease landmark j.
    conditional = d2 / np.maximum(q, EPS)

    return conditional


def save_coupling(T, path):
    np.savez_compressed(path, coupling=T)


def plot_trajectory(df):
    fig, ax = plt.subplots(figsize=(6.4, 4.5))

    for K, g in df.groupby("K_requested"):
        g = g.sort_values("u")
        ax.plot(
            g["u"],
            g["gw_distance"],
            marker="o",
            label=f"K={K}",
        )

    ax.set_xlabel("SUTRA hierarchy coordinate, u")
    ax.set_ylabel("GW distance")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        OUT / "kidney_gw_distance_vs_u_v1.png",
        dpi=300,
    )
    fig.savefig(
        OUT / "kidney_gw_distance_vs_u_v1.pdf",
    )
    plt.close(fig)


def plot_support_audit(df):
    fig, ax = plt.subplots(figsize=(6.6, 4.6))

    for sample, g in df.groupby("sample"):
        g = g.sort_values("u")
        ax.plot(
            g["u"],
            g["n_for_90pct_mass"],
            marker="o",
            label=sample,
        )

    ax.set_xlabel("SUTRA hierarchy coordinate, u")
    ax.set_ylabel("Objects required for 90% tissue mass")
    ax.legend(frameon=False)
    fig.tight_layout()

    fig.savefig(
        OUT / "kidney_support_mass_audit_v1.png",
        dpi=300,
    )
    plt.close(fig)


def main():
    np.random.seed(SEED)

    print("=" * 100)
    print("SUTRA FIGURE 5 — KIDNEY GW GEODESIC FEASIBILITY V1")
    print("=" * 100)
    print("POT:", ot.__version__)
    print("Output:", OUT)

    state = {}

    # Load native Level-0 contact graph once per specimen.
    for role, sample in SAMPLES.items():
        print("\nLoading", sample)
        A0 = load_level0_graph(sample)
        cat = checkpoint_catalog(sample)

        state[role] = {
            "sample": sample,
            "A0": A0,
            "catalog": cat,
        }

        print(
            "Level-0 nodes:", A0.shape[0],
            "undirected contacts:", A0.nnz // 2,
            "checkpoints:", len(cat),
        )

    support_rows = []
    gw_rows = []

    # Cache quotient states because K changes but checkpoint does not.
    quotient_cache = {}

    for u_target in U_TARGETS:
        print("\n" + "-" * 100)
        print("TARGET u =", u_target)
        print("-" * 100)

        selected = {}

        for role in ["reference", "disease"]:
            sample = state[role]["sample"]
            cat = state[role]["catalog"]

            row = choose_checkpoint(cat, u_target)
            cp_path = Path(row["path"])
            labels = np.load(cp_path)["labels"]

            key = (sample, int(row["step"]))

            if key not in quotient_cache:
                Q, mass, obj_labels, inverse = quotient_graph(
                    state[role]["A0"],
                    labels,
                )
                quotient_cache[key] = (
                    Q,
                    mass,
                    obj_labels,
                    inverse,
                )

            Q, mass, obj_labels, inverse = quotient_cache[key]

            ms = mass_spectrum(mass)

            print(
                role,
                sample,
                "checkpoint", int(row["step"]),
                "actual_u", round(float(row["u"]), 6),
                "objects", len(mass),
                "M1", int(mass.max()),
                "K90", ms[0.90],
            )

            support_rows.append({
                "role": role,
                "sample": sample,
                "u_target": float(u_target),
                "u": float(row["u"]),
                "step": int(row["step"]),
                "n_objects": len(mass),
                "max_mass": float(mass.max()),
                "n_for_50pct_mass": ms[0.50],
                "n_for_80pct_mass": ms[0.80],
                "n_for_90pct_mass": ms[0.90],
                "n_for_95pct_mass": ms[0.95],
                "n_for_99pct_mass": ms[0.99],
            })

            selected[role] = {
                "row": row,
                "Q": Q,
                "mass": mass,
                "obj_labels": obj_labels,
                "inverse": inverse,
            }

        for K in K_VALUES:
            print(f"\nGW K={K}")

            sketches = {}

            for role in ["reference", "disease"]:
                x = selected[role]

                anchors = choose_anchors(
                    x["Q"],
                    x["mass"],
                    K,
                )

                owner, smass, _ = assign_to_anchors(
                    x["Q"],
                    x["mass"],
                    anchors,
                )

                C, p, S = sketch_relational_metric(
                    x["Q"],
                    x["mass"],
                    anchors,
                    owner,
                    smass,
                )

                sketches[role] = {
                    "anchors": anchors,
                    "owner": owner,
                    "mass": smass,
                    "C": C,
                    "p": p,
                    "S": S,
                }

                print(
                    role,
                    "requested", K,
                    "actual", len(anchors),
                    "mass sum", smass.sum(),
                )

            a = sketches["reference"]
            b = sketches["disease"]

            T, gw2, gw = run_gw(
                a["C"],
                b["C"],
                a["p"],
                b["p"],
            )

            distortion_disease = object_distortion(
                a["C"],
                b["C"],
                a["p"],
                b["p"],
                T,
            )

            print(
                "GW^2 =", gw2,
                "GW =", gw,
                "coupling mass =", T.sum(),
            )

            u_actual = min(
                float(selected["reference"]["row"]["u"]),
                float(selected["disease"]["row"]["u"]),
            )

            tag = (
                f"u{u_target:.2f}"
                .replace(".", "p")
                + f"_K{K}"
            )

            save_coupling(
                T,
                OUT / f"coupling_{tag}.npz",
            )

            np.savez_compressed(
                OUT / f"sketch_{tag}.npz",
                C_reference=a["C"],
                C_disease=b["C"],
                p_reference=a["p"],
                p_disease=b["p"],
                anchors_reference=a["anchors"],
                anchors_disease=b["anchors"],
                owner_reference=a["owner"],
                owner_disease=b["owner"],
                disease_conditional_distortion=distortion_disease,
            )

            gw_rows.append({
                "u_target": float(u_target),
                "u_reference": float(
                    selected["reference"]["row"]["u"]
                ),
                "u_disease": float(
                    selected["disease"]["row"]["u"]
                ),
                "u": u_actual,
                "step_reference": int(
                    selected["reference"]["row"]["step"]
                ),
                "step_disease": int(
                    selected["disease"]["row"]["step"]
                ),
                "K_requested": K,
                "K_reference": len(a["anchors"]),
                "K_disease": len(b["anchors"]),
                "gw_squared": gw2,
                "gw_distance": gw,
                "coupling_mass": float(T.sum()),
                "max_disease_conditional_distortion":
                    float(distortion_disease.max()),
                "median_disease_conditional_distortion":
                    float(np.median(distortion_disease)),
            })

    support = pd.DataFrame(support_rows)
    gw = pd.DataFrame(gw_rows)

    support.to_csv(
        OUT / "support_mass_audit.csv",
        index=False,
    )
    gw.to_csv(
        OUT / "gw_scale_trajectory.csv",
        index=False,
    )

    plot_support_audit(support)
    plot_trajectory(gw)

    # Convergence summary across K.
    pivot = gw.pivot_table(
        index="u_target",
        columns="K_requested",
        values="gw_distance",
    )

    pivot.to_csv(
        OUT / "gw_K_convergence.csv"
    )

    print("\n" + "=" * 100)
    print("GW SCALE TRAJECTORY")
    print("=" * 100)
    print(
        gw[
            [
                "u_target",
                "u_reference",
                "u_disease",
                "K_requested",
                "K_reference",
                "K_disease",
                "gw_distance",
            ]
        ].to_string(index=False)
    )

    print("\n" + "=" * 100)
    print("K CONVERGENCE")
    print("=" * 100)
    print(pivot.to_string())

    # Select strongest scale at highest requested resolution.
    hi = gw[
        gw["K_requested"] == max(K_VALUES)
    ].copy()

    best = hi.loc[
        hi["gw_distance"].idxmax()
    ].to_dict()

    with (
        OUT / "strongest_scale.json"
    ).open("w") as f:
        json.dump(
            {
                k: (
                    float(v)
                    if isinstance(v, (np.floating, float))
                    else int(v)
                    if isinstance(v, (np.integer, int))
                    else v
                )
                for k, v in best.items()
            },
            f,
            indent=2,
        )

    provenance = {
        "analysis": "kidney GW geodesic feasibility v1",
        "reference": SAMPLES["reference"],
        "disease": SAMPLES["disease"],
        "structural_input": (
            "frozen Level-0 native cell-cell contact graph, "
            "quotiented by production v0911 SUTRA labels"
        ),
        "expression_used_in_GW": False,
        "mechanics_used_in_GW": False,
        "spatial_coordinates_used_in_GW": False,
        "relational_metric": (
            "shortest-path distance on deterministic "
            "mass-preserving sketches of SUTRA quotient contact graphs"
        ),
        "distance_normalization": (
            "each sketch shortest-path matrix divided by "
            "its median positive pairwise distance"
        ),
        "gw_loss": "square_loss",
        "K_values": K_VALUES,
        "interpretation": (
            "GW coordinate is relational interpolation geometry, "
            "not biological time"
        ),
    }

    with (
        OUT / "provenance.json"
    ).open("w") as f:
        json.dump(provenance, f, indent=2)

    print("\nStrongest K=256 scale:")
    print(json.dumps(best, indent=2, default=float))

    print("\nCOMPLETE")
    print(OUT)


if __name__ == "__main__":
    main()
