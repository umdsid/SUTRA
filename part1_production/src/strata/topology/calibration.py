from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree


@dataclass(frozen=True)
class CalibrationConfig:
    epsilons: tuple[float, ...] = (
        0.0, 0.025, 0.05, 0.075, 0.10,
        0.15, 0.20, 0.25, 0.35, 0.50
    )
    min_shared_length: float = 0.25
    plateau_rel_edge_change: float = 0.04
    plateau_rel_degree_change: float = 0.04
    plateau_iso_abs_change: float = 0.005
    plateau_required_steps: int = 2
    high_conf_persistence: float = 0.70
    admissible_persistence: float = 0.40
    max_boundary_samples: int = 512


def _finite_geom(g):
    if g is None:
        return False
    try:
        return (not g.is_empty) and np.isfinite(float(g.area)) and float(g.area) > 0
    except Exception:
        return False


def _sample_linework(geom: BaseGeometry, max_samples: int = 512) -> np.ndarray:
    """
    Sample points approximately uniformly along polygon boundary linework.
    Used only to quantify boundary-gap geometry for already spatially indexed
    candidate pairs; never used as a centroid-contact surrogate.
    """
    b = geom.boundary
    L = float(b.length)
    if not np.isfinite(L) or L <= 0:
        return np.empty((0, 2), dtype=float)
    n = max(16, min(max_samples, int(math.ceil(L / max(L/max_samples, 0.05)))))
    fracs = np.linspace(0.0, 1.0, n, endpoint=False)
    pts = []
    for f in fracs:
        try:
            p = b.interpolate(float(f), normalized=True)
            pts.append((float(p.x), float(p.y)))
        except Exception:
            pass
    return np.asarray(pts, dtype=float)


def paired_boundary_metrics(gi: BaseGeometry, gj: BaseGeometry, max_samples: int = 512):
    """
    Symmetric nearest-boundary correspondence metrics.

    Returns:
      min_gap              exact geometry distance between boundaries
      mean_near_gap        mean symmetric sampled nearest-boundary distance
      max_near_gap         robust 95th percentile symmetric distance
      facing_length_scale  harmonic-like proxy from matched sampled support

    This quantifies whether two measured polygon boundaries genuinely face each
    other. It does not use centroid distance for edge creation.
    """
    bi, bj = gi.boundary, gj.boundary
    min_gap = float(bi.distance(bj))

    pi = _sample_linework(gi, max_samples=max_samples)
    pj = _sample_linework(gj, max_samples=max_samples)
    if len(pi) == 0 or len(pj) == 0:
        return {
            "min_gap": min_gap,
            "mean_near_gap": math.inf,
            "q95_near_gap": math.inf,
            "n_samples_i": int(len(pi)),
            "n_samples_j": int(len(pj)),
        }

    ti, tj = cKDTree(pi), cKDTree(pj)
    di, _ = tj.query(pi, k=1)
    dj, _ = ti.query(pj, k=1)
    d = np.concatenate([di, dj]).astype(float)

    return {
        "min_gap": min_gap,
        "mean_near_gap": float(np.mean(d)),
        "q95_near_gap": float(np.quantile(d, 0.95)),
        "n_samples_i": int(len(pi)),
        "n_samples_j": int(len(pj)),
    }


def candidate_pairs(polygons: dict[str, BaseGeometry], max_epsilon: float):
    ids, geoms = [], []
    for cid, g in polygons.items():
        if _finite_geom(g):
            ids.append(str(cid))
            geoms.append(g)

    tree = STRtree(geoms)
    seen = set()
    out = []

    for i, gi in enumerate(geoms):
        q = gi.buffer(max_epsilon) if max_epsilon > 0 else gi
        for jraw in tree.query(q):
            j = int(jraw)
            if j <= i:
                continue
            key = (i, j)
            if key in seen:
                continue
            seen.add(key)
            gj = geoms[j]
            # Hard geometric candidate gate by measured boundary distance.
            try:
                gap = float(gi.boundary.distance(gj.boundary))
            except Exception:
                continue
            if np.isfinite(gap) and gap <= max_epsilon:
                out.append((ids[i], ids[j], gi, gj, gap))
    return out


def calibrate_sample(
    polygons: dict[str, BaseGeometry],
    config: CalibrationConfig,
):
    eps = np.asarray(config.epsilons, dtype=float)
    max_eps = float(np.max(eps))
    pairs = candidate_pairs(polygons, max_eps)

    pair_rows = []
    edge_sets = {float(e): set() for e in eps}

    # Compute geometry once per candidate pair.
    for cell_i, cell_j, gi, gj, min_gap in pairs:
        metrics = paired_boundary_metrics(
            gi, gj, max_samples=config.max_boundary_samples
        )
        row = {
            "cell_i": cell_i,
            "cell_j": cell_j,
            **metrics,
        }
        # A contact at epsilon requires both local closeness and a nontrivial
        # amount of facing boundary support. We estimate facing support using
        # the same measured boundary strip criterion as Tranche 2.
        support_by_eps = {}
        for e in eps:
            e = float(e)
            if e == 0.0:
                try:
                    shared = gi.boundary.intersection(gj.boundary)
                    support = 0.0 if shared is None else float(shared.length)
                except Exception:
                    support = 0.0
            else:
                try:
                    near_i = gi.boundary.intersection(gj.boundary.buffer(e))
                    near_j = gj.boundary.intersection(gi.boundary.buffer(e))
                    li = 0.0 if near_i is None else float(near_i.length)
                    lj = 0.0 if near_j is None else float(near_j.length)
                    support = 0.5 * (li + lj)
                except Exception:
                    support = 0.0
            support_by_eps[e] = support
            row[f"support_eps_{e:.3f}"] = support

            if min_gap <= e + 1e-12 and support >= config.min_shared_length:
                edge_sets[e].add((cell_i, cell_j))

        row["n_eps_present"] = int(sum((cell_i, cell_j) in edge_sets[float(e)] for e in eps))
        row["persistence_fraction"] = float(row["n_eps_present"] / len(eps))
        pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)

    valid_ids = [str(k) for k,g in polygons.items() if _finite_geom(g)]
    n_nodes = len(valid_ids)
    sweep_rows = []
    prev_edges = None

    for e in eps:
        e = float(e)
        E = edge_sets[e]
        deg = {c:0 for c in valid_ids}
        for a,b in E:
            deg[a] += 1
            deg[b] += 1
        degree_values = np.fromiter(deg.values(), dtype=float, count=n_nodes) if n_nodes else np.array([])
        n_edges = len(E)
        isolated = int(np.sum(degree_values == 0)) if n_nodes else 0
        mean_deg = float(np.mean(degree_values)) if n_nodes else 0.0
        median_deg = float(np.median(degree_values)) if n_nodes else 0.0

        if prev_edges is None:
            added = removed = 0
            jacc = None
        else:
            added = len(E - prev_edges)
            removed = len(prev_edges - E)
            union = len(E | prev_edges)
            jacc = float(len(E & prev_edges) / union) if union else 1.0

        sweep_rows.append({
            "epsilon": e,
            "n_edges": n_edges,
            "mean_degree": mean_deg,
            "median_degree": median_deg,
            "n_isolated": isolated,
            "fraction_isolated": float(isolated/n_nodes) if n_nodes else None,
            "edges_added_from_previous": added,
            "edges_removed_from_previous": removed,
            "jaccard_with_previous": jacc,
        })
        prev_edges = E

    sweep = pd.DataFrame(sweep_rows)

    # Stability metrics between successive epsilons.
    for col, out_col in [
        ("n_edges", "rel_edge_change"),
        ("mean_degree", "rel_degree_change"),
    ]:
        prev = sweep[col].shift(1)
        denom = prev.abs().replace(0, np.nan)
        sweep[out_col] = ((sweep[col] - prev).abs() / denom).fillna(np.inf)

    sweep["iso_abs_change"] = (sweep["fraction_isolated"] - sweep["fraction_isolated"].shift(1)).abs()
    sweep.loc[0, "iso_abs_change"] = np.inf

    stable = (
        (sweep["rel_edge_change"] <= config.plateau_rel_edge_change)
        & (sweep["rel_degree_change"] <= config.plateau_rel_degree_change)
        & (sweep["iso_abs_change"] <= config.plateau_iso_abs_change)
    )
    sweep["stable_step"] = stable

    # Earliest epsilon that starts a run of required stable transitions.
    chosen = None
    req = int(config.plateau_required_steps)
    stable_arr = stable.to_numpy(bool)
    for idx in range(1, len(sweep)):
        end = min(len(sweep), idx + req)
        if end - idx == req and stable_arr[idx:end].all():
            chosen = float(sweep.iloc[idx]["epsilon"])
            break

    # Conservative fallback: first epsilon >=0.1 with highest previous-edge Jaccard.
    fallback_used = False
    if chosen is None:
        fallback_used = True
        cand = sweep[sweep["epsilon"] >= 0.10].copy()
        if len(cand):
            jj = cand["jaccard_with_previous"].fillna(-1.0)
            chosen = float(cand.loc[jj.idxmax(), "epsilon"])
        else:
            chosen = float(eps[-1])

    chosen_edges = edge_sets[chosen]
    # Persistence over eps >= chosen: a stringent measure of whether an edge
    # survives as tolerance is relaxed further.
    tail_eps = [float(e) for e in eps if float(e) >= chosen]
    confidence_rows = []
    for a,b in sorted(chosen_edges):
        tail_p = sum((a,b) in edge_sets[e] for e in tail_eps) / len(tail_eps)
        global_p = sum((a,b) in edge_sets[float(e)] for e in eps) / len(eps)
        prow = pair_df[(pair_df.cell_i==a)&(pair_df.cell_j==b)]
        if len(prow):
            r = prow.iloc[0]
            min_gap = float(r.min_gap)
            mean_gap = float(r.mean_near_gap)
            q95_gap = float(r.q95_near_gap)
            support = float(r[f"support_eps_{chosen:.3f}"])
        else:
            min_gap = mean_gap = q95_gap = support = math.nan

        # Confidence is about how early/stably the interface appears relative
        # to the calibrated admissible threshold.
        if min_gap == 0.0 or (global_p >= config.high_conf_persistence and min_gap <= chosen*0.5):
            cls = "high_confidence"
        elif global_p >= config.admissible_persistence:
            cls = "admissible"
        else:
            cls = "marginal"

        confidence_rows.append({
            "cell_i": a,
            "cell_j": b,
            "chosen_epsilon": chosen,
            "min_gap": min_gap,
            "mean_near_gap": mean_gap,
            "q95_near_gap": q95_gap,
            "shared_support_at_chosen": support,
            "global_persistence_fraction": float(global_p),
            "tail_persistence_fraction": float(tail_p),
            "confidence_class": cls,
        })

    confidence = pd.DataFrame(confidence_rows)

    cert = {
        "chosen_epsilon": chosen,
        "fallback_used": fallback_used,
        "plateau_found": not fallback_used,
        "n_nodes": n_nodes,
        "n_candidate_pairs": len(pairs),
        "n_certified_edges": len(chosen_edges),
        "mean_degree_at_chosen": float(
            sweep.loc[sweep.epsilon == chosen, "mean_degree"].iloc[0]
        ),
        "fraction_isolated_at_chosen": float(
            sweep.loc[sweep.epsilon == chosen, "fraction_isolated"].iloc[0]
        ),
        "high_confidence_fraction": float(
            (confidence["confidence_class"]=="high_confidence").mean()
        ) if len(confidence) else 0.0,
        "marginal_fraction": float(
            (confidence["confidence_class"]=="marginal").mean()
        ) if len(confidence) else 0.0,
    }

    return sweep, pair_df, confidence, cert
