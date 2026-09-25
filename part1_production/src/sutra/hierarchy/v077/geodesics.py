"""Directed graph geodesics for STRATA v0.7.7."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components


def build_directed_cost_graph(level1_edges,n_nodes):
    e=level1_edges[
        level1_edges.geometry_state_resolved.astype(bool)
        & np.isfinite(level1_edges.local_forward_cost)
        & np.isfinite(level1_edges.local_reverse_cost)
        & (level1_edges.local_forward_cost>=0)
        & (level1_edges.local_reverse_cost>=0)
    ].copy()

    i=e.super_i.to_numpy(np.int64)
    j=e.super_j.to_numpy(np.int64)
    fij=e.local_forward_cost.to_numpy(np.float64)
    fji=e.local_reverse_cost.to_numpy(np.float64)

    A=sparse.coo_matrix(
        (
            np.concatenate([fij,fji]),
            (
                np.concatenate([i,j]),
                np.concatenate([j,i]),
            ),
        ),
        shape=(n_nodes,n_nodes),
    ).tocsr()
    A.sum_duplicates()
    return A,e


def graph_component_summary(A):
    U=((A+A.T)>0).astype(np.int8)
    ncomp,labels=connected_components(U,directed=False,return_labels=True)
    counts=np.bincount(labels,minlength=ncomp)
    return labels,{
        "n_components":int(ncomp),
        "largest_component_size":int(counts.max()) if len(counts) else 0,
        "largest_component_fraction":float(counts.max()/A.shape[0]) if A.shape[0] else 0.0,
    }


def choose_landmarks(rho,component_labels,max_sources=16):
    rho=np.asarray(rho,dtype=np.float64)
    labels=np.asarray(component_labels,dtype=np.int64)
    if len(labels)==0:
        return np.array([],dtype=np.int64)

    counts=np.bincount(labels)
    largest=int(np.argmax(counts))
    q=np.flatnonzero(labels==largest)
    order=q[np.argsort(rho[q],kind="mergesort")[::-1]]
    return order[:min(max_sources,len(order))].astype(np.int64)


def landmark_geodesics(A,sources):
    if len(sources)==0:
        return pd.DataFrame(),{}

    D=dijkstra(
        A,directed=True,
        indices=np.asarray(sources,dtype=np.int64),
        return_predecessors=False,
    )

    rows=[]
    for a,src in enumerate(sources):
        for b,dst in enumerate(sources):
            if a==b: continue
            dij=float(D[a,dst])
            dji=float(D[b,src])
            if np.isfinite(dij) and np.isfinite(dji) and dij>0 and dji>0:
                rows.append({
                    "source":int(src),
                    "target":int(dst),
                    "distance_forward":dij,
                    "distance_reverse":dji,
                    "distance_asymmetry_ratio":max(dij/dji,dji/dij),
                    "signed_log_distance_ratio":float(np.log(dij/dji)),
                })

    out=pd.DataFrame(rows)
    summary={
        "n_landmarks":int(len(sources)),
        "n_finite_ordered_pairs":int(len(out)),
        "distance_asymmetry_ratio_median":
            float(out.distance_asymmetry_ratio.median()) if len(out) else np.nan,
        "distance_asymmetry_ratio_q90":
            float(out.distance_asymmetry_ratio.quantile(.90)) if len(out) else np.nan,
        "distance_asymmetry_ratio_q95":
            float(out.distance_asymmetry_ratio.quantile(.95)) if len(out) else np.nan,
        "distance_asymmetry_ratio_max":
            float(out.distance_asymmetry_ratio.max()) if len(out) else np.nan,
    }
    return out,summary


def triangle_audit(A,sources,max_triplets=256):
    if len(sources)<3:
        return {"n_triangle_checks":0,"max_violation":0.0,"pass":True}

    D=dijkstra(
        A,directed=True,
        indices=np.asarray(sources,dtype=np.int64),
        return_predecessors=False,
    )
    checks=0
    worst=0.0
    n=len(sources)
    for i in range(n):
        for j in range(n):
            if j==i: continue
            for k in range(n):
                if k==i or k==j: continue
                dij=D[i,sources[j]]
                djk=D[j,sources[k]]
                dik=D[i,sources[k]]
                if np.isfinite(dij) and np.isfinite(djk) and np.isfinite(dik):
                    violation=float(dik-(dij+djk))
                    worst=max(worst,violation)
                    checks+=1
                    if checks>=max_triplets:
                        return {
                            "n_triangle_checks":checks,
                            "max_violation":worst,
                            "pass":bool(worst<=1e-8),
                        }
    return {
        "n_triangle_checks":checks,
        "max_violation":worst,
        "pass":bool(worst<=1e-8),
    }
