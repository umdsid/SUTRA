from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr

DP_COLUMNS=("delta_pressure_z","delta_p_z","pressure_difference_z")

def resolve_dp_column(df):
    for c in DP_COLUMNS:
        if c in df.columns:
            return c
    raise RuntimeError(f"no signed delta-p column found; expected {DP_COLUMNS}")

def reconstruct_pressure_residuals(edge_rel,n_cells):
    dcol=resolve_dp_column(edge_rel)
    i=edge_rel.cell_i_index.to_numpy(np.int64)
    j=edge_rel.cell_j_index.to_numpy(np.int64)
    dp=edge_rel[dcol].to_numpy(np.float64)
    valid=(edge_rel.mechanics_delta_p_valid.astype(bool).to_numpy()
           if "mechanics_delta_p_valid" in edge_rel.columns
           else np.isfinite(dp))
    q=valid & np.isfinite(dp)
    ii=i[q]; jj=j[q]; yy=dp[q]
    rows=np.arange(len(ii),dtype=np.int64)
    B=sparse.coo_matrix(
        (
            np.concatenate([np.ones(len(rows)), -np.ones(len(rows))]),
            (
                np.concatenate([rows,rows]),
                np.concatenate([ii,jj]),
            ),
        ),
        shape=(len(rows),n_cells),
    ).tocsr()
    sol=lsqr(B,yy,atol=1e-10,btol=1e-10,
             iter_lim=max(1000,min(20000,5*n_cells)),show=False)
    p=sol[0].astype(np.float64)
    resid=np.asarray(B@p-yy,dtype=np.float64)
    source_index=np.flatnonzero(q)
    E=pd.DataFrame({
        "source_edge_row":source_index.astype(np.int64),
        "cell_i_index":ii,
        "cell_j_index":jj,
        "delta_p_observed":yy,
        "pressure_i":p[ii],
        "pressure_j":p[jj],
        "delta_p_reconstructed":p[ii]-p[jj],
        "pressure_residual":resid,
        "abs_pressure_residual":np.abs(resid),
    })
    return p,E,{
        "lsqr_istop":int(sol[1]),
        "lsqr_iterations":int(sol[2]),
        "lsqr_r1norm":float(sol[3]),
        "lsqr_r2norm":float(sol[4]),
        "lsqr_anorm":float(sol[5]),
        "lsqr_acond":float(sol[6]),
        "lsqr_arnorm":float(sol[7]),
        "lsqr_xnorm":float(sol[8]),
    }

def robust_tail_threshold(absr):
    x=np.asarray(absr,dtype=np.float64)
    x=x[np.isfinite(x)]
    med=float(np.median(x))
    q95=float(np.quantile(x,.95))
    q99=float(np.quantile(x,.99))
    q999=float(np.quantile(x,.999))
    spread=max(q95-med,1e-15)
    threshold=max(q999,med+50.0*spread)
    return {
        "median_abs":med,
        "q95_abs":q95,
        "q99_abs":q99,
        "q999_abs":q999,
        "max_abs":float(np.max(x)),
        "mean_abs":float(np.mean(x)),
        "rms":float(np.sqrt(np.mean(x*x))),
        "robust_tail_threshold":float(threshold),
    }

def attach_tail_flags(E,tail):
    out=E.copy()
    out["extreme_tail"]=out.abs_pressure_residual>float(tail["robust_tail_threshold"])
    out["above_q99"]=out.abs_pressure_residual>float(tail["q99_abs"])
    out["above_q999"]=out.abs_pressure_residual>float(tail["q999_abs"])
    return out

def pressure_components(E,n_cells):
    if len(E)==0:
        return np.full(n_cells,-1,dtype=np.int64),pd.DataFrame()
    A=sparse.coo_matrix(
        (
            np.ones(2*len(E),dtype=np.int8),
            (
                np.concatenate([E.cell_i_index.to_numpy(np.int64),E.cell_j_index.to_numpy(np.int64)]),
                np.concatenate([E.cell_j_index.to_numpy(np.int64),E.cell_i_index.to_numpy(np.int64)]),
            ),
        ),
        shape=(n_cells,n_cells),
    ).tocsr()
    ncomp,labels=connected_components(A,directed=False,return_labels=True)
    x=E.copy()
    x["component"]=labels[x.cell_i_index.to_numpy(np.int64)]
    rows=[]
    for comp,g in x.groupby("component",sort=False):
        nodes=np.unique(np.concatenate([g.cell_i_index.to_numpy(np.int64),g.cell_j_index.to_numpy(np.int64)]))
        absr=g.abs_pressure_residual.to_numpy(np.float64)
        rows.append({
            "component":int(comp),
            "n_nodes":int(len(nodes)),
            "n_edges":int(len(g)),
            "n_extreme_tail":int(g.extreme_tail.sum()) if "extreme_tail" in g else 0,
            "extreme_edge_fraction":float(g.extreme_tail.mean()) if "extreme_tail" in g else 0.0,
            "residual_median_abs":float(np.median(absr)),
            "residual_q95_abs":float(np.quantile(absr,.95)),
            "residual_q99_abs":float(np.quantile(absr,.99)),
            "residual_max_abs":float(np.max(absr)),
            "residual_rms":float(np.sqrt(np.mean(absr*absr))),
        })
    return labels,pd.DataFrame(rows)

def level1_selection_enrichment(E,boundary_level1):
    if boundary_level1 is None or len(boundary_level1)==0:
        return {
            "n_level1_selected_pairs":0,
            "n_pressure_edges_in_selected_pairs":0,
            "selected_edge_extreme_fraction":np.nan,
            "all_edge_extreme_fraction":float(E.extreme_tail.mean()) if len(E) else np.nan,
            "extreme_enrichment_ratio":np.nan,
        }
    selected=boundary_level1[boundary_level1.selected_geometry.astype(bool)].copy()
    keys=set((min(int(a),int(b)),max(int(a),int(b)))
             for a,b in zip(selected.super_i,selected.super_j))
    edge_keys=[(min(int(a),int(b)),max(int(a),int(b)))
               for a,b in zip(E.cell_i_index,E.cell_j_index)]
    mask=np.array([k in keys for k in edge_keys],dtype=bool)
    Es=E[mask]
    all_frac=float(E.extreme_tail.mean()) if len(E) else np.nan
    sel_frac=float(Es.extreme_tail.mean()) if len(Es) else np.nan
    enrich=(float(sel_frac/all_frac)
            if np.isfinite(sel_frac) and np.isfinite(all_frac) and all_frac>0
            else np.nan)
    return {
        "n_level1_selected_pairs":int(len(selected)),
        "n_pressure_edges_in_selected_pairs":int(len(Es)),
        "selected_edge_extreme_fraction":sel_frac,
        "all_edge_extreme_fraction":all_frac,
        "extreme_enrichment_ratio":enrich,
    }

def summarize_tail(E,tail,components,enrichment):
    n_ext=int(E.extreme_tail.sum())
    n_q999=int(E.above_q999.sum())
    finite=bool(np.isfinite(E.pressure_residual.to_numpy(np.float64)).all())
    rms_to_q999=float(tail["rms"]/(tail["q999_abs"]+1e-300))
    max_to_q999=float(tail["max_abs"]/(tail["q999_abs"]+1e-300))
    comp_ext=components[components.n_extreme_tail>0] if len(components) else components
    return {
        **tail,
        "n_constraints":int(len(E)),
        "n_above_q999":n_q999,
        "n_extreme_tail":n_ext,
        "extreme_tail_fraction":float(n_ext/len(E)) if len(E) else 0.0,
        "n_components":int(len(components)),
        "n_components_with_extreme_tail":int(len(comp_ext)),
        "rms_to_q999_ratio":rms_to_q999,
        "max_to_q999_ratio":max_to_q999,
        "all_residuals_finite":finite,
        "sparse_tail_dominates_rms":bool(
            finite and max_to_q999>100.0 and n_ext/max(len(E),1)<0.01
        ),
        "level1_selection_enrichment":enrichment,
    }
