from __future__ import annotations
from dataclasses import dataclass
from collections import deque
import numpy as np
import pandas as pd
from scipy import linalg

@dataclass(frozen=True)
class ObservabilityConfig:
    dense_nvar_cap: int = 450
    dense_nullity_cap: int = 40
    numerical_rel_tol: float = 1e-9

def hopcroft_karp_columns(A):
    C=A.tocsc()
    nrow,ncol=A.shape
    adj=[C.indices[C.indptr[j]:C.indptr[j+1]].tolist() for j in range(ncol)]
    NIL=-1
    pair_u=np.full(ncol,NIL,dtype=int)
    pair_v=np.full(nrow,NIL,dtype=int)
    dist=np.zeros(ncol,dtype=int)

    def bfs():
        q=deque();found=False
        for u in range(ncol):
            if pair_u[u]==NIL:
                dist[u]=0;q.append(u)
            else:
                dist[u]=-1
        while q:
            u=q.popleft()
            for v in adj[u]:
                u2=pair_v[v]
                if u2==NIL:
                    found=True
                elif dist[u2]<0:
                    dist[u2]=dist[u]+1
                    q.append(u2)
        return found

    def dfs(u):
        for v in adj[u]:
            u2=pair_v[v]
            if u2==NIL or (dist[u2]==dist[u]+1 and dfs(u2)):
                pair_u[u]=v;pair_v[v]=u
                return True
        dist[u]=-1
        return False

    while bfs():
        for u in range(ncol):
            if pair_u[u]==NIL:
                dfs(u)
    return pair_u,pair_v,adj

def structural_unresolved_columns(A):
    col_to_row,row_to_col,adj=hopcroft_karp_columns(A)
    unmatched=np.flatnonzero(col_to_row<0)
    seen_c=set(int(x) for x in unmatched)
    seen_r=set()
    q=deque(("c",int(x)) for x in unmatched)
    while q:
        side,node=q.popleft()
        if side=="c":
            matched_row=col_to_row[node]
            for r in adj[node]:
                if r==matched_row:
                    continue
                if r not in seen_r:
                    seen_r.add(r);q.append(("r",r))
        else:
            c=int(row_to_col[node])
            if c>=0 and c not in seen_c:
                seen_c.add(c);q.append(("c",c))
    unresolved=np.zeros(A.shape[1],dtype=bool)
    if seen_c:
        unresolved[list(seen_c)]=True
    return unresolved,{
        "matching_rank":int(np.sum(col_to_row>=0)),
        "unmatched_variables":int(len(unmatched)),
        "dm_unresolved_variables":int(unresolved.sum()),
    }

def variable_classes(meta):
    n=len(meta["eidx"])+len(meta["cidx"])+len(meta["bidx"])
    classes=np.empty(n,dtype=object)
    ids=np.empty(n,dtype=object)
    for k,i in meta["eidx"].items():
        classes[i]="tension";ids[i]=int(k)
    for k,i in meta["cidx"].items():
        classes[i]="cell_pressure";ids[i]=int(k)
    for k,i in meta["bidx"].items():
        classes[i]="boundary_pressure";ids[i]=int(k)
    return classes,ids

def numerical_nullspace(A,cfg:ObservabilityConfig):
    n=A.shape[1]
    if n>cfg.dense_nvar_cap:
        return None,{"status":"SKIP_NVAR_CAP"}
    Z=linalg.null_space(A.toarray(),rcond=cfg.numerical_rel_tol)
    if Z.shape[1]>cfg.dense_nullity_cap:
        return None,{"status":"SKIP_NULLITY_CAP","numerical_nullity":int(Z.shape[1])}
    return Z,{"status":"PASS","numerical_nullity":int(Z.shape[1])}

def nullspace_row_norms(Z):
    if Z is None:
        return None
    if Z.shape[1]==0:
        return np.zeros(Z.shape[0],float)
    return np.linalg.norm(Z,axis=1)

def interface_pressure_contrasts(meta,E,Z,tol):
    rows=[]
    if Z is None:
        return pd.DataFrame(rows)
    scale=max(float(np.linalg.norm(Z)),1.0)
    thresh=tol*scale
    for r in E.itertuples():
        if r.kind!="cell_cell":
            continue
        a=int(r.cell_i);b=int(r.cell_j)
        if a not in meta["cidx"] or b not in meta["cidx"]:
            continue
        norm=float(np.linalg.norm(
            Z[meta["cidx"][a],:]-Z[meta["cidx"][b],:]
        ))
        rows.append({
            "interface_id":int(r.interface_id),
            "cell_i":a,"cell_j":b,
            "nullspace_contrast_norm":norm,
            "numerically_observable":bool(norm<=thresh),
        })
    return pd.DataFrame(rows)

def patch_observability(A,meta,E,cfg:ObservabilityConfig):
    unresolved,dm=structural_unresolved_columns(A)
    classes,ids=variable_classes(meta)
    rows=[]
    for i,(cl,objid) in enumerate(zip(classes,ids)):
        rows.append({
            "column_index":i,
            "variable_class":str(cl),
            "object_id":int(objid),
            "structurally_observable":bool(not unresolved[i]),
        })
    vars_df=pd.DataFrame(rows)
    Z,nmeta=numerical_nullspace(A,cfg)
    norms=nullspace_row_norms(Z)
    if norms is not None:
        scale=max(float(np.linalg.norm(Z)),1.0)
        thresh=cfg.numerical_rel_tol*scale
        vars_df["nullspace_row_norm"]=norms
        vars_df["numerically_observable"]=norms<=thresh
    else:
        vars_df["nullspace_row_norm"]=np.nan
        vars_df["numerically_observable"]=pd.NA

    contrasts=interface_pressure_contrasts(meta,E,Z,cfg.numerical_rel_tol)
    summary={**dm,**nmeta}
    for cl in ("tension","cell_pressure","boundary_pressure"):
        q=vars_df.variable_class==cl
        summary[f"{cl}_n"]=int(q.sum())
        summary[f"{cl}_structural_observable_fraction"]=(
            float(vars_df.loc[q,"structurally_observable"].mean()) if q.any() else np.nan
        )
        qq=q & vars_df.numerically_observable.notna()
        summary[f"{cl}_numerical_observable_fraction"]=(
            float(vars_df.loc[qq,"numerically_observable"].mean()) if qq.any() else np.nan
        )
    summary["pressure_contrast_n"]=int(len(contrasts))
    summary["pressure_contrast_numerical_observable_fraction"]=(
        float(contrasts.numerically_observable.mean()) if len(contrasts) else np.nan
    )
    return summary,vars_df,contrasts
