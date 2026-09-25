"""STRATA v0.7.8 transport holonomy.

Holonomy is computed only from the discrete connection certified in v0.7.7.
No continuum connection or curvature tensor is assumed.

For a resolved closed loop gamma=(v0,...,vk-1,v0), compose the whitened-space
minimal rotations

    H_gamma = R_{v_{k-1},v0} ... R_{v1,v2} R_{v0,v1}.

Each edge rotation differs from identity only on the span of its endpoint
directional frames. Therefore the complete nontrivial loop action lies in the
span of the q-vectors on that loop, dimension <= loop length. We compute
holonomy in that reduced span instead of materializing 543x543 matrices.

Identity-symmetric edges contribute the identity. One-sided and antipodal
edges make the loop unresolved and are never silently repaired.
"""

from __future__ import annotations

from itertools import combinations
import numpy as np
import pandas as pd

from strata_hierarchy.v077.transport import (
    classify_pair,
    rotate_minimal,
)


RESOLVED_STATUSES={"IDENTITY_SYMMETRIC","DIRECTIONAL_ROTATION"}


def canonical_cycle(nodes):
    x=tuple(int(v) for v in nodes)
    n=len(x)
    rots=[x[k:]+x[:k] for k in range(n)]
    xr=tuple(reversed(x))
    rots += [xr[k:]+xr[:k] for k in range(n)]
    return min(rots)


def edge_key(a,b):
    a=int(a); b=int(b)
    return (a,b) if a<b else (b,a)


def registry_lookup(reg):
    out={}
    for r in reg.itertuples(index=False):
        out[edge_key(r.super_i,r.super_j)]=str(r.transport_status)
    return out


def adjacency_from_registry(reg):
    adj={}
    for r in reg.itertuples(index=False):
        u=int(r.super_i); v=int(r.super_j)
        adj.setdefault(u,set()).add(v)
        adj.setdefault(v,set()).add(u)
    return adj


def enumerate_triangles(adj):
    """Enumerate every undirected triangle once."""
    tris=[]
    for u in sorted(adj):
        nu={v for v in adj[u] if v>u}
        for v in sorted(nu):
            common=nu.intersection(w for w in adj.get(v,set()) if w>v)
            for w in sorted(common):
                tris.append((u,v,w))
    return tris


def enumerate_chordless_quads(adj,max_quads=20000):
    """Enumerate deterministic chordless 4-cycles, capped for safety.

    Opposite vertices u,w share two intermediate neighbors v,x. Chordless means
    u--w and v--x are absent.
    """
    pair_to_middle={}
    for m in sorted(adj):
        ns=sorted(adj[m])
        for u,w in combinations(ns,2):
            if u==w:
                continue
            a,b=(u,w) if u<w else (w,u)
            pair_to_middle.setdefault((a,b),[]).append(m)

    seen=set()
    out=[]
    for (u,w),mids in sorted(pair_to_middle.items()):
        if w in adj.get(u,set()):
            continue  # diagonal u--w exists
        mids=sorted(set(mids))
        if len(mids)<2:
            continue
        for v,x in combinations(mids,2):
            if x in adj.get(v,set()):
                continue  # other diagonal exists
            cyc=canonical_cycle((u,v,w,x))
            if cyc in seen:
                continue
            seen.add(cyc)
            out.append(cyc)
            if len(out)>=max_quads:
                return out
    return out


def loop_status(loop,lookup):
    statuses=[]
    n=len(loop)
    for k in range(n):
        u=loop[k]
        v=loop[(k+1)%n]
        s=lookup.get(edge_key(u,v))
        if s is None:
            return {
                "fully_resolved":False,
                "missing_edge":True,
                "n_directional_edges":0,
                "n_identity_edges":0,
                "n_unresolved_edges":1,
                "edge_statuses":tuple(statuses+["MISSING"]),
            }
        statuses.append(s)

    nrot=sum(s=="DIRECTIONAL_ROTATION" for s in statuses)
    nid=sum(s=="IDENTITY_SYMMETRIC" for s in statuses)
    nun=sum(s not in RESOLVED_STATUSES for s in statuses)
    return {
        "fully_resolved":bool(nun==0),
        "missing_edge":False,
        "n_directional_edges":int(nrot),
        "n_identity_edges":int(nid),
        "n_unresolved_edges":int(nun),
        "edge_statuses":tuple(statuses),
    }


def loop_basis(Q,loop,eps=1e-12):
    """Orthonormal basis for span of nonzero directional frames on the loop."""
    V=[]
    for u in loop:
        q=np.asarray(Q[int(u)],dtype=np.float64)
        n=float(np.linalg.norm(q))
        if n>eps:
            V.append(q/n)
    if not V:
        return np.zeros((Q.shape[1],0),dtype=np.float64)

    A=np.stack(V,axis=1)
    U,s,_=np.linalg.svd(A,full_matrices=False)
    rank=int(np.sum(s>max(eps,s[0]*1e-12)))
    return U[:,:rank]


def apply_edge_rotation(qi,qj,x):
    info=classify_pair(qi,qj)
    s=info["transport_status"]
    if s=="IDENTITY_SYMMETRIC":
        return np.asarray(x,dtype=np.float64).copy()
    if s!="DIRECTIONAL_ROTATION":
        raise ValueError(f"unresolved edge transport: {s}")

    ai=np.asarray(qi,dtype=np.float64)/info["rho_i"]
    aj=np.asarray(qj,dtype=np.float64)/info["rho_j"]
    return rotate_minimal(ai,aj,np.asarray(x,dtype=np.float64))


def reduced_edge_matrix(Q,u,v,U):
    m=U.shape[1]
    if m==0:
        return np.eye(0,dtype=np.float64)

    M=np.empty((m,m),dtype=np.float64)
    for k in range(m):
        y=apply_edge_rotation(Q[int(u)],Q[int(v)],U[:,k])
        M[:,k]=U.T@y
    return M


def compose_loop_holonomy(Q,loop,U=None):
    """Compose loop transport in one explicitly chosen reduced basis.

    If U is omitted, construct the canonical numerical basis for this loop.
    If U is supplied, every edge operator is represented in that same basis.
    This is essential when comparing a loop with its reversal: independently
    recomputed SVD bases span the same subspace but need not have identical
    coordinates.
    """
    if U is None:
        U=loop_basis(Q,loop)
    else:
        U=np.asarray(U,dtype=np.float64)

    m=U.shape[1]

    if m==0:
        return np.eye(0,dtype=np.float64),U

    H=np.eye(m,dtype=np.float64)
    n=len(loop)
    for k in range(n):
        u=loop[k]; v=loop[(k+1)%n]
        R=reduced_edge_matrix(Q,u,v,U)
        H=R@H
    return H,U


def holonomy_metrics(H):
    m=H.shape[0]
    if m==0:
        return {
            "active_dimension":0,
            "orthogonality_error":0.0,
            "determinant":1.0,
            "identity_deviation_fro":0.0,
            "identity_deviation_normalized":0.0,
            "spectral_angle_rms":0.0,
            "spectral_angle_max":0.0,
        }

    I=np.eye(m)
    orth=float(np.linalg.norm(H.T@H-I,ord="fro"))
    det=float(np.linalg.det(H))
    dev=float(np.linalg.norm(H-I,ord="fro"))
    eig=np.linalg.eigvals(H)
    ang=np.abs(np.angle(eig))
    return {
        "active_dimension":int(m),
        "orthogonality_error":orth,
        "determinant":det,
        "identity_deviation_fro":dev,
        "identity_deviation_normalized":float(dev/np.sqrt(m)),
        "spectral_angle_rms":float(np.sqrt(np.mean(ang*ang))),
        "spectral_angle_max":float(np.max(ang)),
    }



def reverse_loop_same_basepoint(loop):
    """Return the inverse closed path while preserving the loop basepoint.

    For loop (v0,v1,...,vk-1), the inverse based at v0 is

        (v0,vk-1,...,v1).

    A plain tuple reversal changes the basepoint to vk-1 and therefore gives
    a conjugate holonomy rather than the inverse matrix at the original
    tangent space.
    """
    loop=tuple(loop)
    if len(loop)<=1:
        return loop
    return (loop[0],)+tuple(reversed(loop[1:]))


def reverse_inverse_error(Q,loop):
    """Compare a based loop with its inverse at the same basepoint and basis."""
    H,U=compose_loop_holonomy(Q,loop)
    inv_loop=reverse_loop_same_basepoint(loop)
    Hr,_=compose_loop_holonomy(Q,inv_loop,U=U)
    m=H.shape[0]
    if m==0:
        return 0.0
    return float(np.linalg.norm(Hr@H-np.eye(m),ord="fro"))


def coordinate_invariance_check(Q,loop,seed=0):
    """Holonomy spectral angles are invariant under orthogonal relabeling."""
    d=Q.shape[1]
    rng=np.random.default_rng(seed)
    A=rng.normal(size=(d,min(d,8)))
    # Full ambient random orthogonal matrix is wasteful at d=543. Build an
    # orthogonal reflection instead: P=I-2uu^T.
    u=A[:,0]
    u/=np.linalg.norm(u)
    Qp=Q-2.0*(Q@u)[:,None]*u[None,:]

    H,_=compose_loop_holonomy(Q,loop)
    Hp,_=compose_loop_holonomy(Qp,loop)

    a=holonomy_metrics(H)
    b=holonomy_metrics(Hp)
    return float(max(
        abs(a["spectral_angle_rms"]-b["spectral_angle_rms"]),
        abs(a["spectral_angle_max"]-b["spectral_angle_max"]),
        abs(a["identity_deviation_normalized"]-b["identity_deviation_normalized"]),
    ))


def evaluate_loops(Q,loops,lookup,loop_type,coord_checks=32):
    rows=[]
    ncoord=0

    for idx,loop in enumerate(loops):
        st=loop_status(loop,lookup)
        row={
            "loop_type":loop_type,
            "loop_index":int(idx),
            "loop_nodes":";".join(map(str,loop)),
            "loop_length":int(len(loop)),
            **{k:v for k,v in st.items() if k!="edge_statuses"},
            "edge_statuses":";".join(st["edge_statuses"]),
        }

        if not st["fully_resolved"]:
            row.update({
                "active_dimension":np.nan,
                "orthogonality_error":np.nan,
                "determinant":np.nan,
                "identity_deviation_fro":np.nan,
                "identity_deviation_normalized":np.nan,
                "spectral_angle_rms":np.nan,
                "spectral_angle_max":np.nan,
                "reverse_inverse_error":np.nan,
                "coordinate_invariance_error":np.nan,
            })
            rows.append(row)
            continue

        H,U=compose_loop_holonomy(Q,loop)
        met=holonomy_metrics(H)
        rev=reverse_inverse_error(Q,loop)

        coord=np.nan
        if ncoord<coord_checks and st["n_directional_edges"]>0:
            coord=coordinate_invariance_check(Q,loop,seed=177+idx)
            ncoord+=1

        row.update({
            **met,
            "reverse_inverse_error":rev,
            "coordinate_invariance_error":coord,
        })
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_loop_table(df):
    if len(df)==0:
        return {
            "n_candidate_loops":0,
            "n_resolved_loops":0,
            "resolved_fraction":0.0,
            "n_directional_resolved_loops":0,
            "directional_resolved_fraction":0.0,
        }

    q=df[df.fully_resolved.astype(bool)]
    d=q[q.n_directional_edges>0]

    out={
        "n_candidate_loops":int(len(df)),
        "n_resolved_loops":int(len(q)),
        "resolved_fraction":float(len(q)/len(df)),
        "n_directional_resolved_loops":int(len(d)),
        "directional_resolved_fraction":float(len(d)/len(q)) if len(q) else 0.0,
        "n_identity_only_resolved_loops":int((q.n_directional_edges==0).sum()),
    }

    if len(d):
        out.update({
            "holonomy_angle_rms_median":float(d.spectral_angle_rms.median()),
            "holonomy_angle_rms_q90":float(d.spectral_angle_rms.quantile(.90)),
            "holonomy_angle_rms_q95":float(d.spectral_angle_rms.quantile(.95)),
            "holonomy_angle_max_q95":float(d.spectral_angle_max.quantile(.95)),
            "holonomy_angle_max_max":float(d.spectral_angle_max.max()),
            "identity_deviation_q95":float(
                d.identity_deviation_normalized.quantile(.95)
            ),
            "orthogonality_error_max":float(d.orthogonality_error.max()),
            "reverse_inverse_error_max":float(d.reverse_inverse_error.max()),
            "coordinate_invariance_error_max":float(
                d.coordinate_invariance_error.dropna().max()
            ) if d.coordinate_invariance_error.notna().any() else 0.0,
        })
    else:
        out.update({
            "holonomy_angle_rms_median":0.0,
            "holonomy_angle_rms_q90":0.0,
            "holonomy_angle_rms_q95":0.0,
            "holonomy_angle_max_q95":0.0,
            "holonomy_angle_max_max":0.0,
            "identity_deviation_q95":0.0,
            "orthogonality_error_max":0.0,
            "reverse_inverse_error_max":0.0,
            "coordinate_invariance_error_max":0.0,
        })

    return out


def certify_loop_table(df,orth_tol=1e-10,reverse_tol=1e-10,coord_tol=1e-9):
    if len(df)==0:
        return {
            "has_loops":False,
            "identity_loops_zero":True,
            "orthogonality_pass":True,
            "reverse_inverse_pass":True,
            "coordinate_invariance_pass":True,
            "certificate_pass":True,
        }

    q=df[df.fully_resolved.astype(bool)]
    ident=q[q.n_directional_edges==0]
    directional=q[q.n_directional_edges>0]

    identity_zero=bool(
        len(ident)==0
        or (
            (ident.identity_deviation_normalized.fillna(0)<=1e-12).all()
            and (ident.spectral_angle_max.fillna(0)<=1e-12).all()
        )
    )
    orth=bool(
        len(directional)==0
        or directional.orthogonality_error.max()<=orth_tol
    )
    rev=bool(
        len(directional)==0
        or directional.reverse_inverse_error.max()<=reverse_tol
    )
    c=directional.coordinate_invariance_error.dropna()
    coord=bool(len(c)==0 or c.max()<=coord_tol)

    return {
        "has_loops":bool(len(df)>0),
        "identity_loops_zero":identity_zero,
        "orthogonality_pass":orth,
        "reverse_inverse_pass":rev,
        "coordinate_invariance_pass":coord,
        "certificate_pass":bool(identity_zero and orth and rev and coord),
    }
