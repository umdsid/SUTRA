from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra

from sutra.hierarchy.v092.sweep import (
    aggregate_completed_edges, completed_pair_cost, greedy_disjoint, apply_batch
)

def schedule_fraction(removed,cfg):
    """Validated ultraslow start; larger batches only after substantial reduction."""
    r=float(removed)
    for stage in cfg["adaptive_schedule"]:
        if r < float(stage["until_removed_fraction"]):
            return float(stage["fraction"]),str(stage["name"])
    z=cfg["adaptive_schedule"][-1]
    return float(z["fraction"]),str(z["name"])

def weighted_supernode_xy(xy,labels,node_ids):
    lab=np.asarray(labels,np.int64)
    xy=np.asarray(xy,float)
    out=np.zeros((len(node_ids),2),float)
    sizes=np.zeros(len(node_ids),np.int64)
    pos={int(u):k for k,u in enumerate(node_ids)}
    for u in node_ids:
        idx=np.flatnonzero(lab==u)
        k=pos[int(u)]
        sizes[k]=len(idx)
        out[k]=xy[idx].mean(axis=0)
    return out,sizes

def aggregate_expression(Y,labels,node_ids):
    lab=np.asarray(labels,np.int64)
    Y=np.asarray(Y,np.float32)
    out=np.empty((len(node_ids),Y.shape[1]),np.float32)
    for k,u in enumerate(node_ids):
        out[k]=Y[lab==u].mean(axis=0)
    return out

def topology_stats(superedges,node_ids):
    n=len(node_ids)
    if n==0:
        return dict(components=0,cycle_rank=0,mean_degree=0.,max_degree=0)
    pos={int(u):k for k,u in enumerate(node_ids)}
    if len(superedges)==0:
        return dict(components=n,cycle_rank=0,mean_degree=0.,max_degree=0)
    i=np.array([pos[int(x)] for x in superedges.super_i],np.int64)
    j=np.array([pos[int(x)] for x in superedges.super_j],np.int64)
    A=sparse.coo_matrix(
        (np.ones(2*len(i)),(np.r_[i,j],np.r_[j,i])),
        shape=(n,n)
    ).tocsr()
    C,_=connected_components(A,directed=False)
    deg=np.asarray((A>0).sum(axis=1)).ravel()
    E=int(len(superedges))
    return dict(
        components=int(C),
        cycle_rank=int(E-n+C),
        mean_degree=float(deg.mean()),
        max_degree=int(deg.max()) if len(deg) else 0,
    )

def summarize_array(a,prefix):
    x=np.asarray(a,float)
    x=x[np.isfinite(x)]
    if len(x)==0:return {}
    q=np.quantile(x,[.01,.05,.25,.5,.75,.95,.99])
    return {
        f"{prefix}_mean":float(x.mean()),
        f"{prefix}_sd":float(x.std()),
        f"{prefix}_q01":float(q[0]),
        f"{prefix}_q05":float(q[1]),
        f"{prefix}_q25":float(q[2]),
        f"{prefix}_q50":float(q[3]),
        f"{prefix}_q75":float(q[4]),
        f"{prefix}_q95":float(q[5]),
        f"{prefix}_q99":float(q[6]),
        f"{prefix}_max":float(x.max()),
    }

def shannon_sizes(sizes):
    x=np.asarray(sizes,float)
    p=x/x.sum()
    return float(-(p*np.log(np.maximum(p,1e-300))).sum())

def derivative_table(df,xcol="removed_fraction"):
    """First and second finite differences against cumulative reduction."""
    x=df[xcol].to_numpy(float)
    out=df.copy()
    numeric=[c for c in df.columns if c!=xcol and pd.api.types.is_numeric_dtype(df[c])]
    for c in numeric:
        y=df[c].to_numpy(float)
        if len(y)<2 or np.all(~np.isfinite(y)):
            continue
        yy=pd.Series(y).interpolate(limit_direction="both").to_numpy(float)
        dx=np.gradient(x)
        dx=np.where(np.abs(dx)>1e-12,dx,np.nan)
        d1=np.gradient(yy)/dx
        d2=np.gradient(d1)/dx
        out[f"d1_{c}"]=d1
        out[f"d2_{c}"]=d2
    return out

def current_transport_registry(Q,superedges):
    from sutra.hierarchy.v077.transport import classify_pair
    rows=[]
    for r in superedges.itertuples(index=False):
        u=int(r.super_i); v=int(r.super_j)
        info=classify_pair(Q[u],Q[v])
        rows.append((u,v,info["transport_status"]))
    return pd.DataFrame(rows,columns=["super_i","super_j","transport_status"])

def directed_path_stats(superedges,node_ids,sample_sources=24,seed=177):
    """Directed path geometry from current completed cost + signaling direction."""
    if len(node_ids)<2 or len(superedges)==0:
        return {}
    pos={int(u):k for k,u in enumerate(node_ids)}
    i=np.array([pos[int(x)] for x in superedges.super_i],np.int64)
    j=np.array([pos[int(x)] for x in superedges.super_j],np.int64)
    base=np.maximum(superedges.completed_pair_cost.to_numpy(float),1e-9)
    d=superedges.cellchat_directionality.to_numpy(float)
    f=base*np.exp(np.clip(d,-4,4))
    r=base*np.exp(np.clip(-d,-4,4))
    A=sparse.coo_matrix(
        (np.r_[f,r],(np.r_[i,j],np.r_[j,i])),
        shape=(len(node_ids),len(node_ids))
    ).tocsr()
    rng=np.random.default_rng(seed)
    ns=min(sample_sources,len(node_ids))
    src=np.sort(rng.choice(len(node_ids),size=ns,replace=False))
    D=dijkstra(A,directed=True,indices=src)
    ratios=[]
    vals=[]
    for a,s in enumerate(src):
        finite=np.isfinite(D[a])&(D[a]>0)
        vals.extend(D[a,finite].tolist())
        targets=np.flatnonzero(finite)
        for t in targets[:256]:
            back=dijkstra(A,directed=True,indices=int(t),limit=np.inf)[s]
            if np.isfinite(back) and back>0:
                ratios.append(max(D[a,t]/back,back/D[a,t]))
    out={}
    out.update(summarize_array(vals,"geodesic_length"))
    out.update(summarize_array(ratios,"geodesic_reversal_ratio"))
    return out

def enumerate_triangles_simple(superedges,limit=5000):
    adj={}
    for r in superedges.itertuples(index=False):
        u=int(r.super_i);v=int(r.super_j)
        adj.setdefault(u,set()).add(v);adj.setdefault(v,set()).add(u)
    out=[]
    for u in sorted(adj):
        for v in sorted(x for x in adj[u] if x>u):
            for w in sorted(x for x in adj[u].intersection(adj[v]) if x>v):
                out.append((u,v,w))
                if len(out)>=limit:return out
    return out

def polygon_area(P):
    P=np.asarray(P,float)
    x=P[:,0]; y=P[:,1]
    return .5*abs(float(np.dot(x,np.roll(y,-1))-np.dot(y,np.roll(x,-1))))

def heavy_geometry_snapshot(superedges,node_ids,xy_node,Bnode,G,triangle_limit=5000):
    """Transport/holonomy snapshot on current supernodes."""
    from sutra.hierarchy.v077.transport import whiten_covectors
    from sutra.hierarchy.v078.holonomy import (
        compose_loop_holonomy,holonomy_metrics
    )

    H,Hinv,Qarr,rho=whiten_covectors(G,Bnode)
    # Qarr rows follow node_ids; map to stable supernode IDs for loop utilities.
    qmap={int(u):Qarr[k] for k,u in enumerate(node_ids)}
    d=max(int(max(node_ids))+1 if len(node_ids) else 0,1)
    Q=np.zeros((d,Qarr.shape[1]),float)
    for u,q in qmap.items(): Q[u]=q

    out={}
    out.update(summarize_array(rho,"directional_dual_norm"))

    reg=current_transport_registry(Q,superedges)
    vc=reg.transport_status.value_counts()
    n=max(len(reg),1)
    out["transport_rotate_fraction"]=float(vc.get("DIRECTIONAL_ROTATION",0)/n)
    out["transport_identity_fraction"]=float(vc.get("IDENTITY_SYMMETRIC",0)/n)
    out["transport_unresolved_fraction"]=float(1-(vc.get("DIRECTIONAL_ROTATION",0)+vc.get("IDENTITY_SYMMETRIC",0))/n)

    out.update(directed_path_stats(superedges,node_ids))

    triangles=enumerate_triangles_simple(superedges,triangle_limit)
    pos={int(u):k for k,u in enumerate(node_ids)}
    angles=[];dens=[];resolved=0
    for loop in triangles:
        statuses=[]
        ok=True
        for a,b in [(loop[0],loop[1]),(loop[1],loop[2]),(loop[2],loop[0])]:
            s=reg[((reg.super_i==min(a,b))&(reg.super_j==max(a,b)))].transport_status
            if len(s)==0 or s.iloc[0] not in ("DIRECTIONAL_ROTATION","IDENTITY_SYMMETRIC"):
                ok=False; break
        if not ok: continue
        resolved+=1
        Hloop,_=compose_loop_holonomy(Q,loop)
        m=holonomy_metrics(Hloop)
        ang=float(m["spectral_angle_rms"])
        area=polygon_area(np.array([xy_node[pos[u]] for u in loop]))
        angles.append(ang)
        if area>1e-12: dens.append(ang/area)

    out["triangle_candidates"]=int(len(triangles))
    out["triangle_resolved"]=int(resolved)
    out.update(summarize_array(angles,"holonomy_angle_rms"))
    out.update(summarize_array(dens,"holonomy_density"))
    return out,reg
