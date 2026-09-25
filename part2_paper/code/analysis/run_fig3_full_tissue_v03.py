#!/usr/bin/env python3
"""
SUTRA Figure 3 full-tissue matched relational benchmark v0.3

Scientific design
-----------------
Every method receives the same Level-0 cells and the same measured-gene matrix.

Methods:
  PCA
  Nyström-approximated kernel PCA
  UMAP
  t-SNE

SUTRA is NOT treated as a fifth dimensionality-reduction algorithm. It supplies
the physical candidate-interface graph and the subset of candidate interfaces
that are SUTRA-admissible.

All quantitative metrics are computed on the FULL tissue. Edge thinning is used
ONLY for drawing the full-tissue maps.

Main metrics
------------
1. Physical-interface enrichment:
       P(latent edge is a physical candidate interface)
       ------------------------------------------------
       P(random cell pair is a physical candidate interface)

2. Tissue-constraint concordance:
       P(SUTRA-admissible | latent edge hits physical candidate interface)

The latter is explicitly a concordance with the SUTRA tissue criterion, NOT an
independent ground-truth benchmark.

Scalability note
----------------
Exact RBF kernel PCA is O(N^2) in memory and O(N^3) in eigendecomposition and is
not scientifically reasonable for ~10^5 cells. We therefore use a deterministic
Nyström approximation to the RBF kernel, followed by PCA of the approximate
feature map. The figure label is "kPCA*" and the caption/manifest records the
approximation.
"""

from __future__ import annotations
from pathlib import Path
import argparse, json, math, time
import numpy as np
import pandas as pd
import h5py
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.kernel_approximation import Nystroem
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

try:
    import umap
except Exception:
    umap=None

METHODS=["PCA","kPCA*","UMAP","t-SNE"]
KS=[2,3,4,5,8,10,15,25]
METHOD_COLORS={
    "PCA":"tab:blue",
    "kPCA*":"tab:orange",
    "UMAP":"tab:green",
    "t-SNE":"tab:red",
}

def dec(x):
    return x.decode() if isinstance(x,(bytes,bytearray)) else str(x)

def load10x(path):
    with h5py.File(path,"r") as f:
        g=f["matrix"]
        X=sp.csc_matrix(
            (g["data"][:],g["indices"][:],g["indptr"][:]),
            shape=tuple(g["shape"][:])
        ).T.tocsr()
        barcodes=np.array([dec(x) for x in g["barcodes"][:]],dtype=object)
        genes=np.array([dec(x) for x in g["features"]["name"][:]],dtype=object)
    return X,barcodes,genes

def load_cells(path):
    d=pd.read_parquet(path) if ".parquet" in str(path) else pd.read_csv(path)
    lc={c.lower():c for c in d.columns}
    iid=next((lc[k] for k in ("cell_id","cell","barcode","cellid") if k in lc),None)
    if iid is None:
        raise ValueError(f"Could not identify cell-ID column in {path}")
    xy=None
    for a,b in (
        ("x_centroid","y_centroid"),
        ("centroid_x","centroid_y"),
        ("x","y"),
        ("spatial_x","spatial_y"),
        ("x_location","y_location"),
    ):
        if a in lc and b in lc:
            xy=(lc[a],lc[b]);break
    if xy is None:
        raise ValueError(f"Could not identify x/y columns in {path}")
    out=d[[iid,xy[0],xy[1]]].copy()
    out.columns=["cell_id","x","y"]
    out["cell_id"]=out.cell_id.astype(str)
    out["_production_row"]=np.arange(len(out),dtype=np.int64)
    return out

def align_matrix_cells(X,barcodes,cells):
    pos={str(b):i for i,b in enumerate(barcodes)}
    keep=cells.cell_id.isin(pos)
    cells=cells.loc[keep].copy()
    idx=np.array([pos[c] for c in cells.cell_id],dtype=np.int64)
    return X[idx],cells.reset_index(drop=True)

def normalize_select(X,genes,n_genes=2000):
    lib=np.asarray(X.sum(axis=1)).ravel()
    scale=np.divide(1e4,lib,out=np.zeros_like(lib,dtype=float),where=lib>0)
    X=(sp.diags(scale)@X).tocsr()
    X.data=np.log1p(X.data)

    mu=np.asarray(X.mean(axis=0)).ravel()
    var=np.asarray(X.power(2).mean(axis=0)).ravel()-mu*mu
    idx=np.argsort(var)[::-1][:min(n_genes,X.shape[1])]
    return X[:,idx],genes[idx]

def incremental_pca(X,n_components=50,batch=2048):
    ncomp=min(n_components,X.shape[1],X.shape[0]-1)
    ipca=IncrementalPCA(n_components=ncomp,batch_size=batch)
    for i in range(0,X.shape[0],batch):
        ipca.partial_fit(X[i:i+batch].toarray())
    chunks=[]
    for i in range(0,X.shape[0],batch):
        chunks.append(ipca.transform(X[i:i+batch].toarray()))
    return np.vstack(chunks),ipca

def nystrom_kpca(Z50,n_components=1024,landmark_seed=17):
    # Gamma follows sklearn RBF default convention on the PCA representation.
    gamma=1.0/max(Z50.shape[1],1)
    ncomp=min(n_components,Z50.shape[0])
    ny=Nystroem(
        kernel="rbf",
        gamma=gamma,
        n_components=ncomp,
        random_state=landmark_seed,
    )
    Phi=ny.fit_transform(Z50)
    Z2=PCA(n_components=2,random_state=17).fit_transform(Phi)
    return Z2,{"gamma":gamma,"nystrom_components":ncomp}

def fit_embeddings(X):
    if umap is None:
        raise RuntimeError("umap-learn not installed.")
    Z50,_=incremental_pca(X,n_components=50,batch=2048)
    out={"PCA":Z50[:,:2]}

    Zk,kmeta=nystrom_kpca(Z50,n_components=min(1024,len(Z50)))
    out["kPCA*"]=Zk

    out["UMAP"]=umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=.2,
        metric="euclidean",
        random_state=17,
        low_memory=True,
    ).fit_transform(Z50)

    out["t-SNE"]=TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=17,
        max_iter=1000,
        method="barnes_hut",
        angle=.5,
    ).fit_transform(Z50)

    return out,kmeta

def find_ledger(root,sample):
    p=root/"results/hierarchy_v0911_specimen_local_contextual_flow/ledger"/sample/"candidate_boundaries/step_000000.parquet"
    if p.exists(): return p
    hits=[q for q in (root/"results").rglob("step_000000.parquet")
          if sample in str(q) and "candidate_boundaries" in str(q)
          and not any(t in str(q).lower() for t in ("hrm","smoke","test","null"))]
    if not hits: raise FileNotFoundError(f"No Level-0 ledger for {sample}")
    return sorted(hits,key=lambda q:len(str(q)))[0]

def choose_input(root,sample,kind):
    if kind=="matrix":
        p=root/"data"/sample/"cell_feature_matrix.h5"
    else:
        p=root/"data"/sample/"cells.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    return p

def resolve_edges(edge_df,cells):
    """
    Resolve Level-0 endpoints on the FULL aligned tissue.
    Supports explicit cell IDs or integer production-row indices.
    """
    id_to_local={c:i for i,c in enumerate(cells.cell_id.astype(str))}
    prod_to_local={int(r):i for i,r in enumerate(cells._production_row)}

    ai=edge_df.super_i.astype(str)
    bi=edge_df.super_j.astype(str)
    id_hit=np.mean(ai.isin(id_to_local)&bi.isin(id_to_local))

    int_hit=-1.0
    aa=bb=None
    try:
        aa=pd.to_numeric(edge_df.super_i,errors="raise").astype(np.int64)
        bb=pd.to_numeric(edge_df.super_j,errors="raise").astype(np.int64)
        int_hit=np.mean(aa.isin(prod_to_local)&bb.isin(prod_to_local))
    except Exception:
        pass

    mode="cell_id" if id_hit>=int_hit else "production_row"
    cand=set();adm=set()
    if mode=="cell_id":
        for u,v,ok in zip(ai,bi,edge_df.admissible):
            if u not in id_to_local or v not in id_to_local: continue
            i,j=id_to_local[u],id_to_local[v]
            if i==j:continue
            e=(min(i,j),max(i,j));cand.add(e)
            if bool(ok):adm.add(e)
    else:
        for u,v,ok in zip(aa,bb,edge_df.admissible):
            u=int(u);v=int(v)
            if u not in prod_to_local or v not in prod_to_local:continue
            i,j=prod_to_local[u],prod_to_local[v]
            if i==j:continue
            e=(min(i,j),max(i,j));cand.add(e)
            if bool(ok):adm.add(e)

    audit={
        "endpoint_mode":mode,
        "id_pair_hit_fraction":float(id_hit),
        "production_row_pair_hit_fraction":float(int_hit),
        "n_cells":int(len(cells)),
        "n_candidate_edges":int(len(cand)),
        "n_admissible_edges":int(len(adm)),
        "admissible_fraction_of_candidate":float(len(adm)/len(cand)) if cand else None,
    }
    if not cand:
        raise RuntimeError("No mapped candidate edges:\n"+json.dumps(audit,indent=2))
    return cand,adm,audit

def knn_edges(Z,k):
    nn=NearestNeighbors(
        n_neighbors=min(k+1,len(Z)),
        algorithm="auto",
        n_jobs=-1,
    )
    ind=nn.fit(Z).kneighbors(return_distance=False)[:,1:]
    E=set()
    for i,row in enumerate(ind):
        for j in row:
            j=int(j)
            if i!=j:E.add((min(i,j),max(i,j)))
    return E

def evaluate(E,cand,adm,n):
    ne=max(len(E),1)
    hit_c=len(E&cand)
    hit_a=len(E&adm)
    total_pairs=n*(n-1)/2
    p0=len(cand)/total_pairs
    p=hit_c/ne
    enrichment=p/p0 if p0>0 else np.nan
    concordance=hit_a/hit_c if hit_c>0 else np.nan
    return {
        "n_edges":len(E),
        "candidate_hits":hit_c,
        "admissible_hits":hit_a,
        "candidate_precision":p,
        "candidate_recall":hit_c/max(len(cand),1),
        "random_pair_candidate_density":p0,
        "physical_interface_enrichment":enrichment,
        "tissue_constraint_concordance":concordance,
    }

def full_tissue_edge_sample(E,cand,adm,max_edges=1400,seed=17):
    """
    Visualization only.
    Priority: show recovered candidate/admissible hits, then a deterministic
    sample of nonphysical latent edges.
    """
    rng=np.random.default_rng(seed)
    hit_a=list(E&adm)
    hit_c=list((E&cand)-adm)
    other=list(E-cand)

    def sample_list(x,n):
        if len(x)<=n:return x
        idx=rng.choice(len(x),size=n,replace=False)
        return [x[i] for i in idx]

    na=min(len(hit_a),max_edges//3)
    nc=min(len(hit_c),max_edges//3)
    no=max_edges-na-nc
    return (
        sample_list(hit_a,na),
        sample_list(hit_c,nc),
        sample_list(other,no),
    )

def draw_full_tissue(ax,cells,E,cand,adm,title,scale_bar=False):
    ax.scatter(
        cells.x,cells.y,
        s=.18,c=".78",alpha=.65,
        linewidths=0,rasterized=True
    )
    A,C,O=full_tissue_edge_sample(E,cand,adm,max_edges=1200,seed=17)
    # nonphysical latent edges very faint
    for i,j in O:
        ax.plot(
            [cells.x.iloc[i],cells.x.iloc[j]],
            [cells.y.iloc[i],cells.y.iloc[j]],
            color=".80",lw=.18,alpha=.12,rasterized=True
        )
    for i,j in C:
        ax.plot(
            [cells.x.iloc[i],cells.x.iloc[j]],
            [cells.y.iloc[i],cells.y.iloc[j]],
            color="#f28e00",lw=.40,alpha=.60,rasterized=True
        )
    for i,j in A:
        ax.plot(
            [cells.x.iloc[i],cells.x.iloc[j]],
            [cells.y.iloc[i],cells.y.iloc[j]],
            color="#149c38",lw=.48,alpha=.75,rasterized=True
        )
    ax.set_title(title,fontsize=7.3,fontweight="bold",pad=3)
    ax.set_aspect("equal")
    ax.axis("off")

def curve(ax,df,sample,metric,title,L,logy=False,baseline=None):
    d=df[df["sample"]==sample]
    for m in METHODS:
        q=d[d.method==m]
        ax.plot(
            q.k,q[metric],
            marker="o",ms=3.1,lw=1.25,
            label=m,color=METHOD_COLORS[m]
        )
    if baseline is not None:
        ax.axhline(baseline,color=".35",ls="--",lw=.9)
    if logy: ax.set_yscale("log")
    ax.set_title(title,fontsize=9,fontweight="bold",pad=7)
    ax.set_xlabel(r"Latent-neighbor budget, $k$",fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.text(-.12,1.07,L,transform=ax.transAxes,fontsize=13,fontweight="bold")

def run_sample(root,sample):
    M=choose_input(root,sample,"matrix")
    C=choose_input(root,sample,"cells")
    X,bc,genes=load10x(M)
    cells=load_cells(C)
    X,cells=align_matrix_cells(X,bc,cells)
    X,genes=normalize_select(X,genes,2000)

    edge_df=pd.read_parquet(find_ledger(root,sample))[["super_i","super_j","admissible"]]
    cand,adm,audit=resolve_edges(edge_df,cells)
    print(f"\n[{sample}] FULL-TISSUE EDGE AUDIT")
    print(json.dumps(audit,indent=2))

    t0=time.time()
    emb,kmeta=fit_embeddings(X)
    print(f"[{sample}] embeddings complete in {(time.time()-t0)/60:.2f} min")

    rows=[]
    vis={}
    for method,Z in emb.items():
        for k in KS:
            E=knn_edges(Z,k)
            ev=evaluate(E,cand,adm,len(cells))
            rows.append(dict(sample=sample,method=method,k=k,**ev))
            if k==5:
                vis[method]=E

    # SUTRA panel displays its admissible physical graph.
    vis["SUTRA"]=adm

    return {
        "cells":cells,
        "cand":cand,
        "adm":adm,
        "embeddings":emb,
        "vis_edges":vis,
        "metrics":pd.DataFrame(rows),
        "audit":audit,
        "kernel_meta":kmeta,
        "matrix":str(M),
        "cells_path":str(C),
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=str(Path.home()/"Desktop/SUTRA"))
    ap.add_argument("--outdir",default=str(Path.home()/"Desktop/SUTRA/figures/fig3_full_tissue_v03"))
    args=ap.parse_args()

    root=Path(args.root).expanduser()
    out=Path(args.outdir).expanduser()
    out.mkdir(parents=True,exist_ok=True)

    B=run_sample(root,"healthy_reference")
    K=run_sample(root,"nondiseased_kidney")
    metrics=pd.concat(
        [
            B["metrics"],
            K["metrics"],
        ],
        ignore_index=True,
    )
    metrics.to_csv(out/"Figure3_full_tissue_metrics.csv",index=False)

    # ---------------- MAIN FIGURE ----------------
    fig=plt.figure(figsize=(11.4,8.3),facecolor="white")
    gs=fig.add_gridspec(
        3,2,
        height_ratios=[1.55,1.0,1.0],
        hspace=.39,wspace=.25,
        left=.060,right=.985,top=.945,bottom=.105
    )

    # A/B full tissue maps.
    for col,obj,L,name in [
        (0,B,"A","Brain"),
        (1,K,"B","Kidney"),
    ]:
        sub=gs[0,col].subgridspec(1,5,wspace=.025)
        for j,m in enumerate(["SUTRA","PCA","kPCA*","UMAP","t-SNE"]):
            ax=fig.add_subplot(sub[0,j])
            draw_full_tissue(
                ax,obj["cells"],obj["vis_edges"][m],
                obj["cand"],obj["adm"],
                "SUTRA\n(tissue graph)" if m=="SUTRA" else f"{m}\n(kNN)"
            )
        frame=fig.add_subplot(gs[0,col],frameon=False)
        frame.set_xticks([]);frame.set_yticks([])
        frame.text(-.07,1.04,L,transform=frame.transAxes,fontsize=14,fontweight="bold")
        frame.text(.50,1.04,name,transform=frame.transAxes,ha="center",fontsize=11,fontweight="bold")

    # C/D physical-interface enrichment.
    curve(
        fig.add_subplot(gs[1,0]),metrics,"healthy_reference",
        "physical_interface_enrichment",
        "Physical-interface enrichment","C",
        logy=True,baseline=1.0
    )
    curve(
        fig.add_subplot(gs[1,1]),metrics,"nondiseased_kidney",
        "physical_interface_enrichment",
        "Physical-interface enrichment","D",
        logy=True,baseline=1.0
    )

    # E/F SUTRA-constraint concordance among recovered physical interfaces.
    baseB=B["audit"]["admissible_fraction_of_candidate"]
    baseK=K["audit"]["admissible_fraction_of_candidate"]

    curve(
        fig.add_subplot(gs[2,0]),metrics,"healthy_reference",
        "tissue_constraint_concordance",
        "Tissue-constraint concordance","E",
        baseline=baseB
    )
    axF=fig.add_subplot(gs[2,1])
    curve(
        axF,metrics,"nondiseased_kidney",
        "tissue_constraint_concordance",
        "Tissue-constraint concordance","F",
        baseline=baseK
    )

    # y labels and annotations.
    fig.axes[-4].set_ylabel("Enrichment over random cell pairs",fontsize=8)
    fig.axes[-3].set_ylabel("Enrichment over random cell pairs",fontsize=8)
    fig.axes[-2].set_ylabel("Admissible fraction among recovered\nphysical interfaces",fontsize=8)
    fig.axes[-1].set_ylabel("Admissible fraction among recovered\nphysical interfaces",fontsize=8)

    # one legend for methods + one compact edge key
    h,l=axF.get_legend_handles_labels()
    fig.legend(
        h,l,
        loc="lower center",
        bbox_to_anchor=(.36,.018),
        ncol=4,frameon=False,fontsize=8
    )
    fig.text(
        .735,.030,
        "Map edges:  orange = recovered physical interface   ·   green = recovered & SUTRA-admissible",
        ha="center",fontsize=7.2
    )

    fig.savefig(out/"Figure3_full_tissue_v03.png",dpi=400,facecolor="white")
    fig.savefig(out/"Figure3_full_tissue_v03.pdf",facecolor="white")
    plt.close(fig)

    manifest={
        "version":"0.3",
        "comparison_type":"matched relational benchmark on full Level-0 tissue",
        "methods":METHODS,
        "k_values":KS,
        "full_tissue_metrics":True,
        "visualization_edge_thinning_only":True,
        "SUTRA_role":"physical candidate/admissibility reference; not a generic dimensionality-reduction method",
        "kernel_PCA":"Nyström approximation to RBF kernel PCA",
        "brain":{
            "n_cells":len(B["cells"]),
            "audit":B["audit"],
            "kernel_meta":B["kernel_meta"],
        },
        "kidney":{
            "n_cells":len(K["cells"]),
            "audit":K["audit"],
            "kernel_meta":K["kernel_meta"],
        },
        "tissue_constraint_concordance_note":"Concordance with SUTRA's tissue-admissibility criterion; not independent ground truth.",
        "heldout_biology":"not part of v0.3",
    }
    (out/"Figure3_full_tissue_v03_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n"
    )

    print("\nWROTE",out/"Figure3_full_tissue_v03.png")
    print("WROTE",out/"Figure3_full_tissue_metrics.csv")
    print("WROTE",out/"Figure3_full_tissue_v03_manifest.json")

if __name__=="__main__":
    main()
