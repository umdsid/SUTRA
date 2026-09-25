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
    rejected=max(hit_c-hit_a,0)
    return {
        "n_edges":len(E),
        "nonphysical_fraction":(len(E)-hit_c)/ne,
        "physical_rejected_fraction":rejected/ne,
        "physical_admissible_fraction":hit_a/ne,
        "candidate_hits":hit_c,
        "admissible_hits":hit_a,
    }

def knn_indices(Z,k):
    nn=NearestNeighbors(n_neighbors=min(k+1,len(Z)),n_jobs=-1)
    return nn.fit(Z).kneighbors(return_distance=False)[:,1:]

def knn_edges_from_indices(ind):
    E=set()
    for i,row in enumerate(ind):
        for j in row:
            j=int(j)
            if i!=j:
                E.add((min(i,j),max(i,j)))
    return E

def cell_concordance_from_knn(ind,cand,adm):
    """
    Per-cell fraction of outgoing latent neighbors that are BOTH physical
    candidate interfaces and SUTRA-admissible.
    """
    n=len(ind)
    q=np.zeros(n,float)
    qphys=np.zeros(n,float)
    for i,row in enumerate(ind):
        hit_adm=0
        hit_phys=0
        for j in row:
            j=int(j)
            e=(min(i,j),max(i,j))
            if e in cand:
                hit_phys+=1
            if e in adm:
                hit_adm+=1
        denom=max(len(row),1)
        q[i]=hit_adm/denom
        qphys[i]=hit_phys/denom
    return q,qphys

def sutra_cell_fraction(n,cand,adm):
    deg_c=np.zeros(n,int); deg_a=np.zeros(n,int)
    for i,j in cand:
        deg_c[i]+=1;deg_c[j]+=1
    for i,j in adm:
        deg_a[i]+=1;deg_a[j]+=1
    q=np.divide(deg_a,deg_c,out=np.zeros(n,float),where=deg_c>0)
    return q,deg_c,deg_a

def edge_spans(cells,E):
    x=cells.x.to_numpy(float); y=cells.y.to_numpy(float)
    if not E:
        return np.array([],float)
    a=np.fromiter((e[0] for e in E),dtype=np.int64,count=len(E))
    b=np.fromiter((e[1] for e in E),dtype=np.int64,count=len(E))
    return np.hypot(x[a]-x[b],y[a]-y[b])

def ecdf(v):
    v=np.sort(np.asarray(v,float))
    if len(v)==0:
        return v,v
    y=np.arange(1,len(v)+1)/len(v)
    return v,y

def run_sample(root,sample,k_vis=5):
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

    method_edges={}
    method_knn={}
    concord={}
    physical_fraction={}
    rows=[]
    for method,Z in emb.items():
        ind=knn_indices(Z,k_vis)
        E=knn_edges_from_indices(ind)
        q,qphys=cell_concordance_from_knn(ind,cand,adm)
        method_knn[method]=ind
        method_edges[method]=E
        concord[method]=q
        physical_fraction[method]=qphys
        ev=evaluate(E,cand,adm,len(cells))
        rows.append(dict(sample=sample,method=method,**ev))

    # SUTRA itself: cell-level fraction of candidate interfaces that are admissible.
    qsutra,degc,dega=sutra_cell_fraction(len(cells),cand,adm)
    concord["SUTRA"]=qsutra

    # SUTRA relational composition is its candidate graph by definition.
    rows.append(dict(
        sample=sample,
        method="SUTRA",
        n_edges=len(cand),
        nonphysical_fraction=0.0,
        physical_rejected_fraction=(len(cand)-len(adm))/len(cand),
        physical_admissible_fraction=len(adm)/len(cand),
        candidate_hits=len(cand),
        admissible_hits=len(adm),
    ))

    # Physical spans. Comparators use their complete kNN graph; SUTRA uses admissible edges.
    spans={m:edge_spans(cells,E) for m,E in method_edges.items()}
    spans["SUTRA"]=edge_spans(cells,adm)
    candidate_span=edge_spans(cells,cand)
    median_candidate=float(np.median(candidate_span))

    return {
        "cells":cells,"cand":cand,"adm":adm,
        "embeddings":emb,"method_edges":method_edges,
        "concordance":concord,"physical_fraction":physical_fraction,
        "composition":pd.DataFrame(rows),
        "spans":spans,"median_candidate_span":median_candidate,
        "audit":audit,"kernel_meta":kmeta,
        "matrix":str(M),"cells_path":str(C),
    }

def tissue_concordance_row(fig,slot,obj,panel,organ):
    """
    Five full-tissue maps on fixed physical coordinates:
      PCA, kPCA*, UMAP, t-SNE: q_i = fraction of k latent neighbors that are
      SUTRA-admissible physical interfaces.
      SUTRA: q_i = admissible candidate degree / candidate degree.
    """
    sub=slot.subgridspec(1,5,wspace=.035)
    order=["PCA","kPCA*","UMAP","t-SNE","SUTRA"]
    ims=[]
    for j,m in enumerate(order):
        ax=fig.add_subplot(sub[0,j])
        q=obj["concordance"][m]
        im=ax.scatter(
            obj["cells"].x,obj["cells"].y,
            c=q,s=.32,cmap="viridis",vmin=0,vmax=1,
            linewidths=0,rasterized=True
        )
        ax.set_aspect("equal");ax.axis("off")
        ttl=m if m!="SUTRA" else "SUTRA\n(admissible/candidate)"
        if m!="SUTRA":
            ttl=f"{m}\n(admissible latent neighbors)"
        ax.set_title(ttl,fontsize=6.5,fontweight="bold",pad=2)
        ims.append(im)

    frame=fig.add_subplot(slot,frameon=False)
    frame.set_xticks([]);frame.set_yticks([])
    frame.text(-.065,1.04,panel,transform=frame.transAxes,
               fontsize=14,fontweight="bold")
    frame.text(.50,1.04,organ,transform=frame.transAxes,
               ha="center",fontsize=10.5,fontweight="bold")
    return ims[-1]

def span_panel(ax,obj,panel,title):
    order=["PCA","kPCA*","UMAP","t-SNE","SUTRA"]
    colors={
        "PCA":"tab:blue","kPCA*":"tab:orange",
        "UMAP":"tab:green","t-SNE":"tab:red","SUTRA":"black"
    }
    med=obj["median_candidate_span"]
    for m in order:
        v=obj["spans"][m]/med
        # deterministic thinning for plotting only; ECDF itself remains representative
        if len(v)>50000:
            rng=np.random.default_rng(17)
            v=v[rng.choice(len(v),50000,replace=False)]
        x,y=ecdf(v)
        ax.plot(x,y,lw=1.3,label=m,color=colors[m])
    ax.axvline(1,color=".45",ls="--",lw=.8)
    ax.set_xscale("log")
    ax.set_xlabel("Physical span / median candidate-interface span",fontsize=7.5)
    ax.set_ylabel("Cumulative fraction of relations",fontsize=7.5)
    ax.set_title(title,fontsize=8.5,fontweight="bold",pad=6)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=7)
    ax.text(-.12,1.06,panel,transform=ax.transAxes,
            fontsize=13,fontweight="bold")

def composition_panel(ax,obj,panel,title):
    d=obj["composition"].set_index("method")
    order=["PCA","kPCA*","UMAP","t-SNE","SUTRA"]
    x=np.arange(len(order))
    a=np.array([d.loc[m,"nonphysical_fraction"] for m in order])
    b=np.array([d.loc[m,"physical_rejected_fraction"] for m in order])
    c=np.array([d.loc[m,"physical_admissible_fraction"] for m in order])

    ax.bar(x,a,label="Nonphysical relation",color=".82")
    ax.bar(x,b,bottom=a,label="Physical / not SUTRA-admissible",color="#f28e00")
    ax.bar(x,c,bottom=a+b,label="Physical + SUTRA-admissible",color="#159c38")
    ax.set_xticks(x,order,rotation=25,ha="right",fontsize=7)
    ax.set_ylim(0,1)
    ax.set_ylabel("Fraction of relations",fontsize=7.5)
    ax.set_title(title,fontsize=8.5,fontweight="bold",pad=6)
    ax.spines["top"].set_visible(False);ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y",labelsize=7)
    ax.text(-.12,1.06,panel,transform=ax.transAxes,
            fontsize=13,fontweight="bold")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",default=str(Path.home()/"Desktop/SUTRA"))
    ap.add_argument("--k",type=int,default=5)
    ap.add_argument("--outdir",default=str(Path.home()/"Desktop/SUTRA/figures/fig3_relational_geometry_v04"))
    args=ap.parse_args()

    root=Path(args.root).expanduser()
    out=Path(args.outdir).expanduser()
    out.mkdir(parents=True,exist_ok=True)

    B=run_sample(root,"healthy_reference",k_vis=args.k)
    K=run_sample(root,"nondiseased_kidney",k_vis=args.k)

    comp=pd.concat([B["composition"],K["composition"]],ignore_index=True)
    comp.to_csv(out/"Figure3_relational_composition.csv",index=False)

    # Cell-level concordance source data
    for sample,obj in [("healthy_reference",B),("nondiseased_kidney",K)]:
        d=pd.DataFrame({
            "cell_id":obj["cells"].cell_id.astype(str),
            "x":obj["cells"].x,
            "y":obj["cells"].y,
        })
        for m in ["PCA","kPCA*","UMAP","t-SNE","SUTRA"]:
            d[f"{m}_concordance"]=obj["concordance"][m]
        d.to_parquet(out/f"{sample}__cellwise_relational_concordance.parquet",index=False)

    # ---------------- Main conceptual figure ----------------
    fig=plt.figure(figsize=(11.4,8.25),facecolor="white")
    gs=fig.add_gridspec(
        3,2,
        height_ratios=[1.45,1.0,1.0],
        hspace=.38,wspace=.26,
        left=.060,right=.985,top=.95,bottom=.10
    )

    im1=tissue_concordance_row(fig,gs[0,0],B,"A","Brain")
    im2=tissue_concordance_row(fig,gs[0,1],K,"B","Kidney")

    # One shared colorbar for A-B
    cax=fig.add_axes([.985,.675,.008,.16])
    cb=fig.colorbar(im2,cax=cax)
    cb.set_label("Relational concordance",fontsize=7)
    cb.ax.tick_params(labelsize=6)

    span_panel(
        fig.add_subplot(gs[1,0]),B,"C",
        "Physical span of inferred relations (Brain)"
    )
    span_panel(
        fig.add_subplot(gs[1,1]),K,"D",
        "Physical span of inferred relations (Kidney)"
    )

    composition_panel(
        fig.add_subplot(gs[2,0]),B,"E",
        "Relational composition (Brain)"
    )
    axF=fig.add_subplot(gs[2,1])
    composition_panel(
        axF,K,"F",
        "Relational composition (Kidney)"
    )

    # Legends are separated by semantic role.
    h1,l1=fig.axes[-4].get_legend_handles_labels()
    fig.legend(
        h1,l1,loc="lower center",
        bbox_to_anchor=(.31,.015),
        ncol=5,frameon=False,fontsize=7.5
    )
    h2,l2=axF.get_legend_handles_labels()
    fig.legend(
        h2,l2,loc="lower center",
        bbox_to_anchor=(.73,.015),
        ncol=3,frameon=False,fontsize=7.2
    )

    fig.savefig(out/"Figure3_relational_geometry_v04.png",dpi=400,facecolor="white")
    fig.savefig(out/"Figure3_relational_geometry_v04.pdf",facecolor="white")
    plt.close(fig)

    manifest={
        "version":"0.4",
        "purpose":"descriptive comparison of molecular nearness, physical nearness, and SUTRA organizational admissibility",
        "not_a_performance_benchmark":True,
        "k_for_latent_relations":args.k,
        "same_cells_and_molecular_input":True,
        "A_B":{
            "latent_methods":"cell color = fraction of k outgoing latent neighbors that are both physical candidate interfaces and SUTRA-admissible",
            "SUTRA":"cell color = admissible candidate degree / candidate degree",
            "note":"SUTRA color has related but not identical denominator; panel is descriptive, not a scalar performance comparison"
        },
        "C_D":"ECDF of physical Euclidean span of relations, normalized by specimen median candidate-interface span",
        "E_F":"composition of inferred relations into nonphysical / physical-nonadmissible / physical-admissible",
        "SUTRA_bar":"uses the SUTRA candidate graph, therefore has zero nonphysical category by construction",
        "kernel_PCA":"Nyström approximation to RBF kernel PCA",
        "brain_audit":B["audit"],
        "kidney_audit":K["audit"],
        "interpretive_caution":"A relation rejected by SUTRA is not declared biologically false; it is non-admissible under the current SUTRA organizational criterion."
    }
    (out/"Figure3_relational_geometry_v04_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n"
    )

    print("\nWROTE",out/"Figure3_relational_geometry_v04.png")
    print("WROTE",out/"Figure3_relational_composition.csv")
    print("WROTE",out/"Figure3_relational_geometry_v04_manifest.json")

if __name__=="__main__":
    main()
