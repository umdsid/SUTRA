
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import ensure_dir, read_scientific_evaluations, find_col, load_landmarks
from strata_hierarchy.v10931.replay_shards import load_coordinates

def pub_rc():
    plt.rcParams.update({
        "font.size":8,"axes.labelsize":8,"axes.titlesize":8,
        "xtick.labelsize":7,"ytick.labelsize":7,"legend.fontsize":7,
        "axes.linewidth":0.8,"lines.linewidth":1.0,
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"
    })

def save_atomic(fig,stem:Path,cfg:dict,sources,panel_type,sample,extra=None):
    ensure_dir(stem.parent)
    outs=[]
    for fmt in cfg["formats"]:
        p=stem.with_suffix("."+fmt)
        if fmt=="png": fig.savefig(p,dpi=int(cfg["png_dpi"]),bbox_inches="tight")
        else: fig.savefig(p,bbox_inches="tight")
        outs.append(str(p))
    meta={"panel_type":panel_type,"sample":sample,"outputs":outs,
          "source_tables":[str(x) for x in sources],"publication_atomic_panel":True}
    if extra: meta.update(extra)
    stem.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    plt.close(fig)
    return meta

def mass_stats(labels):
    _,c=np.unique(labels,return_counts=True)
    vals,freq=np.unique(c,return_counts=True)
    pmf=freq/freq.sum()
    cx=np.sort(np.unique(c))
    cy=np.array([(c>=x).mean() for x in cx])
    rank=np.sort(c)[::-1]
    return vals,pmf,cx,cy,rank

def plot_trajectory(project,sample,cfg,pdir):
    pub_rc()
    d,p=read_scientific_evaluations(project,sample,cfg)
    if d is None:return []
    nc=find_col(d.columns,["nodes","node_count","n_nodes","active_nodes"])
    ec=find_col(d.columns,["evaluation_index","eval","evaluation","eval_index"])
    if nc is None:return []
    x=np.arange(len(d)) if ec is None else pd.to_numeric(d[ec],errors="coerce")
    y=pd.to_numeric(d[nc],errors="coerce")
    fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
    ax.plot(x,y);ax.set_yscale("log")
    ax.set_xlabel("Evaluation");ax.set_ylabel("Active spatial units");ax.set_title(sample)
    return [save_atomic(fig,pdir/f"{sample}__hierarchy_trajectory",cfg,[p],"hierarchy_trajectory",sample)]

def plot_landmarks(project,sample,cfg,pdir):
    pub_rc()
    p=project/"results"/cfg["required_results"]["final_landscape"]/sample/"final_supported_pareto_landmarks.csv"
    d=pd.read_csv(p).sort_values("center_eval")
    fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
    ax.plot(d["center_eval"],d["center_nodes"],alpha=.5)
    ax.scatter(d["center_eval"],d["center_nodes"],s=24)
    ax.set_yscale("log");ax.set_xlabel("Evaluation");ax.set_ylabel("Landmark active units");ax.set_title(sample)
    return [save_atomic(fig,pdir/f"{sample}__supported_landmark_sequence",cfg,[p],"supported_landmark_sequence",sample)]

def plot_mass(project,sample,cfg,pdir):
    pub_rc();out=[]
    for lid,r,entry,center,ep,cp in load_landmarks(project,sample,cfg):
        vals,pmf,cx,cy,rank=mass_stats(center)
        fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
        ax.scatter(vals,pmf,s=12);ax.set_xscale("log");ax.set_yscale("log")
        ax.set_xlabel("Supernode mass (cells)");ax.set_ylabel("Empirical PMF");ax.set_title(f"{sample} · L{lid}")
        out.append(save_atomic(fig,pdir/f"{sample}__L{lid:02d}__mass_pmf",cfg,[cp],"supernode_mass_pmf",sample,{"landmark_id":lid}))
        fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
        ax.scatter(cx,cy,s=12);ax.set_xscale("log");ax.set_yscale("log")
        ax.set_xlabel("Supernode mass (cells)");ax.set_ylabel("P(Mass ≥ m)");ax.set_title(f"{sample} · L{lid}")
        out.append(save_atomic(fig,pdir/f"{sample}__L{lid:02d}__mass_ccdf",cfg,[cp],"supernode_mass_ccdf",sample,{"landmark_id":lid}))
        fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
        ax.scatter(np.arange(1,len(rank)+1),rank,s=9);ax.set_xscale("log");ax.set_yscale("log")
        ax.set_xlabel("Rank");ax.set_ylabel("Supernode mass (cells)");ax.set_title(f"{sample} · L{lid}")
        out.append(save_atomic(fig,pdir/f"{sample}__L{lid:02d}__mass_rank_size",cfg,[cp],"supernode_mass_rank_size",sample,{"landmark_id":lid}))
    return out

def plot_spatial(project,sample,cfg,pdir):
    pub_rc();out=[]
    n0=len(pd.read_parquet(project/"data"/sample/"cells.parquet"))
    P,psrc=load_coordinates(project,sample,n0)
    if P is None:return out
    maxp=int(cfg["spatial_panels"]["max_points"])
    for lid,r,entry,center,ep,cp in load_landmarks(project,sample,cfg):
        u,c=np.unique(center,return_counts=True);mass=dict(zip(u,c))
        z=np.log10(np.array([mass[x] for x in center],dtype=float))
        idx=np.arange(len(center))
        if len(idx)>maxp: idx=idx[::max(1,len(idx)//maxp)]
        fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
        sc=ax.scatter(P[idx,0],P[idx,1],c=z[idx],s=float(cfg["spatial_panels"]["point_size"]),rasterized=True)
        ax.set_aspect("equal");ax.set_xlabel("x");ax.set_ylabel("y");ax.set_title(f"{sample} · L{lid}")
        cb=fig.colorbar(sc,ax=ax);cb.set_label("log10(supernode mass)")
        out.append(save_atomic(fig,pdir/f"{sample}__L{lid:02d}__spatial_domain_mass",cfg,[cp,psrc],"spatial_domain_mass",sample,{"landmark_id":lid}))
    return out

def plot_mergers(project,sample,cfg,pdir):
    pub_rc();out=[]
    p=project/"results"/cfg["required_results"]["merger_atlas"]/sample/"landmark_merger_structure.csv"
    d=pd.read_csv(p)
    if not len(d):return out
    vals,freq=np.unique(d["n_entry_children"].astype(int),return_counts=True);pmf=freq/freq.sum()
    fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
    ax.scatter(vals,pmf,s=18);ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("Merger order");ax.set_ylabel("Empirical PMF");ax.set_title(sample)
    out.append(save_atomic(fig,pdir/f"{sample}__merger_order_pmf",cfg,[p],"merger_order_pmf",sample))
    m=d["center_mass"].astype(float).to_numpy();x=np.sort(np.unique(m));y=np.array([(m>=a).mean() for a in x])
    fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
    ax.scatter(x,y,s=18);ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("Merger-parent mass (cells)");ax.set_ylabel("P(Mass ≥ m)");ax.set_title(sample)
    out.append(save_atomic(fig,pdir/f"{sample}__merger_parent_mass_ccdf",cfg,[p],"merger_parent_mass_ccdf",sample))
    return out

def plot_gene_summary(sample,cfg,result_dir,pdir):
    pub_rc()
    p=result_dir/sample/"merger_biology"/"gene_merger_score_summary.csv"
    if not p.exists():return []
    d=pd.read_csv(p)
    if not len(d):return []
    n=int(cfg["gene_analysis"]["top_genes_per_panel"])
    q=d.head(n).sort_values("median_concordance_z")
    fig,ax=plt.subplots(figsize=cfg["figure_size_inches"])
    ax.scatter(q["median_concordance_z"],np.arange(len(q)),s=22)
    ax.set_yticks(np.arange(len(q)));ax.set_yticklabels(q["gene"])
    ax.set_xlabel("Median merger-concordance z");ax.set_ylabel("");ax.set_title(sample)
    return [save_atomic(fig,pdir/f"{sample}__top_merger_genes",cfg,[p],"top_merger_genes",sample)]

def plot_cross_brain(cfg,result_dir,pdir):
    pub_rc();tabs=[]
    for s in cfg["samples"]:
        p=result_dir/s["id"]/"merger_biology"/"gene_merger_score_summary.csv"
        if p.exists():tabs.append((s["id"],pd.read_csv(p).set_index("gene")))
    if len(tabs)<2:return []
    genes=set(tabs[0][1].index)
    for _,d in tabs[1:]:genes&=set(d.index)
    rows=[]
    for g in genes:
        vals=[float(d.loc[g,"median_concordance_z"]) for _,d in tabs]
        rows.append((g,float(np.median(vals)),vals))
    rows.sort(key=lambda x:x[1],reverse=True);rows=rows[:30]
    if not rows:return []
    M=np.array([r[2] for r in rows]);names=[r[0] for r in rows]
    fig,ax=plt.subplots(figsize=(5.2,5.5))
    im=ax.imshow(M,aspect="auto")
    ax.set_yticks(np.arange(len(names)));ax.set_yticklabels(names)
    ax.set_xticks(np.arange(len(tabs)));ax.set_xticklabels([x[0] for x in tabs],rotation=45,ha="right")
    cb=fig.colorbar(im,ax=ax);cb.set_label("Median merger-concordance z");ax.set_title("Brain cohort")
    src=[str(result_dir/s["id"]/"merger_biology"/"gene_merger_score_summary.csv") for s in cfg["samples"]]
    return [save_atomic(fig,pdir/"brain__merger_gene_recurrence",cfg,src,"brain_merger_gene_recurrence","brain_cohort")]

def generate_all_panels(project,cfg,result_dir,panel_root):
    manifest=[]
    for s in cfg["samples"]:
        sample=s["id"];pdir=ensure_dir(panel_root/sample)
        manifest+=plot_trajectory(project,sample,cfg,pdir)
        manifest+=plot_landmarks(project,sample,cfg,pdir)
        manifest+=plot_mass(project,sample,cfg,pdir)
        manifest+=plot_spatial(project,sample,cfg,pdir)
        manifest+=plot_mergers(project,sample,cfg,pdir)
        manifest+=plot_gene_summary(sample,cfg,result_dir,pdir)
    manifest+=plot_cross_brain(cfg,result_dir,ensure_dir(panel_root/"cohort"))
    mp=panel_root/"panel_manifest.json";mp.write_text(json.dumps(manifest,indent=2))
    return manifest,mp
