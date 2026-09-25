
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .core import ensure, load_landmark_rows, mass_stats, representative_landmarks

def set_style(cfg):
    s=cfg["style"]
    plt.rcParams.update({
        "font.size":s["font_size"],"axes.labelsize":s["font_size"],
        "axes.titlesize":s["title_size"],"xtick.labelsize":s["font_size"]-1,
        "ytick.labelsize":s["font_size"]-1,"legend.fontsize":s["font_size"]-1,
        "axes.linewidth":0.8,"lines.linewidth":s["line_width"],
        "pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none",
        "savefig.transparent":False
    })

def save(fig,stem,cfg,sources,panel_type,sample,extra=None):
    ensure(stem.parent); outs=[]
    for fmt in cfg["style"]["formats"]:
        p=stem.with_suffix("."+fmt)
        kw={"bbox_inches":"tight"}
        if fmt=="png":kw["dpi"]=int(cfg["style"]["png_dpi"])
        fig.savefig(p,**kw);outs.append(str(p))
    meta={"panel_type":panel_type,"sample":sample,"outputs":outs,
          "sources":[str(x) for x in sources],"atomic_panel":True}
    if extra:meta.update(extra)
    stem.with_suffix(".json").write_text(json.dumps(meta,indent=2))
    plt.close(fig);return meta

def hierarchy_landmarks(project,sample,cfg,pdir):
    p=project/cfg["inputs"]["final_landscape"]/sample/"final_supported_pareto_landmarks.csv"
    d=pd.read_csv(p).sort_values("center_eval")
    fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
    ax.plot(d["center_eval"],d["center_nodes"]);ax.scatter(d["center_eval"],d["center_nodes"],s=cfg["style"]["marker_size"])
    ax.set_yscale("log");ax.set_xlabel("Hierarchy evaluation");ax.set_ylabel("Active spatial units")
    return [save(fig,pdir/f"{sample}__supported_landmarks",cfg,[p],"supported_landmarks",sample)]

def mass_spectrum_representatives(project,sample,cfg,pdir):
    out=[]
    rows=representative_landmarks(load_landmark_rows(project,sample,cfg),5)
    for lid,r,labels,src in rows:
        _,m=np.unique(labels,return_counts=True)
        vals,freq=np.unique(m,return_counts=True);pmf=freq/freq.sum()
        cx=np.sort(np.unique(m));cy=np.array([(m>=x).mean() for x in cx])
        rank=np.sort(m)[::-1]
        specs=[("pmf",vals,pmf,"Supernode mass (cells)","Empirical PMF"),
               ("ccdf",cx,cy,"Supernode mass (cells)","P(Mass ≥ m)"),
               ("rank",np.arange(1,len(rank)+1),rank,"Rank","Supernode mass (cells)")]
        for typ,x,y,xl,yl in specs:
            fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
            ax.scatter(x,y,s=cfg["style"]["marker_size"]);ax.set_xscale("log");ax.set_yscale("log")
            ax.set_xlabel(xl);ax.set_ylabel(yl)
            out.append(save(fig,pdir/f"{sample}__L{lid:02d}__mass_{typ}",cfg,[src],f"mass_{typ}",sample,{"landmark_id":lid}))
    return out

def mass_stat_trajectory(project,sample,cfg,pdir):
    rows=[]
    sources=[]
    for lid,r,labels,src in load_landmark_rows(project,sample,cfg):
        st=mass_stats(labels);sources.append(src)
        rows.append({"landmark_id":lid,"nodes":int(r["center_nodes"]),"gini":st["gini"],
                     "n_eff_fraction":st["n_eff_fraction"],"max_mass_fraction":st["max_mass_fraction"]})
    d=pd.DataFrame(rows); csv=pdir/f"{sample}__mass_stat_trajectory.csv";d.to_csv(csv,index=False)
    out=[]
    for col,label in [("gini","Mass Gini"),("n_eff_fraction","Effective-domain fraction"),("max_mass_fraction","Largest-domain mass fraction")]:
        fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
        ax.plot(d["nodes"],d[col]);ax.scatter(d["nodes"],d[col],s=cfg["style"]["marker_size"])
        ax.set_xscale("log");ax.invert_xaxis();ax.set_xlabel("Active spatial units");ax.set_ylabel(label)
        out.append(save(fig,pdir/f"{sample}__{col}_trajectory",cfg,[csv],f"{col}_trajectory",sample))
    return out

def merger_panels(project,sample,cfg,pdir):
    p=project/cfg["inputs"]["merger_atlas"]/sample/"landmark_merger_structure.csv"
    d=pd.read_csv(p);out=[]
    if not len(d):return out
    x,y=np.unique(d["n_entry_children"].astype(int),return_counts=True);y=y/y.sum()
    fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
    ax.scatter(x,y,s=cfg["style"]["marker_size"]);ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("Merger order");ax.set_ylabel("Empirical PMF")
    out.append(save(fig,pdir/f"{sample}__merger_order_pmf",cfg,[p],"merger_order_pmf",sample))
    m=d["center_mass"].astype(float).to_numpy();x=np.sort(np.unique(m));y=np.array([(m>=a).mean() for a in x])
    fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
    ax.scatter(x,y,s=cfg["style"]["marker_size"]);ax.set_xscale("log");ax.set_yscale("log")
    ax.set_xlabel("Merger-parent mass (cells)");ax.set_ylabel("P(Mass ≥ m)")
    out.append(save(fig,pdir/f"{sample}__merger_parent_mass_ccdf",cfg,[p],"merger_parent_mass_ccdf",sample))
    fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
    ax.scatter(d["center_mass"],d["n_entry_children"],s=8,alpha=.45)
    ax.set_xscale("log");ax.set_yscale("log");ax.set_xlabel("Parent mass (cells)");ax.set_ylabel("Merger order")
    out.append(save(fig,pdir/f"{sample}__merger_order_vs_mass",cfg,[p],"merger_order_vs_mass",sample))
    return out

def gene_trajectory_panels(project,sample,cfg,rdir,pdir):
    p=rdir/sample/"gene_trajectories"/"gene_landmark_trajectories.csv"
    if not p.exists():return []
    d=pd.read_csv(p)
    summ=project/cfg["inputs"]["brain_pipeline"]/sample/"merger_biology"/"gene_merger_score_summary.csv"
    s=pd.read_csv(summ).head(int(cfg["gene_trajectory"]["top_per_sample"]))
    genes=list(s["gene"])
    q=d[d["gene"].isin(genes)]
    piv=q.pivot_table(index="gene",columns="landmark_id",values="z",aggfunc="mean").reindex(genes)
    fig,ax=plt.subplots(figsize=(cfg["style"]["wide_width_in"],max(3.2,.18*len(genes)+1)))
    im=ax.imshow(piv.to_numpy(),aspect="auto")
    ax.set_yticks(np.arange(len(piv)));ax.set_yticklabels(piv.index)
    ax.set_xticks(np.arange(len(piv.columns)));ax.set_xticklabels(piv.columns)
    ax.set_xlabel("Supported landmark");ax.set_ylabel("")
    cb=fig.colorbar(im,ax=ax);cb.set_label("Merger-concordance z")
    return [save(fig,pdir/f"{sample}__top_gene_landmark_heatmap",cfg,[p,summ],"gene_landmark_heatmap",sample)]

def cell_subsampling_panel(sample,cfg,rdir,pdir):
    p=rdir/sample/"stability"/"cell_subsampling_stability.csv"
    if not p.exists():return []
    d=pd.read_csv(p);out=[]
    for col,label in [("gini","Mass Gini"),("n_eff_fraction","Effective-domain fraction"),("mass_wasserstein","Wasserstein distance")]:
        g=d.groupby(["retention","landmark_id"])[col].median().reset_index()
        fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
        for ret,q in g.groupby("retention"):
            ax.plot(q["landmark_id"],q[col],marker="o",markersize=2.5,label=f"{int(ret*100)}%")
        ax.set_xlabel("Supported landmark");ax.set_ylabel(label);ax.legend(frameon=False,title="Cells retained")
        out.append(save(fig,pdir/f"{sample}__cell_subsampling_{col}",cfg,[p],f"cell_subsampling_{col}",sample))
    return out

def scheme_null_panel(sample,cfg,rdir,pdir):
    p=rdir/sample/"stability"/"reduction_scheme_nulls.csv"
    if not p.exists():return []
    d=pd.read_csv(p);out=[]
    for col,label in [("gini","Mass Gini"),("n_eff_fraction","Effective-domain fraction"),("max_mass_fraction","Largest-domain mass fraction")]:
        fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
        models=list(d["model"].drop_duplicates())
        for mi,model in enumerate(models):
            q=d[d["model"]==model]
            vals=[]
            xs=[]
            for lid,z in q.groupby("landmark_id"):
                vals.append(float(np.median(z[col])));xs.append(lid)
            ax.plot(xs,vals,marker="o",markersize=2.5,label=model.replace("_"," "))
        ax.set_xlabel("Representative landmark");ax.set_ylabel(label);ax.legend(frameon=False)
        out.append(save(fig,pdir/f"{sample}__scheme_control_{col}",cfg,[p],f"scheme_control_{col}",sample))
    return out

def cross_brain_mass(project,cfg,pdir):
    rows=[]
    src=[]
    for s in cfg["samples"]:
        sample=s["id"]
        for lid,r,labels,p in load_landmark_rows(project,sample,cfg):
            st=mass_stats(labels);src.append(p)
            rows.append({"sample":sample,"landmark_id":lid,"nodes":int(r["center_nodes"]),
                         "gini":st["gini"],"n_eff_fraction":st["n_eff_fraction"],"max_mass_fraction":st["max_mass_fraction"]})
    d=pd.DataFrame(rows);csv=pdir/"brain__mass_statistics.csv";d.to_csv(csv,index=False)
    out=[]
    for col,label in [("gini","Mass Gini"),("n_eff_fraction","Effective-domain fraction"),("max_mass_fraction","Largest-domain mass fraction")]:
        fig,ax=plt.subplots(figsize=(cfg["style"]["single_width_in"],cfg["style"]["height_in"]))
        for sample,q in d.groupby("sample"):
            q=q.sort_values("nodes",ascending=False)
            ax.plot(q["nodes"],q[col],marker="o",markersize=2.5,label=sample)
        ax.set_xscale("log");ax.invert_xaxis();ax.set_xlabel("Active spatial units");ax.set_ylabel(label);ax.legend(frameon=False)
        out.append(save(fig,pdir/f"brain__{col}_comparison",cfg,[csv],f"brain_{col}_comparison","brain_cohort"))
    return out
