from __future__ import annotations
import argparse,json
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np,pandas as pd
import pyarrow as pa,pyarrow.parquet as pq
import matplotlib.pyplot as plt

from sutra.hierarchy.v108.mass_spectrum import (
    normalized_mass,log_binned_pmf,empirical_ccdf,rank_size,lorenz_curve,
    local_loglog_slope,concentration_trajectory
)

SAMPLES=("alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc")

def req(p):
    p=Path(p)
    if not p.exists():raise FileNotFoundError(p)
    return p

def wpq(df,p):
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),p,compression="zstd")

def savefig(fig,path):
    fig.tight_layout()
    fig.savefig(path,bbox_inches="tight")
    plt.close(fig)

def plot_family(sample,summary,units,out,cfg):
    # PMF overlay
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    for lid,g in units.groupby("landmark_id"):
        pmf=log_binned_pmf(g.mass,cfg["bins_per_decade"])
        q=pmf["count"]>0
        ax.loglog(pmf.loc[q,"bin_center"],pmf.loc[q,"pmf"],marker="o",linewidth=1,label=str(lid))
    ax.set_xlabel("supernode mass")
    ax.set_ylabel("PMF")
    ax.legend(fontsize=7)
    savefig(fig,out/f"{sample}_mass_pmf_loglog.pdf")

    # CCDF overlay
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    for lid,g in units.groupby("landmark_id"):
        c=empirical_ccdf(g.mass)
        ax.loglog(c.mass,c.ccdf,linewidth=1.2,label=str(lid))
    ax.set_xlabel("supernode mass")
    ax.set_ylabel("P(M ≥ m)")
    ax.legend(fontsize=7)
    savefig(fig,out/f"{sample}_mass_ccdf_loglog.pdf")

    # rank-size overlay
    fig,ax=plt.subplots(figsize=(7.2,5.4))
    for lid,g in units.groupby("landmark_id"):
        r=rank_size(g.mass)
        ax.loglog(r["rank"],r.mass_fraction,linewidth=1.2,label=str(lid))
    ax.set_xlabel("rank")
    ax.set_ylabel("normalized supernode mass")
    ax.legend(fontsize=7)
    savefig(fig,out/f"{sample}_mass_rank_size_loglog.pdf")

    # Lorenz
    fig,ax=plt.subplots(figsize=(6.4,5.4))
    ax.plot([0,1],[0,1],linestyle="--",linewidth=1)
    for lid,g in units.groupby("landmark_id"):
        l=lorenz_curve(g.mass)
        ax.plot(l.fraction_supernodes,l.fraction_mass,linewidth=1.2,label=str(lid))
    ax.set_xlabel("fraction of supernodes")
    ax.set_ylabel("fraction of tissue mass")
    ax.legend(fontsize=7)
    savefig(fig,out/f"{sample}_mass_lorenz.pdf")

    # concentration trajectory
    tr=concentration_trajectory(summary)
    if len(tr):
        fig,ax=plt.subplots(figsize=(7.2,5.4))
        if "effective_domain_number" in tr.columns:
            ax.semilogy(tr.minimum_removed_fraction,tr.effective_domain_number,marker="o",label="N_eff")
        if "K80" in tr.columns:
            ax.semilogy(tr.minimum_removed_fraction,tr.K80,marker="o",label="K80")
        if "formal_supernodes" in tr.columns:
            ax.semilogy(tr.minimum_removed_fraction,tr.formal_supernodes,marker="o",label="formal N")
        ax.set_xlabel("removed fraction")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        savefig(fig,out/f"{sample}_mass_concentration_trajectory.pdf")

def one(project_s,sample,cfg):
    project=Path(project_s)
    src=project/"results"/"hierarchy_v107_structural_domain_audit"/sample
    units=pd.read_parquet(req(src/"supernode_mass_heterogeneity.parquet"))
    summary=pd.read_parquet(req(src/"landmark_domain_summary.parquet"))
    if len(units)==0:raise RuntimeError(f"{sample}: empty supernode mass table")

    units=units.copy()
    units["normalized_mass"]=np.nan
    pmf_parts=[];ccdf_parts=[];rank_parts=[];lorenz_parts=[];slope_rows=[]
    for lid,g in units.groupby("landmark_id",sort=False):
        nm=normalized_mass(g.mass)
        units.loc[g.index,"normalized_mass"]=nm

        pmf=log_binned_pmf(g.mass,cfg["bins_per_decade"])
        pmf.insert(0,"landmark_id",lid);pmf_parts.append(pmf)

        cc=empirical_ccdf(g.mass)
        cc.insert(0,"landmark_id",lid);ccdf_parts.append(cc)

        rr=rank_size(g.mass)
        rr.insert(0,"landmark_id",lid);rank_parts.append(rr)

        ll=lorenz_curve(g.mass)
        ll.insert(0,"landmark_id",lid);lorenz_parts.append(ll)

        # descriptive central-window slopes only, not model claims
        cq=cc[(cc.ccdf<=cfg["ccdf_fit_upper"])&(cc.ccdf>=cfg["ccdf_fit_lower"])]
        slope,r2,n=local_loglog_slope(cq.mass,cq.ccdf,cfg["min_fit_points"])
        slope_rows.append({
            "landmark_id":lid,
            "ccdf_loglog_slope":slope,
            "ccdf_loglog_r2":r2,
            "ccdf_fit_points":n,
            "fit_is_descriptive_only":True
        })

    pmf=pd.concat(pmf_parts,ignore_index=True)
    ccdf=pd.concat(ccdf_parts,ignore_index=True)
    rank=pd.concat(rank_parts,ignore_index=True)
    lorenz=pd.concat(lorenz_parts,ignore_index=True)
    slopes=pd.DataFrame(slope_rows)
    traj=concentration_trajectory(summary)

    out=project/"results"/"hierarchy_v108_supernode_mass_spectrum"/sample
    out.mkdir(parents=True,exist_ok=True)
    wpq(units,out/"supernode_mass_normalized.parquet")
    wpq(pmf,out/"mass_pmf_logbins.parquet")
    wpq(ccdf,out/"mass_ccdf_empirical.parquet")
    wpq(rank,out/"mass_rank_size.parquet")
    wpq(lorenz,out/"mass_lorenz.parquet")
    wpq(slopes,out/"mass_loglog_slope_descriptive.parquet")
    wpq(traj,out/"mass_concentration_trajectory.parquet")
    plot_family(sample,summary,units,out,cfg)

    cert={
        "sample":sample,
        "landmarks":int(units.landmark_id.nunique()),
        "pmf_exported":True,
        "ccdf_exported":True,
        "rank_size_exported":True,
        "lorenz_exported":True,
        "normalized_mass_exported":True,
        "concentration_trajectory_exported":True,
        "loglog_slopes_descriptive_only":True,
        "scientific_hierarchy_changed":False,
        "new_coarse_graining_performed":False,
        "status":"PASS"
    }
    (out/"mass_spectrum_certificate.json").write_text(json.dumps(cert,indent=2)+"\n")
    return cert

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    ap.add_argument("--config",default="configs/hierarchy_v108_supernode_mass_spectrum.json")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    cfg=json.loads(req(project/a.config).read_text())

    c=json.loads(req(project/"results"/"hierarchy_v107_structural_domain_audit"/
        "structural_domain_audit_global_certificate.json").read_text())
    if c.get("MASS_WEIGHTED_DOMAIN_STRUCTURE_READY") is not True:
        raise SystemExit("ERROR: v1.0.7 mass-weighted structure not certified")

    print("STRATA 1.0.8 | Supernode mass-spectrum audit")
    print("Frozen hierarchy is unchanged.")
    print("Generating log-log PMF, empirical CCDF, rank-size and Lorenz curves.")
    print("Masses are also normalized by total tissue mass for cross-specimen comparison.")
    print("Any log-log slopes are descriptive summaries only; no distribution family is asserted.")
    print(f"Parallel specimen workers: {min(a.workers,3)}\n")

    reps=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        fs={ex.submit(one,str(project),s,cfg):s for s in SAMPLES}
        for f in as_completed(fs):
            r=f.result();reps.append(r)
            print(f"[DONE] {r['sample']}: landmarks={r['landmarks']} PMF/CCDF/rank/Lorenz PASS",flush=True)

    reps.sort(key=lambda x:SAMPLES.index(x["sample"]))
    gate=all(r["status"]=="PASS" for r in reps)
    out=project/"results"/"hierarchy_v108_supernode_mass_spectrum"
    out.mkdir(parents=True,exist_ok=True)
    g={
        "strata_version":"1.0.8",
        "stage":"supernode mass-spectrum audit",
        "sample_reports":reps,
        "MASS_SPECTRUM_AUDIT_GATE":"PASS" if gate else "HOLD",
        "MASS_HETEROGENEITY_FIGURES_READY":bool(gate),
        "SCIENTIFIC_HIERARCHY_CHANGED":False,
        "NEW_COARSE_GRAINING_PERFORMED":False,
        "READY_FOR_HIERARCHY_ATLAS":bool(gate)
    }
    p=out/"mass_spectrum_global_certificate.json"
    p.write_text(json.dumps(g,indent=2)+"\n")
    print(f"\nMASS SPECTRUM AUDIT GATE: {g['MASS_SPECTRUM_AUDIT_GATE']}")
    print(f"MASS HETEROGENEITY FIGURES READY: {g['MASS_HETEROGENEITY_FIGURES_READY']}")
    print("SCIENTIFIC HIERARCHY CHANGED: False")
    print(f"READY FOR HIERARCHY ATLAS: {g['READY_FOR_HIERARCHY_ATLAS']}")
    print(f"Certificate: {p}")
if __name__=="__main__":main()
