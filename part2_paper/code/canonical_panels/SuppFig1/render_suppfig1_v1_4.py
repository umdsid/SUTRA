#!/usr/bin/env python3
from pathlib import Path
import argparse, json, csv, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

SAMPLES = ["healthy_reference","alzheimers","gbm_reference_addon","nondiseased_kidney","prcc"]
LABELS = {
    "healthy_reference":"Healthy brain",
    "alzheimers":"Alzheimer's",
    "gbm_reference_addon":"GBM",
    "nondiseased_kidney":"Nondiseased kidney",
    "prcc":"PRCC",
}
ORGANS = {"healthy_reference":"Brain","alzheimers":"Brain","gbm_reference_addon":"Brain",
          "nondiseased_kidney":"Kidney","prcc":"Kidney"}

plt.rcParams.update({
    "font.family":"sans-serif", "font.size":8.0, "axes.labelsize":8.0,
    "axes.titlesize":8.5, "xtick.labelsize":7.0, "ytick.labelsize":7.0,
    "legend.fontsize":7.0, "axes.linewidth":0.7, "pdf.fonttype":42, "ps.fonttype":42,
    "savefig.bbox":"tight", "savefig.pad_inches":0.08,
})

def read_json(p):
    with open(p) as f: return json.load(f)

def read_steps(p):
    rows=[]
    with open(p) as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return pd.DataFrame(rows)

def save(fig, out, stem):
    fig.savefig(out/f"{stem}.pdf")
    fig.savefig(out/f"{stem}.png", dpi=450)
    plt.close(fig)

def panel_letter(ax, letter):
    ax.text(-0.13, 1.08, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", clip_on=False)

def organ_gap_x():
    return np.array([0,1,2,4,5], float)

def style_axis(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(length=2.5, width=.6)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--hierarchy-root", required=True)
    ap.add_argument("--out", required=True)
    a=ap.parse_args()
    H=Path(a.hierarchy_root).expanduser().resolve()
    out=Path(a.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    src=out/"source_data"; src.mkdir(exist_ok=True)

    cal=pd.read_csv(H/"specimen_local_calibration_audit.csv")
    cal=cal.set_index("sample").loc[SAMPLES].reset_index()
    summaries=[]; allsteps={}
    for s in SAMPLES:
        d=H/"ledger"/s
        fs=read_json(d/"flow_summary.json")
        st=read_steps(d/"steps.jsonl")
        if st.empty: raise RuntimeError(f"{s}: empty steps")
        allsteps[s]=st
        first,last=st.iloc[0],st.iloc[-1]
        summaries.append({
            "sample":s,"organ":ORGANS[s],"level0_cells":fs["level0_cells"],
            "final_nodes":fs["final_nodes"],"total_merges":fs["total_merges"],
            "removed_fraction":fs["removed_fraction"],"microsteps":fs["microsteps"],
            "initial_edges":int(first["superedges"]),"initial_allowed":int(first["merge_allowed"]),
            "initial_allowed_fraction":float(first["merge_allowed_fraction"]),
            "terminal_edges":int(last["superedges"]),"terminal_allowed":int(last["merge_allowed"]),
            "terminal_allowed_fraction":float(last["merge_allowed_fraction"]),
            "initial_mechanics_available_fraction":float(first["mechanics_available_local_fraction"]),
            "terminal_mechanics_available_fraction":float(last["mechanics_available_local_fraction"]),
            "natural_exhaustion":bool(fs["natural_exhaustion"]),"stop_reason":fs["stop_reason"],
            "accounting_residual":int(fs["level0_cells"]-fs["total_merges"]-fs["final_nodes"]),
        })
    audit=pd.DataFrame(summaries)
    assert (audit.accounting_residual==0).all(), audit[["sample","accounting_residual"]]
    assert (audit.terminal_allowed==0).all()
    assert audit.natural_exhaustion.all()
    assert (audit.stop_reason=="no_contextually_admissible_boundaries").all()
    assert (~cal.pooled_across_specimens.astype(bool)).all()
    assert cal.specimen_local.astype(bool).all()
    assert (~cal.later_level_rescaling.astype(bool)).all()

    cal.to_csv(src/"panel_A_calibration.csv",index=False)
    audit.to_csv(src/"construction_audit.csv",index=False)

    # Compact step table with only plotted variables.
    flow=[]
    for s,st in allsteps.items():
        z=st.copy()
        u=z["hierarchy_coordinate_removed_fraction"].astype(float)
        for i,r in z.iterrows():
            flow.append({
                "sample":s, "hierarchy_progress":float(u.loc[i]),
                "candidate_interfaces":int(r["superedges"]),
                "admissible_interfaces":int(r["merge_allowed"]),
                "admissible_fraction":float(r["merge_allowed_fraction"]),
                "evidence_mean":float(r["evidence_mean"]),
                "geometry_reliability_mean":float(r["geom_reliability_mean"]),
                "mechanics_reliability_mean":float(r["mech_reliability_mean"]),
                "mechanics_available_fraction":float(r["mechanics_available_local_fraction"]),
            })
    flow=pd.DataFrame(flow)
    flow.to_csv(src/"panel_C_flow.csv",index=False)

    # B source: first-step relational summaries.
    b=[]
    for s,st in allsteps.items():
        r=st.iloc[0]
        b.append({"sample":s,"evidence_mean":r["evidence_mean"],
                  "geometry_reliability_mean":r["geom_reliability_mean"],
                  "mechanics_reliability_mean":r["mech_reliability_mean"],
                  "mechanics_available_fraction":r["mechanics_available_local_fraction"]})
    b=pd.DataFrame(b)
    b.to_csv(src/"panel_B_level0_evidence.csv",index=False)

    x=organ_gap_x()
    xt=[LABELS[s] for s in SAMPLES]

    # A: grouped multicategory bars; raw heterogeneous scales shown on log y-axis.
    fig,ax=plt.subplots(figsize=(6.65,3.15))
    vals=[
        ("Molecular","scale_molecular"),("Geometry","scale_geometry"),
        ("Mechanically informed","scale_mechanics"),
        ("Communication support","scale_communication_support"),
        ("Initial threshold","initial_threshold"),("Maximum threshold","maximum_threshold")
    ]
    width=.12
    offsets=(np.arange(len(vals))-(len(vals)-1)/2)*width
    for off,(lab,col) in zip(offsets,vals):
        ax.bar(x+off,cal[col].astype(float),width=width*.92,label=lab)
    ax.set_yscale("log")
    ax.set_ylabel("Specimen-local scale or threshold")
    short=["Healthy\nbrain","Alzheimer's","GBM","Nondiseased\nkidney","PRCC"]
    ax.set_xticks(x,short)
    ax.axvline(3,color="0.85",lw=.8)
    ax.text(1,1.015,"Brain",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.text(4.5,1.015,"Kidney",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.legend(frameon=False,ncol=3,loc="lower center",bbox_to_anchor=(.5,1.18),
              columnspacing=1.0,handletextpad=.4)
    style_axis(ax); panel_letter(ax,"A")
    fig.subplots_adjust(left=.13,right=.98,bottom=.22,top=.68)
    save(fig,out,"SuppFig1A_specimen_local_calibration")

    # B: grouped multicategory bars for bounded Level-0 summaries.
    fig,ax=plt.subplots(figsize=(6.65,2.95))
    metrics=[("Evidence","evidence_mean"),("Geometry reliability","geometry_reliability_mean"),
             ("Mechanics reliability","mechanics_reliability_mean"),
             ("Mechanics available","mechanics_available_fraction")]
    width=.16
    offsets=(np.arange(len(metrics))-(len(metrics)-1)/2)*width
    for off,(lab,col) in zip(offsets,metrics):
        ax.bar(x+off,b[col].astype(float),width=width*.92,label=lab)
    ax.set_ylim(0,1.04)
    ax.set_ylabel("Level-0 relational evidence / availability")
    short=["Healthy\nbrain","Alzheimer's","GBM","Nondiseased\nkidney","PRCC"]
    ax.set_xticks(x,short)
    ax.axvline(3,color="0.85",lw=.8)
    ax.text(1,1.015,"Brain",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.text(4.5,1.015,"Kidney",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.legend(frameon=False,ncol=4,loc="lower center",bbox_to_anchor=(.5,1.18),
              columnspacing=.9,handletextpad=.35)
    style_axis(ax); panel_letter(ax,"B")
    fig.subplots_adjust(left=.13,right=.98,bottom=.22,top=.69)
    save(fig,out,"SuppFig1B_level0_relational_evidence")

    # C: two aligned axes in one standalone panel; brain/kidney kept as separate rows.
    fig=plt.figure(figsize=(6.65,4.4))
    gs=fig.add_gridspec(2,1,hspace=.48,left=.12,right=.98,bottom=.13,top=.88)
    for row,organ in enumerate(["Brain","Kidney"]):
        ax=fig.add_subplot(gs[row,0])
        for s in [q for q in SAMPLES if ORGANS[q]==organ]:
            z=flow[flow["sample"]==s]
            # Normalize counts to Level-0 candidate count to make specimens comparable.
            c=z["candidate_interfaces"]/z["candidate_interfaces"].iloc[0]
            a_=z["admissible_interfaces"]/z["candidate_interfaces"].iloc[0]
            ax.plot(z["hierarchy_progress"],c,lw=1.35,label=f"{LABELS[s]} candidates")
            ax.plot(z["hierarchy_progress"],a_,lw=1.1,ls="--",label=f"{LABELS[s]} admissible")
        ax.set_xlim(0,0.85); ax.set_ylim(bottom=0)
        ax.set_ylabel("Fraction of Level-0\ncandidate interfaces")
        ax.text(.01,1.035,organ,transform=ax.transAxes,ha="left",va="bottom",fontweight="bold",fontsize=8,clip_on=False)
        ax.legend(frameon=False,ncol=2 if organ=="Kidney" else 3,loc="upper center",
                  bbox_to_anchor=(.55,1.02),fontsize=6.3,columnspacing=.8,handlelength=1.6)
        style_axis(ax)
        if row==1: ax.set_xlabel("Hierarchy progress, u")
        else: ax.tick_params(labelbottom=False)
        if row==0: panel_letter(ax,"C")
    save(fig,out,"SuppFig1C_candidate_admissible_flow")

    # D exact accounting: stacked merges + terminal = Level0, with residual displayed.
    fig,ax=plt.subplots(figsize=(6.65,3.0))
    term=audit["final_nodes"].to_numpy()
    merges=audit["total_merges"].to_numpy()
    ax.bar(x,term,width=.62,label="Terminal objects")
    ax.bar(x,merges,bottom=term,width=.62,label="Accepted merges")
    for xi,n0 in zip(x,audit["level0_cells"]):
        ax.text(xi,n0*1.02,f"{int(n0):,}",ha="center",va="bottom",fontsize=6.5)
    ax.set_ylabel("Number of objects"); ax.set_xticks(x,["Healthy\nbrain","Alzheimer\'s","GBM","Nondiseased\nkidney","PRCC"])
    ax.axvline(3,color="0.85",lw=.8)
    ax.legend(frameon=False,ncol=2,loc="upper left",bbox_to_anchor=(0,1.43))
    ax.text(1,1.015,"Brain",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.text(4.5,1.015,"Kidney",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.text(.99,1.30,r"$N_0-M-N_{\mathrm{terminal}}=0$ for all five specimens",transform=ax.transAxes,ha="right",va="bottom",fontsize=7.2)
    style_axis(ax); panel_letter(ax,"D"); fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.63)
    save(fig,out,"SuppFig1D_exact_ledger_accounting")

    # E natural exhaustion. Candidate bars; explicit zero marker/label for admissible.
    fig,ax=plt.subplots(figsize=(6.65,3.0))
    cand=audit["terminal_edges"].to_numpy()
    ax.bar(x,cand,width=.62,label="Candidate interfaces")
    ax.scatter(x,np.zeros_like(x),marker="x",s=72,linewidths=2.2,zorder=8,clip_on=False,label="Contextually admissible = 0")
    for xi,v in zip(x,cand):
        ax.text(xi,v*1.025,f"{int(v):,}",ha="center",va="bottom",fontsize=6.5)
        ax.text(xi,max(cand)*.022,"0",ha="center",va="bottom",fontsize=7.5,fontweight="bold",zorder=9)
    ax.set_ylim(-max(cand)*0.035, max(cand)*1.12)
    ax.set_ylabel("Number of interfaces at termination"); ax.set_xticks(x,["Healthy\nbrain","Alzheimer\'s","GBM","Nondiseased\nkidney","PRCC"])
    ax.axvline(3,color="0.85",lw=.8)
    ax.legend(frameon=False,ncol=2,loc="upper left",bbox_to_anchor=(0,1.43))
    ax.text(1,1.015,"Brain",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.text(4.5,1.015,"Kidney",transform=ax.get_xaxis_transform(),ha="center",va="bottom",fontsize=7.5)
    ax.text(.99,1.30,"Natural exhaustion in all five specimens",transform=ax.transAxes,ha="right",va="bottom",fontsize=7.2)
    style_axis(ax); panel_letter(ax,"E"); fig.subplots_adjust(left=.13,right=.98,bottom=.23,top=.63)
    save(fig,out,"SuppFig1E_natural_exhaustion")

    manifest={
        "figure":"Supplementary Figure 1",
        "mode":"standalone_panels",
        "panels":["A","B","C","D","E"],
        "assembled_figure":False,
        "hierarchy_rerun":False,
        "scientific_recalculation":False,
        "display_revision":"V1.4 final visual repair: C organ labels outside data field; E zero-admissibility markers given unclipped baseline clearance; A/B grouped bars retained",
        "source":"frozen v0911 specimen-local contextual-flow hierarchy",
        "validated":{
            "accounting_residual_zero_all":True,
            "terminal_admissible_zero_all":True,
            "natural_exhaustion_all":True,
            "stop_reason_all":"no_contextually_admissible_boundaries",
            "specimen_local_calibration_all":True,
            "pooled_across_specimens_all":False,
            "later_level_rescaling_all":False,
        }
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print("PASS — SuppFig1 validation and standalone rendering")
    print(audit[["sample","level0_cells","total_merges","final_nodes","accounting_residual",
                 "terminal_edges","terminal_allowed","natural_exhaustion"]].to_string(index=False))
    print("OUTPUT",out)

if __name__=="__main__":
    main()
