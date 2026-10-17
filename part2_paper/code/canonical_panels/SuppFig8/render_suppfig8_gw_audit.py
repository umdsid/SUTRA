#!/usr/bin/env python3
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def paths(root, disease):
    b = root / disease
    return {
        "geo": b/"gw_geodesic_v2",
        "ms": b/"gw_multistart_dense_v5",
    }

def load(root, disease):
    p = paths(root, disease)
    return {
        "traj": pd.read_csv(p["geo"]/"gw_scale_trajectory.csv"),
        "conv": pd.read_csv(p["geo"]/"gw_K_convergence.csv"),
        "quant": pd.read_csv(p["geo"]/"metric_quantization_audit.csv"),
        "runs": pd.read_csv(p["ms"]/"multistart_all_runs.csv"),
        "sum": pd.read_csv(p["ms"]/"multistart_scale_summary.csv"),
    }

def find_results(sutra):
    candidates = [
        sutra/"audits"/"BRAIN_GW_AND_ORGANIZATIONAL_FAMILY_V1_6_5"/"brain_gw_results",
        sutra/"audits"/"BRAIN_GW_AND_ORGANIZATIONAL_FAMILY_V1_6_10"/"brain_gw_results",
    ]
    for p in candidates:
        if (p/"alzheimers").is_dir() and (p/"gbm_reference_addon").is_dir():
            return p
    hits = list((sutra/"audits").glob("BRAIN_GW_AND_ORGANIZATIONAL_FAMILY*/brain_gw_results"))
    for p in sorted(hits, reverse=True):
        if (p/"alzheimers").is_dir() and (p/"gbm_reference_addon").is_dir():
            return p
    raise FileNotFoundError("Frozen brain GW result tree not found.")

def validate(d):
    assert len(d["conv"]) == 25
    assert len(d["sum"]) == 25
    assert len(d["runs"]) == 325
    assert set(map(str, [64,128,256,512])).issubset(set(d["conv"].columns.astype(str)))

def summary(d):
    c=d["conv"]
    rel=np.abs(c["256"]-c["512"])/np.maximum(np.abs(c["512"]),1e-15)
    r=d["runs"]
    t=d["traj"]
    q=d["quant"]
    masses={}
    for sample,g in q.groupby("sample"):
        masses[str(sample)] = float(g["dominant_component_mass_fraction"].median())
    return {
        "n_states":25,
        "multistart_runs":325,
        "K256_to_K512_relative_change_median":float(np.median(rel)),
        "K256_to_K512_relative_change_max":float(np.max(rel)),
        "max_row_marginal_error":float(r["row_marginal_error"].abs().max()),
        "max_column_marginal_error":float(r["column_marginal_error"].abs().max()),
        "max_coupling_mass_error":float((r["coupling_mass"]-1).abs().max()),
        "max_multistart_objective_audit_error":float(r["objective_delta"].abs().max()),
        "max_geodesic_objective_audit_error":float(t["objective_audit_delta"].abs().max()),
        "principal_domain_mass":masses,
    }

def setup(ax, letter, title):
    ax.text(-0.12,1.07,letter,transform=ax.transAxes,fontsize=17,fontweight="bold",va="top")
    ax.set_title(title, fontsize=13)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sutra-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results-root", default=None)
    a=ap.parse_args()
    sutra=Path(a.sutra_root).expanduser().resolve()
    out=Path(a.out).expanduser().resolve(); out.mkdir(parents=True,exist_ok=True)
    rr=Path(a.results_root).expanduser().resolve() if a.results_root else find_results(sutra)
    print("Frozen results:", rr)
    ad=load(rr,"alzheimers"); gb=load(rr,"gbm_reference_addon")
    validate(ad); validate(gb)
    sm={"alzheimers":summary(ad),"gbm_reference_addon":summary(gb)}
    (out/"SUPPFIG8_NUMERICAL_SUMMARY.json").write_text(json.dumps(sm,indent=2))

    # Four-panel composition: A/B top, C/D bottom. Connected-domain accounting
    # is intentionally omitted; invariant masses remain in JSON/caption.
    fig,axs=plt.subplots(2,2,figsize=(12.2,9.0))
    ax=axs[0,0]; setup(ax,"a","Landmark-resolution convergence")
    u=ad["conv"]["u_target"]
    for k,m in zip(["64","128","256","512"],["o","s","^","D"]):
        ax.plot(u,ad["conv"][k],marker=m,ms=3.5,lw=1.3,label=f"Alzheimer’s, K={k}")
        ax.plot(u,gb["conv"][k],marker=m,ms=3.5,lw=1.3,ls="--",alpha=.72,label=f"GBM-related, K={k}")
    ax.set_xlabel("Organizational coordinate, $u$"); ax.set_ylabel("GW distance")
    ax.legend(fontsize=7,ncol=2,frameon=False,loc="upper left")

    ax=axs[0,1]; setup(ax,"b","Highest-resolution change")
    for d,label in [(ad,"Alzheimer’s"),(gb,"GBM-related")]:
        c=d["conv"]; rel=100*np.abs(c["256"]-c["512"])/np.maximum(np.abs(c["512"]),1e-15)
        ax.plot(c["u_target"],rel,marker="o",ms=3.5,lw=1.5,label=label)
    ax.axhline(10,ls=":",lw=1,color="0.45")
    ax.set_xlabel("Organizational coordinate, $u$")
    ax.set_ylabel(r"$|K256-K512|/K512$ (%)")
    ax.legend(frameon=False,fontsize=8)

    ax=axs[1,0]; setup(ax,"c","Multistart optimization sensitivity")
    for d,label in [(ad,"Alzheimer’s"),(gb,"GBM-related")]:
        s=d["sum"]
        ax.plot(s["u_target"],s["independent_minus_best_gw"],marker="o",ms=3.5,lw=1.5,label=label)
    ax.axhline(0,ls=":",lw=1,color="0.45")
    ax.set_xlabel("Organizational coordinate, $u$")
    ax.set_ylabel("Independent-start − best GW distance")
    ax.legend(frameon=False,fontsize=8)

    ax=axs[1,1]; setup(ax,"d","Transport and objective numerical certificate")
    cats=["row marginal","column marginal","coupling mass","objective audit","geodesic objective"]
    x=np.arange(len(cats))
    def errs(d):
        r=d["runs"]; t=d["traj"]
        return np.array([
            r["row_marginal_error"].abs().max(),
            r["column_marginal_error"].abs().max(),
            (r["coupling_mass"]-1).abs().max(),
            r["objective_delta"].abs().max(),
            t["objective_audit_delta"].abs().max()
        ])
    ax.scatter(x-.08,np.maximum(errs(ad),1e-18),s=35,label="Alzheimer’s")
    ax.scatter(x+.08,np.maximum(errs(gb),1e-18),s=35,label="GBM-related")
    ax.axhline(1e-12,ls=":",lw=1,color="0.45")
    ax.set_yscale("log"); ax.set_ylim(2e-16,1.5e-12)
    ax.set_xticks(x,cats,rotation=18,ha="right")
    ax.set_ylabel("Maximum absolute numerical error")
    ax.legend(frameon=False,fontsize=8,loc="lower right")

    fig.suptitle("GW numerical robustness across independent brain comparisons",fontsize=15,y=.995)
    fig.tight_layout(rect=[0,0,1,.975])
    pdf=out/"SUTRA_Supplementary_Figure_8_GW_numerical_audit.pdf"
    png=out/"SUTRA_Supplementary_Figure_8_GW_numerical_audit.png"
    fig.savefig(pdf,bbox_inches="tight")
    fig.savefig(png,dpi=450,bbox_inches="tight")
    plt.close(fig)
    print("WROTE",pdf); print("WROTE",png)
    print(json.dumps(sm,indent=2))

if __name__=="__main__":
    main()
