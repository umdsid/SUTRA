
from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd

STATUS_MAP={"PASS":1.0,"HOLD":0.0}

def load_csv(p:Path):
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_csv(p)

def normalize_status(s:pd.Series):
    return s.astype(str).str.upper().map(STATUS_MAP)

def exact_join_support(trajectory:pd.DataFrame,replay:pd.DataFrame):
    """
    Join reconstructed frozen-rule support states to the repaired trajectory by
    exact (evaluation, nodes) keys. No nearest matching or interpolation.
    """
    t=trajectory.copy()
    r=replay.copy()

    te="eval" if "eval" in t.columns else "evaluation_index"
    tn="nodes"
    re="evaluation_index"
    rn="nodes"

    for c in [te,tn]:
        t[c]=pd.to_numeric(t[c],errors="coerce")
    for c in [re,rn]:
        r[c]=pd.to_numeric(r[c],errors="coerce")

    keep=[re,rn,"mass_status","expr_status","spatial_status"]
    r=r[keep].copy()
    r["mass_support_exact"]=normalize_status(r["mass_status"])
    r["expression_support_exact"]=normalize_status(r["expr_status"])
    r["spatial_support_exact"]=normalize_status(r["spatial_status"])

    if r.duplicated([re,rn]).any():
        raise RuntimeError("Replay contains duplicate exact evaluation/node keys.")
    if t.duplicated([te,tn]).any():
        raise RuntimeError("Trajectory contains duplicate exact evaluation/node keys.")

    out=t.merge(
        r[[re,rn,"mass_support_exact","expression_support_exact","spatial_support_exact"]],
        left_on=[te,tn],right_on=[re,rn],how="left",validate="one_to_one"
    )
    if re != te and re in out.columns:
        out=out.drop(columns=[re])
    if rn != tn and rn in out.columns:
        out=out.drop(columns=[rn])

    cov={
        "mass":float(out["mass_support_exact"].notna().mean()),
        "expr":float(out["expression_support_exact"].notna().mean()),
        "spatial":float(out["spatial_support_exact"].notna().mean())
    }
    return out,cov

def recompute_basin_supports(basins:pd.DataFrame,traj:pd.DataFrame):
    """
    Preserve v1.1.1.1 basin geometry exactly; only replace the previously missing
    support summaries with the exact frozen-rule replay.
    """
    out=basins.copy()
    if "entry_eval" not in out or "exit_eval" not in out:
        raise RuntimeError("Repaired basin table lacks entry_eval/exit_eval.")

    vals=[]
    for _,b in out.iterrows():
        lo=float(b["entry_eval"]); hi=float(b["exit_eval"])
        x=traj[(traj["eval"]>=lo)&(traj["eval"]<=hi)]
        def mean(c):
            a=pd.to_numeric(x[c],errors="coerce").to_numpy(float)
            return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan
        vals.append({
            "mass_support":mean("mass_support_exact"),
            "expression_support":mean("expression_support_exact"),
            "spatial_support":mean("spatial_support_exact"),
            "support_evaluations":int(len(x)),
            "all_three_pass_fraction":float(np.mean(
                (x["mass_support_exact"]==1.0)&
                (x["expression_support_exact"]==1.0)&
                (x["spatial_support_exact"]==1.0)
            )) if len(x) else np.nan
        })
    v=pd.DataFrame(vals,index=out.index)
    for c in v.columns:
        out[c]=v[c]
    return out

def pareto_layer1(df:pd.DataFrame,objectives):
    """
    Exact nondominance under the already-frozen objective list.
    Missing objectives are never imputed.
    """
    if df.empty:
        z=df.copy(); z["pareto_layer"]=pd.Series(dtype=int); return z
    X=df[list(objectives)].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    valid=np.all(np.isfinite(X),axis=1)
    front=np.zeros(len(df),dtype=bool)
    ids=np.where(valid)[0]
    for i in ids:
        dominated=False
        for j in ids:
            if i==j: continue
            if np.all(X[j]>=X[i]) and np.any(X[j]>X[i]):
                dominated=True; break
        front[i]=not dominated
    out=df.copy()
    out["pareto_layer"]=np.where(front,1,2)
    out["pareto_eligible"]=valid
    return out

def load_strict_historical(project:Path,sample:str,cfg):
    # Prefer the already-certified strict v1.1.1.3 export.
    p=project/"results"/cfg["exact_replay_stage"]/sample/"strict_historical_landmarks.json"
    if not p.exists():
        raise FileNotFoundError(p)
    obj=json.loads(p.read_text())
    nodes=[int(x) for x in obj.get("nodes",[]) if int(x)>=20]
    return nodes,obj.get("sources",[])

def historical_alignment(front:pd.DataFrame,old_nodes):
    rows=[]
    if not old_nodes: return pd.DataFrame(rows)
    old=np.asarray(old_nodes,float)
    for _,r in front.iterrows():
        n=float(r["center_nodes"])
        j=int(np.argmin(np.abs(np.log(old/n))))
        o=float(old[j])
        rows.append({
            "center_eval":int(r["center_eval"]),
            "candidate_nodes":int(n),
            "nearest_historical_nodes":int(o),
            "absolute_node_difference":int(abs(n-o)),
            "relative_node_difference":float(abs(n-o)/o),
            "absolute_log_scale_distance":float(abs(math.log(n/o))),
            "exact_historical_match":bool(int(n)==int(o))
        })
    return pd.DataFrame(rows)

def analyze(project:Path,sample:str,cfg,outdir:Path):
    bdir=project/"results"/cfg["repaired_landscape_stage"]/sample
    rdir=project/"results"/cfg["exact_replay_stage"]/sample

    traj=load_csv(bdir/"normalized_evaluation_trajectory_repaired.csv")
    bas=load_csv(bdir/"persistence_basins_repaired.csv")
    rep=load_csv(rdir/"exact_frozen_terminal_replay.csv")

    joined,cov=exact_join_support(traj,rep)
    if any(v<cfg["minimum_exact_support_coverage"] for v in cov.values()):
        raise RuntimeError(f"{sample}: exact replay coverage below requirement: {cov}")

    supported=recompute_basin_supports(bas,joined)

    # Remove any old Pareto labels produced when support was unresolved.
    for c in ["pareto_layer","pareto_eligible"]:
        if c in supported.columns:
            supported=supported.drop(columns=[c])
    ranked=pareto_layer1(supported,cfg["pareto_objectives"])
    front=ranked[(ranked["pareto_layer"]==1)&(ranked["pareto_eligible"])].copy()
    front=front.sort_values("center_eval").reset_index(drop=True)

    old,sources=load_strict_historical(project,sample,cfg)
    align=historical_alignment(front,old)

    sdir=outdir/sample
    sdir.mkdir(parents=True,exist_ok=True)
    joined.to_csv(sdir/"supported_evaluation_trajectory.csv",index=False)
    ranked.to_csv(sdir/"supported_persistence_basins.csv",index=False)
    front.to_csv(sdir/"final_supported_pareto_landmarks.csv",index=False)
    align.to_csv(sdir/"final_historical_alignment.csv",index=False)
    (sdir/"historical_sources.json").write_text(json.dumps({
        "sample":sample,"nodes":old,"sources":sources
    },indent=2))

    # Summary metrics that are descriptive, not gates.
    exact_matches=int(align["exact_historical_match"].sum()) if len(align) else 0
    median_dist=float(align["absolute_log_scale_distance"].median()) if len(align) else np.nan
    all_pass=[]
    for _,r in front.iterrows():
        all_pass.append(float(r.get("all_three_pass_fraction",np.nan)))

    status="PASS" if len(front)>0 and all(v>=cfg["minimum_exact_support_coverage"] for v in cov.values()) else "HOLD"
    report={
        "sample":sample,
        "evaluation_support_coverage":cov,
        "repaired_basins":int(len(ranked)),
        "final_supported_landmarks":int(len(front)),
        "final_supported_nodes":[int(x) for x in front["center_nodes"].tolist()],
        "historical_nodes":old,
        "exact_historical_matches":exact_matches,
        "median_absolute_log_scale_distance_to_history":median_dist,
        "mean_all_three_pass_fraction_over_final_basins":
            float(np.nanmean(all_pass)) if len(all_pass) and np.isfinite(all_pass).any() else None,
        "mass_is_existence_veto":False,
        "mass_is_pareto_objective":False,
        "pareto_objectives":cfg["pareto_objectives"],
        "status":status
    }
    (sdir/"final_supported_hierarchy_report.json").write_text(json.dumps(report,indent=2))
    return report
