
from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd

def _suffix(c): return str(c).split(".")[-1]

def _status_to_float(v):
    if v is None: return np.nan
    try:
        if pd.isna(v): return np.nan
    except Exception:
        pass
    s=str(v).strip().upper()
    if s in {"PASS","TRUE","YES","Y","1","STABLE"}: return 1.0
    if s in {"HOLD","FAIL","FALSE","NO","N","0","UNSTABLE"}: return 0.0
    try:
        f=float(v)
        if f in (0.0,1.0): return f
    except Exception:
        pass
    return np.nan

def _resolve_support_column(df: pd.DataFrame, kind: str, aliases: list[str]):
    # Exact/suffix aliases first.
    for a in aliases:
        if a in df.columns: return a
    for a in aliases:
        hits=[c for c in df.columns if _suffix(c)==a]
        if len(hits)==1: return hits[0]

    # Then semantic scoring. Prefer explicit gate/status/stability fields and
    # fields that actually contain PASS/HOLD-like values.
    key_terms = {
        "mass": ["mass"],
        "expr": ["expr","expression"],
        "spatial": ["spatial","geometry","geometric"]
    }[kind]
    status_terms=["status","gate","stable","stability","pass","ok","support"]
    candidates=[]
    for c in df.columns:
        low=str(c).lower()
        if not any(t in low for t in key_terms): continue
        score=0
        if any(t in low for t in status_terms): score += 5
        suf=_suffix(c).lower()
        if suf in aliases: score += 10
        vals=df[c].dropna().astype(str).str.upper().head(100)
        if len(vals):
            frac=vals.isin(["PASS","HOLD","FAIL","TRUE","FALSE","YES","NO","0","1","STABLE","UNSTABLE"]).mean()
            score += 8*float(frac)
        # Penalize obvious continuous summary/statistic fields.
        if any(t in low for t in ["mean","median","gini","k50","k80","k90","neff",
                                  "effective","entropy","ratio","fraction","max","min","sd","std"]):
            score -= 4
        candidates.append((score,c))
    candidates.sort(reverse=True,key=lambda x:x[0])
    return candidates[0][1] if candidates and candidates[0][0] > 0 else None

def _first_col(df, names):
    for n in names:
        if n in df.columns: return n
    for n in names:
        hits=[c for c in df.columns if _suffix(c)==n]
        if len(hits)==1: return hits[0]
    return None

def load_source(project: Path, sample: str, cfg):
    base=project/"results"/cfg["source_stage"]/sample
    p=base/"scientific_evaluations.parquet"
    if not p.exists():
        hits=list(base.rglob("*scientific*evaluation*.parquet"))
        if not hits: hits=list(base.rglob("*.parquet"))
        if not hits:
            raise RuntimeError(f"{sample}: no v1.1.0 scientific evaluation parquet under {base}")
        p=max(hits,key=lambda q:q.stat().st_size)
    df=pd.read_parquet(p)
    return p,df

def normalize_source(df: pd.DataFrame, cfg):
    ec=_first_col(df,["eval","evaluation","evaluation_index","scientific_evaluation"])
    nc=_first_col(df,["nodes","n_nodes","node_count","active_nodes"])
    if ec is None or nc is None:
        raise RuntimeError(f"evaluation/node columns unresolved; columns={list(df.columns)}")
    out=pd.DataFrame({
        "eval":pd.to_numeric(df[ec],errors="coerce"),
        "nodes":pd.to_numeric(df[nc],errors="coerce")
    })
    ellc=_first_col(df,["ell","scale","scale_log","log_scale"])
    if ellc:
        out["ell"]=pd.to_numeric(df[ellc],errors="coerce")
    else:
        n0=float(np.nanmax(out["nodes"]))
        out["ell"]=np.log(np.maximum(n0,1.0)/np.maximum(out["nodes"].astype(float),1.0))

    support_sources={}
    for kind in ["mass","expr","spatial"]:
        src=_resolve_support_column(df,kind,cfg["support_aliases"][kind])
        support_sources[kind]=src
        out[kind]=df[src].map(_status_to_float) if src is not None else np.nan

    speed_cols=[]
    for c in df.columns:
        s=_suffix(c)
        if any(s.startswith(pref) for pref in cfg["block_speed_prefixes"]):
            speed_cols.append(c)
    for c in speed_cols:
        out[_suffix(c)]=pd.to_numeric(df[c],errors="coerce")

    out=out.dropna(subset=["eval","nodes"]).sort_values("eval").drop_duplicates("eval",keep="last").reset_index(drop=True)
    schema={
        "source_columns":[str(c) for c in df.columns],
        "support_sources":{k:(str(v) if v is not None else None) for k,v in support_sources.items()},
        "speed_columns":[_suffix(c) for c in speed_cols],
        "support_coverage":{
            k:float(out[k].notna().mean()) for k in ["mass","expr","spatial"]
        }
    }
    return out,schema

def robust_z(x):
    x=np.asarray(x,float)
    med=np.nanmedian(x)
    mad=np.nanmedian(np.abs(x-med))
    if not np.isfinite(mad) or mad<1e-12:
        sd=np.nanstd(x)
        scale=sd if np.isfinite(sd) and sd>1e-12 else 1.0
    else:
        scale=1.4826*mad
    return (x-med)/scale

def activity_state(df,cfg):
    cs=[c for c in df.columns if any(c.startswith(p) for p in cfg["block_speed_prefixes"])]
    if not cs:
        raise RuntimeError("no block-speed columns resolved")
    A=np.column_stack([np.abs(robust_z(df[c].to_numpy(float))) for c in cs])
    finite=np.isfinite(A)
    nfin=finite.sum(axis=1)
    sq=np.where(finite,A*A,0.0).sum(axis=1)
    activity=np.full(len(df),np.nan)
    good=nfin>0
    activity[good]=np.sqrt(sq[good]/nfin[good])

    coh=np.full(len(df),np.nan)
    coh[good]=(np.where(finite,A<=1.0,False).sum(axis=1)[good]/nfin[good])
    return activity,nfin,coh,cs

def trailing_median(x,w):
    return pd.Series(x,dtype=float).rolling(w,min_periods=1).median().to_numpy()

def _is_local_min(y,i,r):
    if not np.isfinite(y[i]): return False
    a=max(0,i-r); b=min(len(y),i+r+1)
    return y[i] <= np.nanmin(y[a:b])

def _is_local_max(y,i,r):
    if not np.isfinite(y[i]): return False
    a=max(0,i-r); b=min(len(y),i+r+1)
    return y[i] >= np.nanmax(y[a:b])

def detect_local_basins(df,cfg):
    activity,nfin,coh,cs=activity_state(df,cfg)
    sm=trailing_median(activity,cfg["smoothing_window"])
    r=int(cfg["local_extremum_radius"])
    mins=[i for i in range(r,len(sm)-r) if _is_local_min(sm,i,r)]
    maxs=[i for i in range(r,len(sm)-r) if _is_local_max(sm,i,r)]

    rows=[]
    for i in mins:
        left=[j for j in maxs if j<i and np.isfinite(sm[j]) and sm[j] > sm[i]]
        right=[j for j in maxs if j>i and np.isfinite(sm[j]) and sm[j] > sm[i]]
        # A proper persistence basin must be enclosed by observed shoulders on both sides.
        if not left or not right:
            continue
        l=max(left); rr=min(right)
        lifetime=rr-l+1
        if lifetime < int(cfg["minimum_basin_lifetime_evals"]):
            continue
        shoulder=min(sm[l],sm[rr])
        depth=max(float(shoulder-sm[i]),0.0)
        if depth<=0: continue
        sl=slice(l,rr+1)
        def support(c):
            x=df[c].to_numpy(float)[sl]
            return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan
        rows.append({
            "center_eval":int(df.loc[i,"eval"]),
            "center_nodes":int(df.loc[i,"nodes"]),
            "entry_eval":int(df.loc[l,"eval"]),
            "exit_eval":int(df.loc[rr,"eval"]),
            "entry_nodes":int(df.loc[l,"nodes"]),
            "exit_nodes":int(df.loc[rr,"nodes"]),
            "lifetime":int(lifetime),
            "depth":depth,
            "activity_min":float(sm[i]),
            "left_shoulder_activity":float(sm[l]),
            "right_shoulder_activity":float(sm[rr]),
            "block_coherence":float(np.nanmean(coh[sl])),
            "resolved_block_median":float(np.nanmedian(nfin[sl])),
            "expression_support":support("expr"),
            "spatial_support":support("spatial"),
            "mass_support":support("mass"),
            "speed_block_count":len(cs)
        })

    # If equal/plateau minima create duplicates inside the same shoulder pair,
    # retain the deepest center, then the one nearest the basin midpoint.
    if not rows:
        return pd.DataFrame()
    x=pd.DataFrame(rows)
    kept=[]
    for (_,g) in x.groupby(["entry_eval","exit_eval"],sort=False):
        g=g.copy()
        mid=(g["entry_eval"].iloc[0]+g["exit_eval"].iloc[0])/2
        g["_mid_dist"]=(g["center_eval"]-mid).abs()
        g=g.sort_values(["depth","_mid_dist"],ascending=[False,True])
        kept.append(g.iloc[0].drop(labels="_mid_dist"))
    return pd.DataFrame(kept).sort_values("center_eval").reset_index(drop=True)

def pareto_layer1(df,cfg):
    if df.empty:
        out=df.copy(); out["pareto_layer"]=pd.Series(dtype=int); return out
    objs=cfg["pareto_objectives"]
    V=[]
    for _,r in df.iterrows():
        row=[]
        for c in objs:
            v=r[c]
            row.append(-np.inf if pd.isna(v) else float(v))
        V.append(row)
    V=np.asarray(V,float)
    front=np.ones(len(V),bool)
    for i in range(len(V)):
        for j in range(len(V)):
            if i==j: continue
            if np.all(V[j]>=V[i]) and np.any(V[j]>V[i]):
                front[i]=False; break
    out=df.copy()
    out["pareto_layer"]=np.where(front,1,2)
    return out

def _extract_nodes_from_obj(obj,sample):
    vals=[]
    def walk(x,key=""):
        if isinstance(x,dict):
            # Prefer sample-specific branches when possible, but recurse all.
            for k,v in x.items(): walk(v,str(k))
        elif isinstance(x,list):
            if any(t in key.lower() for t in ["node","landmark","pareto"]):
                for v in x:
                    if isinstance(v,(int,float)) and float(v)>=2:
                        vals.append(int(round(float(v))))
            for v in x:
                if isinstance(v,(dict,list)): walk(v,key)
        elif isinstance(x,(int,float)):
            if any(t in key.lower() for t in ["node","landmark"]) and float(x)>=2:
                vals.append(int(round(float(x))))
    walk(obj)
    return vals

def discover_previous_landmarks(project:Path,sample:str,cfg):
    records=[]
    for stage in cfg["previous_landmark_stages"]:
        base=project/"results"/stage
        if not base.exists(): continue
        nodes=set()
        sources=[]
        for p in base.rglob("*"):
            if not p.is_file(): continue
            low=p.name.lower()
            if not any(t in low for t in ["landmark","pareto","basin","certificate","report"]): continue
            try:
                if p.suffix.lower()==".json":
                    obj=json.loads(p.read_text())
                    txt=json.dumps(obj)
                    if sample not in txt and sample not in str(p): continue
                    for n in _extract_nodes_from_obj(obj,sample): nodes.add(n)
                    sources.append(str(p))
                elif p.suffix.lower() in {".csv",".tsv"}:
                    d=pd.read_csv(p,sep="\t" if p.suffix.lower()==".tsv" else ",")
                    if "sample" in d.columns:
                        d=d[d["sample"].astype(str)==sample]
                    elif sample not in str(p):
                        continue
                    cols=[c for c in d.columns if "node" in c.lower()]
                    for c in cols:
                        for v in pd.to_numeric(d[c],errors="coerce").dropna():
                            if v>=2: nodes.add(int(round(v)))
                    sources.append(str(p))
            except Exception:
                continue
        if nodes:
            records.append({"stage":stage,"nodes":sorted(nodes,reverse=True),"sources":sources[:20]})
    # Use the first available (newest preferred by config order) as canonical comparison.
    return records

def compare_to_previous(front:pd.DataFrame, previous_records):
    if front.empty or not previous_records:
        return pd.DataFrame()
    prev=previous_records[0]
    old=np.asarray(prev["nodes"],float)
    rows=[]
    for _,r in front.iterrows():
        n=float(r["center_nodes"])
        k=int(np.argmin(np.abs(np.log(old/n))))
        o=float(old[k])
        rows.append({
            "center_eval":int(r["center_eval"]),
            "candidate_nodes":int(n),
            "previous_stage":prev["stage"],
            "nearest_previous_nodes":int(o),
            "absolute_node_difference":int(abs(n-o)),
            "relative_node_difference":float(abs(n-o)/o),
            "scale_distance_abs_log_ratio":float(abs(math.log(n/o)))
        })
    return pd.DataFrame(rows)

def analyze(project:Path,sample:str,cfg,outdir:Path):
    src,raw=load_source(project,sample,cfg)
    df,schema=normalize_source(raw,cfg)
    activity,nfin,coh,cs=activity_state(df,cfg)
    df["resolved_speed_blocks"]=nfin
    df["activity"]=activity
    df["block_coherence"]=coh

    bas=pareto_layer1(detect_local_basins(df,cfg),cfg)
    front=bas[bas["pareto_layer"]==1].copy() if not bas.empty else bas.copy()

    previous=discover_previous_landmarks(project,sample,cfg)
    cmp=compare_to_previous(front,previous)

    sdir=outdir/sample; sdir.mkdir(parents=True,exist_ok=True)
    df.to_csv(sdir/"normalized_evaluation_trajectory_repaired.csv",index=False)
    bas.to_csv(sdir/"persistence_basins_repaired.csv",index=False)
    front.to_csv(sdir/"pareto_landmarks_repaired.csv",index=False)
    cmp.to_csv(sdir/"historical_landmark_alignment.csv",index=False)
    (sdir/"source_schema_resolution.json").write_text(json.dumps(schema,indent=2))
    (sdir/"historical_landmark_sources.json").write_text(json.dumps(previous,indent=2))

    coverage=schema["support_coverage"]
    finite_speed_after0=float(np.mean(nfin[1:]==len(cs))) if len(df)>1 and len(cs)>0 else 0.0
    support_ok=all(coverage[k] >= 0.95 for k in ["mass","expr","spatial"])
    local_ok=(not bas.empty) and bool((bas["lifetime"] < max(25,int(0.25*len(df)))).all())
    status="PASS" if len(front)>0 and support_ok and local_ok and finite_speed_after0>=0.95 else "HOLD"
    rep={
        "sample":sample,
        "source":str(src),
        "evaluations":len(df),
        "speed_blocks":len(cs),
        "full_speed_support_after_eval0":finite_speed_after0,
        "support_coverage":coverage,
        "basins":len(bas),
        "pareto_landmarks":len(front),
        "pareto_nodes":[int(v) for v in front["center_nodes"].tolist()] if len(front) else [],
        "previous_comparison_stage": previous[0]["stage"] if previous else None,
        "historical_alignment_rows":len(cmp),
        "new_merges_performed":False,
        "status":status
    }
    (sdir/"terminal_landscape_repair_report.json").write_text(json.dumps(rep,indent=2))
    return rep
