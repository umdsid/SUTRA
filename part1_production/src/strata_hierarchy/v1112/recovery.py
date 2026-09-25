
from __future__ import annotations
import ast, hashlib, inspect, json, re
from pathlib import Path
import numpy as np
import pandas as pd

PASS_TOKENS={"PASS","TRUE","YES","Y","1","STABLE"}
HOLD_TOKENS={"HOLD","FAIL","FALSE","NO","N","0","UNSTABLE"}

def _suffix(c): return str(c).split(".")[-1].lower()

def status_float(v):
    if v is None: return np.nan
    try:
        if pd.isna(v): return np.nan
    except Exception:
        pass
    s=str(v).strip().upper()
    if s in PASS_TOKENS: return 1.0
    if s in HOLD_TOKENS: return 0.0
    return np.nan

def status_like(series:pd.Series):
    vals=series.dropna()
    if len(vals)==0: return 0.0
    s=vals.astype(str).str.strip().str.upper()
    return float(s.isin(PASS_TOKENS|HOLD_TOKENS).mean())

def find_col(df,names):
    for n in names:
        if n in df.columns: return n
    for n in names:
        hits=[c for c in df.columns if _suffix(c)==n.lower()]
        if len(hits)==1: return hits[0]
    return None

def support_col(df,kind,aliases):
    candidates=[]
    keys={"mass":["mass"],"expr":["expr","expression"],"spatial":["spatial"]}[kind]
    for c in df.columns:
        low=str(c).lower()
        if not any(k in low for k in keys): continue
        frac=status_like(df[c])
        if frac<=0: continue
        score=100*frac
        if _suffix(c) in [a.lower() for a in aliases]: score += 20
        if any(t in low for t in ["status","gate","stable","stability","pass","ok"]): score += 10
        candidates.append((score,c,frac))
    if not candidates: return None
    candidates.sort(reverse=True,key=lambda x:x[0])
    return candidates[0][1]

def read_table(p:Path):
    suf=p.suffix.lower()
    try:
        if suf==".parquet": return pd.read_parquet(p)
        if suf==".csv": return pd.read_csv(p)
        if suf==".tsv": return pd.read_csv(p,sep="\t")
        if suf in {".jsonl",".ndjson"}: return pd.read_json(p,lines=True)
        if suf==".json":
            obj=json.loads(p.read_text())
            if isinstance(obj,list): return pd.DataFrame(obj)
            if isinstance(obj,dict):
                # common record containers
                for k in ["records","evaluations","history","trajectory","rows","checkpoints"]:
                    if k in obj and isinstance(obj[k],list):
                        return pd.DataFrame(obj[k])
                # a dict keyed by evaluation IDs
                if obj and all(isinstance(v,dict) for v in obj.values()):
                    return pd.DataFrame(list(obj.values()))
    except Exception:
        return None
    return None

def inventory_status_sources(project:Path,sample:str,cfg):
    base=project/"results"/cfg["production_stage"]
    roots=[base/sample,base]
    seen=set(); rows=[]
    for root in roots:
        if not root.exists(): continue
        for p in root.rglob("*"):
            if not p.is_file() or p in seen: continue
            seen.add(p)
            if p.suffix.lower() not in {".parquet",".csv",".tsv",".json",".jsonl",".ndjson"}: continue
            # Global files are accepted only if they carry an explicit sample field.
            if root==base and sample not in str(p):
                df=read_table(p)
                if df is None or "sample" not in df.columns: continue
                df=df[df["sample"].astype(str)==sample].copy()
            else:
                df=read_table(p)
                if df is None: continue
                if "sample" in df.columns:
                    d2=df[df["sample"].astype(str)==sample].copy()
                    if len(d2): df=d2
            if df is None or len(df)==0: continue
            ec=find_col(df,cfg["evaluation_aliases"])
            nc=find_col(df,cfg["node_aliases"])
            found={}
            for kind in cfg["required_supports"]:
                c=support_col(df,kind,cfg["support_aliases"][kind])
                if c is not None:
                    found[kind]=str(c)
            if not found: continue
            rows.append({
                "path":str(p),
                "rows":int(len(df)),
                "evaluation_column":str(ec) if ec is not None else None,
                "node_column":str(nc) if nc is not None else None,
                "support_columns":found
            })
    return rows

def load_production_evaluations(project:Path,sample:str,cfg):
    base=project/"results"/cfg["production_stage"]/sample
    p=base/"scientific_evaluations.parquet"
    if not p.exists():
        hits=list(base.rglob("*scientific*evaluation*.parquet"))
        if not hits: raise RuntimeError(f"{sample}: scientific_evaluations.parquet not found")
        p=hits[0]
    d=pd.read_parquet(p)
    ec=find_col(d,cfg["evaluation_aliases"]); nc=find_col(d,cfg["node_aliases"])
    if ec is None or nc is None: raise RuntimeError(f"{sample}: eval/node keys unresolved")
    out=d.copy()
    out["_eval_key"]=pd.to_numeric(out[ec],errors="coerce")
    out["_node_key"]=pd.to_numeric(out[nc],errors="coerce")
    return p,out

def recover_supports(project:Path,sample:str,cfg):
    src,base=load_production_evaluations(project,sample,cfg)
    inv=inventory_status_sources(project,sample,cfg)
    recovered=base[["_eval_key","_node_key"]].copy()
    provenance={k:[] for k in cfg["required_supports"]}

    for kind in cfg["required_supports"]:
        recovered[kind]=np.nan
        for rec in inv:
            if kind not in rec["support_columns"]: continue
            p=Path(rec["path"]); d=read_table(p)
            if d is None or len(d)==0: continue
            if "sample" in d.columns:
                d=d[d["sample"].astype(str)==sample].copy()
            ec=find_col(d,cfg["evaluation_aliases"]); nc=find_col(d,cfg["node_aliases"])
            sc=rec["support_columns"][kind]
            if sc not in d.columns: continue
            dd=d.copy()
            if ec is not None:
                dd["_eval_key"]=pd.to_numeric(dd[ec],errors="coerce")
            if nc is not None:
                dd["_node_key"]=pd.to_numeric(dd[nc],errors="coerce")
            dd["_status"]=dd[sc].map(status_float)

            # Exact two-key join is preferred. Exact single-key join is allowed only
            # when that key is unique in both tables. No nearest matching.
            joined=None; mode=None
            if ec is not None and nc is not None:
                z=dd[["_eval_key","_node_key","_status"]].dropna(subset=["_eval_key","_node_key"]).drop_duplicates(["_eval_key","_node_key"])
                joined=recovered[["_eval_key","_node_key"]].merge(z,on=["_eval_key","_node_key"],how="left")
                mode="evaluation+nodes"
            elif ec is not None and recovered["_eval_key"].is_unique and dd["_eval_key"].is_unique:
                z=dd[["_eval_key","_status"]].dropna(subset=["_eval_key"]).drop_duplicates("_eval_key")
                joined=recovered[["_eval_key"]].merge(z,on="_eval_key",how="left")
                mode="evaluation"
            elif nc is not None and recovered["_node_key"].is_unique and dd["_node_key"].is_unique:
                z=dd[["_node_key","_status"]].dropna(subset=["_node_key"]).drop_duplicates("_node_key")
                joined=recovered[["_node_key"]].merge(z,on="_node_key",how="left")
                mode="nodes"
            if joined is None: continue

            vals=joined["_status"].to_numpy(float)
            mask=np.isnan(recovered[kind].to_numpy(float)) & np.isfinite(vals)
            if mask.any():
                recovered.loc[mask,kind]=vals[mask]
                provenance[kind].append({
                    "path":str(p),"column":sc,"join_mode":mode,
                    "new_rows":int(mask.sum())
                })
            if recovered[kind].notna().mean()>=cfg["minimum_exact_join_coverage"]:
                break

    cov={k:float(recovered[k].notna().mean()) for k in cfg["required_supports"]}
    return src,recovered,inv,provenance,cov

def frozen_rule_inventory(project:Path):
    files=[]
    for rel in [
        "src/strata_hierarchy/v110/terminal_rule.py",
        "configs/hierarchy_v110_production_terminal_rule.json",
        "src/strata_hierarchy/v1100/production.py",
    ]:
        p=project/rel
        if p.exists():
            b=p.read_bytes()
            rec={"path":str(p),"sha256":hashlib.sha256(b).hexdigest()}
            if p.suffix==".py":
                try:
                    tree=ast.parse(b.decode())
                    rec["functions"]=[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]
                except Exception:
                    rec["functions"]=[]
            files.append(rec)
    return files

def isolated_historical_landmarks(project:Path,sample:str,cfg):
    """
    Strict specimen isolation:
      * sample-specific files/directories are accepted directly;
      * global tables require an explicit sample column and are filtered;
      * global JSON is recursively searched ONLY inside branches whose sample key
        exactly equals the requested specimen.
    """
    records=[]
    def numbers_from_named_fields(obj):
        vals=[]
        if isinstance(obj,dict):
            for k,v in obj.items():
                low=str(k).lower()
                if isinstance(v,(int,float)) and ("node" in low or "landmark" in low) and float(v)>=2:
                    vals.append(int(round(float(v))))
                elif isinstance(v,list) and ("node" in low or "landmark" in low):
                    vals.extend(int(round(float(x))) for x in v if isinstance(x,(int,float)) and float(x)>=2)
                elif isinstance(v,(dict,list)):
                    vals.extend(numbers_from_named_fields(v))
        elif isinstance(obj,list):
            for v in obj:
                vals.extend(numbers_from_named_fields(v))
        return vals

    def sample_branch(obj):
        if isinstance(obj,dict):
            if sample in obj:
                return obj[sample]
            # common arrays of sample reports
            for k in ["sample_reports","reports","samples","specimens"]:
                if k in obj:
                    x=obj[k]
                    if isinstance(x,list):
                        for r in x:
                            if isinstance(r,dict) and str(r.get("sample",""))==sample:
                                return r
                    elif isinstance(x,dict) and sample in x:
                        return x[sample]
        return None

    for stage in cfg["historical_stages"]:
        base=project/"results"/stage
        if not base.exists(): continue
        vals=set(); sources=[]
        # 1. sample-specific paths
        for p in base.rglob("*"):
            if not p.is_file(): continue
            if sample not in str(p): continue
            if p.suffix.lower()==".json":
                try:
                    obj=json.loads(p.read_text())
                    vals.update(numbers_from_named_fields(obj)); sources.append(str(p))
                except Exception: pass
            elif p.suffix.lower() in {".csv",".tsv"}:
                try:
                    d=pd.read_csv(p,sep="\t" if p.suffix.lower()==".tsv" else ",")
                    if "sample" in d.columns: d=d[d["sample"].astype(str)==sample]
                    for c in d.columns:
                        if "node" in c.lower() or "landmark" in c.lower():
                            for v in pd.to_numeric(d[c],errors="coerce").dropna():
                                if v>=2: vals.add(int(round(v)))
                    sources.append(str(p))
                except Exception: pass
        # 2. global explicit-sample branches only
        for p in base.glob("*"):
            if not p.is_file(): continue
            if p.suffix.lower()==".json":
                try:
                    obj=json.loads(p.read_text()); br=sample_branch(obj)
                    if br is not None:
                        vals.update(numbers_from_named_fields(br)); sources.append(str(p))
                except Exception: pass
            elif p.suffix.lower() in {".csv",".tsv"}:
                try:
                    d=pd.read_csv(p,sep="\t" if p.suffix.lower()==".tsv" else ",")
                    if "sample" not in d.columns: continue
                    d=d[d["sample"].astype(str)==sample]
                    for c in d.columns:
                        if "node" in c.lower() or "landmark" in c.lower():
                            for v in pd.to_numeric(d[c],errors="coerce").dropna():
                                if v>=2: vals.add(int(round(v)))
                    if len(d): sources.append(str(p))
                except Exception: pass
        if vals:
            records.append({"stage":stage,"nodes":sorted(vals,reverse=True),"sources":sorted(set(sources))})
    return records
