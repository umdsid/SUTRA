
from __future__ import annotations
import hashlib, importlib.util, inspect, json, math, re, types
from pathlib import Path
import numpy as np
import pandas as pd

def sha256_file(p:Path):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for ch in iter(lambda:f.read(1024*1024),b""):
            h.update(ch)
    return h.hexdigest()

def verify_hashes(project:Path, required:dict):
    rows=[]
    ok=True
    for rel,exp in required.items():
        p=project/rel
        got=sha256_file(p) if p.exists() else None
        good=(got==exp)
        rows.append({"path":str(p),"expected":exp,"observed":got,"match":good})
        ok &= good
    return ok,rows

def load_module_from_path(name,p:Path):
    spec=importlib.util.spec_from_file_location(name,p)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def find_eval_table(project:Path,sample:str,cfg):
    base=project/"results"/cfg["production_stage"]/sample
    p=base/"scientific_evaluations.parquet"
    if not p.exists():
        hits=list(base.rglob("*scientific*evaluation*.parquet"))
        if not hits:
            raise RuntimeError(f"{sample}: no scientific evaluation parquet")
        p=hits[0]
    return p,pd.read_parquet(p)

def _suffix(c): return str(c).split(".")[-1].lower()

def find_col(df,names):
    for n in names:
        if n in df.columns: return n
    for n in names:
        hits=[c for c in df.columns if _suffix(c)==n.lower()]
        if len(hits)==1: return hits[0]
    return None

def source_context(project:Path,sample:str,cfg):
    """
    Return the exact saved evaluation table plus pointers to the frozen rule/config.
    This stage never changes them.
    """
    p,df=find_eval_table(project,sample,cfg)
    ec=find_col(df,["evaluation_index","evaluation","eval","scientific_evaluation"])
    nc=find_col(df,["nodes","n_nodes","node_count","active_nodes"])
    if ec is None or nc is None:
        raise RuntimeError(f"{sample}: eval/node columns unresolved")
    x=df.copy()
    x["_eval_key"]=pd.to_numeric(x[ec],errors="coerce")
    x["_node_key"]=pd.to_numeric(x[nc],errors="coerce")
    x=x.dropna(subset=["_eval_key","_node_key"]).sort_values("_eval_key").reset_index(drop=True)
    return p,x

def _status(v):
    s=str(v).strip().upper()
    if s in {"PASS","TRUE","YES","1","STABLE"}: return "PASS"
    if s in {"HOLD","FAIL","FALSE","NO","0","UNSTABLE"}: return "HOLD"
    return None

def discover_rule_api(mod):
    """
    Introspect the frozen terminal-rule module. We deliberately do not assume a
    single function name; the exact exported API is reported and then used when
    compatible with the saved production state.
    """
    funcs={}
    for name,obj in vars(mod).items():
        if inspect.isfunction(obj) and obj.__module__==mod.__name__:
            funcs[name]=str(inspect.signature(obj))
    return funcs

def _candidate_functions(mod):
    priority=[
        "evaluate_once","evaluate_persistent","evaluate_terminal_rule",
        "evaluate_terminal","evaluate","check_terminal_rule","terminal_rule"
    ]
    out=[]
    for n in priority:
        if hasattr(mod,n) and inspect.isfunction(getattr(mod,n)):
            out.append((n,getattr(mod,n)))
    for n,obj in vars(mod).items():
        if inspect.isfunction(obj) and n not in {x[0] for x in out}:
            low=n.lower()
            if any(t in low for t in ["evaluate","terminal","persistent"]):
                out.append((n,obj))
    return out

def _row_mapping(row):
    d={str(k):row[k] for k in row.index}
    # useful aliases without mutating values
    for k,v in list(d.items()):
        d[_suffix(k)]=v
    return d

def _call_compatible(fn,row_map,cfg_obj,state):
    sig=inspect.signature(fn)
    kwargs={}
    required_missing=[]
    for name,p in sig.parameters.items():
        low=name.lower()
        if low in row_map:
            kwargs[name]=row_map[low]
        elif name in row_map:
            kwargs[name]=row_map[name]
        elif low in {"row","record","evaluation","metrics","observables","snapshot"}:
            kwargs[name]=row_map
        elif low in {"cfg","config","rule_config","terminal_config"}:
            kwargs[name]=cfg_obj
        elif low in {"state","history","tracker","persistent_state"}:
            kwargs[name]=state
        elif p.default is not inspect._empty:
            pass
        else:
            required_missing.append(name)
    if required_missing:
        return None,{"compatible":False,"missing":required_missing}
    try:
        out=fn(**kwargs)
        return out,{"compatible":True,"missing":[]}
    except TypeError as e:
        return None,{"compatible":False,"error":repr(e)}


def _flatten_status_leaves(obj, prefix=""):
    leaves=[]
    if obj is None:
        return leaves
    if hasattr(obj,"__dict__") and not isinstance(obj,(dict,list,tuple)):
        try:
            obj=vars(obj)
        except Exception:
            pass
    if isinstance(obj,dict):
        for k,v in obj.items():
            p=f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v,(dict,list,tuple)) or (hasattr(v,"__dict__") and not isinstance(v,(str,int,float,bool,np.generic))):
                leaves.extend(_flatten_status_leaves(v,p))
            else:
                leaves.append((p,v))
    elif isinstance(obj,(list,tuple)):
        for i,v in enumerate(obj):
            p=f"{prefix}[{i}]"
            if isinstance(v,(dict,list,tuple)) or hasattr(v,"__dict__"):
                leaves.extend(_flatten_status_leaves(v,p))
            else:
                leaves.append((p,v))
    return leaves

def _extract_status_triplet(obj):
    """
    Extract the three frozen rule status decisions from arbitrary nested output
    structures. Only explicit Boolean/PASS/HOLD-like leaves whose key path
    contains the corresponding scientific sector are accepted.
    """
    leaves=_flatten_status_leaves(obj)
    out={}
    key_terms={
        "mass":["mass"],
        "expr":["expr","expression"],
        "spatial":["spatial"]
    }
    for kind,terms in key_terms.items():
        candidates=[]
        for path,v in leaves:
            low=path.lower()
            if not any(t in low for t in terms):
                continue
            s=_status(v)
            if s is None:
                continue
            score=0
            if any(t in low for t in ["status","pass","ok","stable","stability","gate"]):
                score += 10
            if low.endswith(tuple(terms)):
                score += 3
            candidates.append((score,path,s))
        if candidates:
            candidates.sort(reverse=True,key=lambda x:x[0])
            out[kind]=candidates[0][2]
    return out if all(k in out for k in ["mass","expr","spatial"]) else None


def _history_representations(records):
    """
    The frozen terminal rule is history-based. Use the cumulative causal prefix
    of saved evaluation records rather than a separate empty state object.
    """
    return [
        ("list_of_dicts", records),
        ("dataframe", pd.DataFrame(records)),
    ]

def _call_history_rule(fn, history_obj, cfg_obj):
    try:
        return fn(history_obj,cfg_obj), None
    except Exception as e:
        return None, repr(e)

def exact_replay(project:Path,sample:str,cfg):
    rule_path=project/"src/sutra.hierarchy/v110/terminal_rule.py"
    cfg_path=project/"configs/hierarchy_v110_production_terminal_rule.json"
    prod_path=project/"src/sutra.hierarchy/v1100/production.py"
    rule=load_module_from_path("strata_v110_terminal_rule_replay",rule_path)
    prod=load_module_from_path("strata_v1100_production_replay",prod_path)
    rule_cfg=json.loads(cfg_path.read_text())
    src,df=source_context(project,sample,cfg)

    api=discover_rule_api(rule)
    prod_api=discover_rule_api(prod)

    if not hasattr(rule,"evaluate_once") or not hasattr(rule,"evaluate_persistent"):
        raise RuntimeError("Frozen rule does not expose evaluate_once/evaluate_persistent as expected.")

    records=[]
    rows=[]
    trace=[]
    used_repr=None

    for i,row in df.iterrows():
        rec={str(k):row[k] for k in df.columns if not str(k).startswith("_")}
        rec["evaluation_index"]=int(row["_eval_key"])
        rec["nodes"]=int(row["_node_key"])
        records.append(rec)

        once_out=None
        persistent_out=None
        attempts=[]
        rep_used_this_eval=None

        for rep_name,hist in _history_representations(records):
            oo,e1=_call_history_rule(rule.evaluate_once,hist,rule_cfg)
            po,e2=_call_history_rule(rule.evaluate_persistent,hist,rule_cfg)
            attempts.append({
                "representation":rep_name,
                "evaluate_once_error":e1,
                "evaluate_persistent_error":e2,
                "evaluate_once_type":type(oo).__name__ if e1 is None else None,
                "evaluate_persistent_type":type(po).__name__ if e2 is None else None,
            })
            if e1 is None and e2 is None:
                once_out=oo
                persistent_out=po
                rep_used_this_eval=rep_name
                used_repr=rep_name
                break

        trip=_extract_status_triplet(once_out)
        source_name="evaluate_once"
        if trip is None:
            trip=_extract_status_triplet(persistent_out)
            source_name="evaluate_persistent"

        confirm=None
        for obj in [persistent_out,once_out]:
            for path,v in _flatten_status_leaves(obj):
                low=path.lower()
                if "confirm" in low:
                    try:
                        confirm=int(v)
                        break
                    except Exception:
                        pass
            if confirm is not None:
                break

        rows.append({
            "evaluation_index":int(row["_eval_key"]),
            "nodes":int(row["_node_key"]),
            "mass_status":trip["mass"] if trip else None,
            "expr_status":trip["expr"] if trip else None,
            "spatial_status":trip["spatial"] if trip else None,
            "confirm":confirm,
            "status_source":source_name if trip else None,
            "history_representation":rep_used_this_eval
        })

        if i < 5:
            def safe_repr(x):
                return repr(x)[:6000]
            trace.append({
                "evaluation":int(row["_eval_key"]),
                "attempts":attempts,
                "evaluate_once_repr":safe_repr(once_out),
                "evaluate_persistent_repr":safe_repr(persistent_out),
                "parsed_triplet":trip,
                "confirm":confirm
            })

    out=pd.DataFrame(rows)
    coverage={k:float(out[k].notna().mean()) for k in ["mass_status","expr_status","spatial_status"]}
    meta={
        "source_evaluations":str(src),
        "rule_api":api,
        "production_api":prod_api,
        "history_semantics":"cumulative causal prefix of saved scientific evaluations",
        "history_representation_used":used_repr,
        "first5_output_trace":trace,
        "coverage":coverage
    }
    return out,meta

def strict_historical_nodes(project:Path,sample:str,cfg):
    """
    Read only explicit node arrays/columns. Generic landmark counts are excluded.
    """
    stage=project/"results"/"hierarchy_v1061_full_landmark_materialization"
    vals=set(); sources=[]
    if not stage.exists(): return [],[]
    for p in stage.rglob("*"):
        if not p.is_file(): continue
        if p.suffix.lower()==".json":
            try:
                obj=json.loads(p.read_text())
            except Exception:
                continue
            def walk(x, sample_ok=False):
                if isinstance(x,dict):
                    local_sample_ok=sample_ok
                    if str(x.get("sample",""))==sample: local_sample_ok=True
                    if sample in x and isinstance(x[sample],(dict,list)):
                        walk(x[sample],True)
                    for k,v in x.items():
                        low=str(k).lower()
                        explicit = low in {
                            "nodes","node_counts","landmark_nodes","pareto_nodes",
                            "principal_landmark_nodes","frozen_landmark_nodes"
                        }
                        if local_sample_ok and explicit:
                            if isinstance(v,list):
                                for z in v:
                                    if isinstance(z,(int,float)) and float(z)>=20:
                                        vals.add(int(round(float(z))))
                            elif isinstance(v,(int,float)) and float(v)>=20:
                                vals.add(int(round(float(v))))
                        if isinstance(v,(dict,list)):
                            walk(v,local_sample_ok)
                elif isinstance(x,list):
                    for v in x: walk(v,sample_ok)
            # sample-specific path counts as sample_ok
            walk(obj, sample in str(p))
            if vals: sources.append(str(p))
        elif p.suffix.lower() in {".csv",".tsv"}:
            try:
                d=pd.read_csv(p,sep="\t" if p.suffix.lower()==".tsv" else ",")
            except Exception:
                continue
            if "sample" in d.columns:
                d=d[d["sample"].astype(str)==sample]
            elif sample not in str(p):
                continue
            explicit=[c for c in d.columns if c.lower() in {
                "nodes","node_count","landmark_nodes","pareto_nodes","principal_landmark_nodes"
            }]
            for c in explicit:
                for v in pd.to_numeric(d[c],errors="coerce").dropna():
                    if v>=20: vals.add(int(round(v)))
            if len(d) and explicit: sources.append(str(p))
    return sorted(vals,reverse=True),sorted(set(sources))
