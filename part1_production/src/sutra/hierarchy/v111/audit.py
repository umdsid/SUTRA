
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
SAMPLES=['healthy_reference','gbm_reference_addon','alzheimers','nondiseased_kidney','prcc']
def _first(df,a):
    for c in a:
        if c in df.columns:return c
    return None
def _status(v):
    if v is None:return np.nan
    s=str(v).strip().upper()
    if s in {'PASS','TRUE','1','YES'}:return 1.0
    if s in {'HOLD','FAIL','FALSE','0','NO'}:return 0.0
    try:return float(v)
    except:return np.nan
def _flat(o,p=''):
    out={}
    if isinstance(o,dict):
        for k,v in o.items():
            q=f'{p}.{k}' if p else str(k)
            if isinstance(v,dict):out.update(_flat(v,q))
            elif isinstance(v,(str,int,float,bool)) or v is None:out[q]=v
    return out
def _frames(d):
    fs=[]
    for p in d.rglob('*'):
        if not p.is_file():continue
        try:
            if p.suffix.lower()=='.parquet':df=pd.read_parquet(p)
            elif p.suffix.lower()=='.csv':df=pd.read_csv(p)
            elif p.suffix.lower() in {'.tsv','.txt'}:df=pd.read_csv(p,sep='\t')
            else:continue
        except:continue
        if len(df)>=2:fs.append((p,df))
    rows=[];src=[]
    for p in d.rglob('*.json'):
        try:o=json.loads(p.read_text())
        except:continue
        f=_flat(o); keys=' '.join(f).lower()
        if any(k in keys for k in ['eval','node','mass','expr','spatial','speed_']):rows.append(f);src.append(str(p))
    if len(rows)>=2:
        x=pd.DataFrame(rows);x['_json_source']=src;fs.append((d/'<json-evaluations>',x))
    return fs
def _norm(df,cfg):
    df=df.copy(); sm={}
    for c in df.columns:sm.setdefault(str(c).split('.')[-1],[]).append(c)
    for a in set(cfg['node_column_aliases']+cfg['eval_column_aliases']+cfg['scale_column_aliases']+cfg['stability_columns']):
        if a not in df.columns and len(sm.get(a,[]))==1:df[a]=df[sm[a][0]]
    ec=_first(df,cfg['eval_column_aliases']);nc=_first(df,cfg['node_column_aliases'])
    if ec is None or nc is None:return None
    o=pd.DataFrame({'eval':pd.to_numeric(df[ec],errors='coerce'),'nodes':pd.to_numeric(df[nc],errors='coerce')})
    sc=_first(df,cfg['scale_column_aliases'])
    if sc:o['ell']=pd.to_numeric(df[sc],errors='coerce')
    else:
        n0=np.nanmax(o.nodes.to_numpy(float));o['ell']=np.log(np.maximum(n0,1)/np.maximum(o.nodes,1))
    for c in cfg['stability_columns']:
        src=c if c in df.columns else (sm.get(c,[None])[0] if len(sm.get(c,[]))==1 else None)
        o[c]=df[src].map(_status) if src else np.nan
    for c in df.columns:
        s=str(c).split('.')[-1]
        if any(s.startswith(p) for p in cfg['block_prefixes']):o[s]=pd.to_numeric(df[c],errors='coerce')
    for c in df.columns:
        if str(c).split('.')[-1] in {'slow','slow_fraction','slow_block_fraction'}:o['slow_fraction']=pd.to_numeric(df[c],errors='coerce');break
    o=o.dropna(subset=['eval','nodes']).sort_values('eval').drop_duplicates('eval',keep='last').reset_index(drop=True)
    return o if len(o)>=2 else None
def load_trajectory(project,sample,cfg):
    base=project/'results'/cfg['source_stage']; roots=[p for p in base.rglob('*') if p.is_dir() and sample in p.name] or [base]
    cand=[];seen=set()
    for r in roots:
        for p,df in _frames(r):
            if str(p) in seen:continue
            seen.add(str(p));n=_norm(df,cfg)
            if n is not None:
                score=(len(n),sum(c.startswith(tuple(cfg['block_prefixes'])) for c in n.columns),int(n[['mass','expr','spatial']].notna().sum().sum()))
                cand.append((score,p,n))
    if not cand:raise RuntimeError(f'{sample}: no parseable v1.1.0 trajectory under {base}')
    cand.sort(key=lambda x:x[0],reverse=True);_,p,n=cand[0]
    if len(n)<cfg['minimum_evaluations']:raise RuntimeError(f'{sample}: only {len(n)} evaluations in {p}')
    return p,n
def robust_z(x):
    x=np.asarray(x,float);m=np.nanmedian(x);mad=np.nanmedian(np.abs(x-m));s=1.4826*mad
    if not np.isfinite(s) or s<1e-12:s=np.nanstd(x)
    if not np.isfinite(s) or s<1e-12:s=1.0
    return (x-m)/s
def activity(df,cfg):
    cs=[c for c in df if any(c.startswith(p) for p in cfg['block_prefixes'])]
    if cs:
        A=np.vstack([np.abs(robust_z(df[c])) for c in cs]).T
        return np.sqrt(np.nanmean(A*A,axis=1)),np.sum(np.isfinite(A),axis=1),np.nanmean(A<=1,axis=1),len(cs)
    A=[]
    for c in ['mass','expr','spatial']:
        x=df[c].to_numpy(float);d=np.full(len(x),np.nan);d[1:]=np.abs(np.diff(x));A.append(d)
    A=np.vstack(A).T
    return np.sqrt(np.nanmean(A*A,axis=1)),np.sum(np.isfinite(A),axis=1),np.nanmean(A==0,axis=1),0
def detect(df,cfg):
    act,res,coh,nb=activity(df,cfg);w=cfg['basin']['smoothing_window'];sm=pd.Series(act).rolling(w,min_periods=1).median().to_numpy();r=cfg['basin']['local_radius'];mins=[]
    for i in range(r,len(sm)-r):
        if np.isfinite(sm[i]) and sm[i]<=np.nanmin(sm[i-r:i+r+1]):mins.append(i)
    bs=[]
    for i in mins:
        cen=sm[i];finite=sm[np.isfinite(sm)];ceil=np.nanmedian(finite) if len(finite) else np.inf;thr=cen+.5*max(ceil-cen,0);l=rr=i;g=0
        while l>0:
            if np.isfinite(sm[l-1]) and sm[l-1]<=thr:l-=1;g=0
            elif g<cfg['basin']['maximum_gap_evals']:l-=1;g+=1
            else:break
        g=0
        while rr<len(sm)-1:
            if np.isfinite(sm[rr+1]) and sm[rr+1]<=thr:rr+=1;g=0
            elif g<cfg['basin']['maximum_gap_evals']:rr+=1;g+=1
            else:break
        life=rr-l+1
        if life<cfg['basin']['minimum_lifetime_evals']:continue
        shoulder=np.nanmedian(np.r_[sm[max(0,l-r):l],sm[rr+1:min(len(sm),rr+r+1)]])
        if not np.isfinite(shoulder):shoulder=np.nanmedian(sm)
        sl=slice(l,rr+1)
        sup=lambda c:float(np.nanmean(df[c].to_numpy(float)[sl])) if np.isfinite(df[c].to_numpy(float)[sl]).any() else np.nan
        bs.append(dict(center_eval=int(df.loc[i,'eval']),center_nodes=int(df.loc[i,'nodes']),entry_eval=int(df.loc[l,'eval']),exit_eval=int(df.loc[rr,'eval']),entry_nodes=int(df.loc[l,'nodes']),exit_nodes=int(df.loc[rr,'nodes']),lifetime=int(life),depth=float(max(shoulder-cen,0)),activity_min=float(cen),block_coherence=float(np.nanmean(coh[sl])),resolved_block_median=float(np.nanmedian(res[sl])),expression_support=sup('expr'),spatial_support=sup('spatial'),mass_support=sup('mass'),speed_block_count=int(nb)))
    bs=sorted(bs,key=lambda b:(-b['depth'],-b['lifetime']));keep=[]
    for b in bs:
        ov=False
        for k in keep:
            a1,a2=b['entry_eval'],b['exit_eval'];c1,c2=k['entry_eval'],k['exit_eval'];inter=max(0,min(a2,c2)-max(a1,c1)+1);union=max(a2,c2)-min(a1,c1)+1
            if union and inter/union>.6:ov=True;break
        if not ov:keep.append(b)
    return pd.DataFrame(sorted(keep,key=lambda b:b['center_eval']))
def pareto(b,cfg):
    if b.empty:return b.assign(pareto_layer=pd.Series(dtype=int))
    obj=cfg['pareto_objectives']['maximize'];V=np.array([[(-np.inf if pd.isna(r[c]) else float(r[c])) for c in obj] for _,r in b.iterrows()]);front=np.ones(len(V),bool)
    for i in range(len(V)):
        for j in range(len(V)):
            if i!=j and np.all(V[j]>=V[i]) and np.any(V[j]>V[i]):front[i]=False;break
    o=b.copy();o['pareto_layer']=np.where(front,1,2);return o
def analyze_sample(project,sample,cfg,outdir):
    src,df=load_trajectory(project,sample,cfg);b=pareto(detect(df,cfg),cfg);sd=outdir/sample;sd.mkdir(parents=True,exist_ok=True);df.to_csv(sd/'normalized_evaluation_trajectory.csv',index=False);b.to_csv(sd/'persistence_basins.csv',index=False);f=b[b.pareto_layer==1] if len(b) else b;f.to_csv(sd/'pareto_landmarks.csv',index=False)
    rep={'sample':sample,'source':str(src),'evaluations':len(df),'basins':len(b),'pareto_landmarks':len(f),'pareto_nodes':[int(x) for x in f.center_nodes.tolist()] if len(f) else [],'new_merges_performed':False,'status':'PASS' if len(f) else 'HOLD'};(sd/'terminal_landscape_report.json').write_text(json.dumps(rep,indent=2));return rep
