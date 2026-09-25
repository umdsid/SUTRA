from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sutra.hierarchy.v10931.replay_shards import discover_stream, load_initial_labels, replay_stream, snapshot_audit, nested_labels

def find_final_landmarks(project:Path,sample:str,cfg):
    p=project/'results'/cfg['final_landscape_stage']/sample/'final_supported_pareto_landmarks.csv'
    d=pd.read_csv(p)
    need=['center_eval','center_nodes','entry_eval','entry_nodes']
    miss=[c for c in need if c not in d.columns]
    if miss: raise RuntimeError(f'{sample}: final landmark columns missing: {miss}')
    return d.sort_values('center_eval').reset_index(drop=True)

def parquet_num_rows(p:Path):
    try:
        import pyarrow.parquet as pq
        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:return None

def infer_n0(project:Path,sample:str,cfg):
    for p in [project/'data'/sample/'cells.parquet', project/'results'/cfg['level0_stage']/sample/'cells.parquet']:
        if p.exists():
            n=parquet_num_rows(p)
            if n and n>100:return int(n)
    return None

def canonical_native_sources(project:Path,sample:str):
    hp=project/'data'/sample/'cell_feature_matrix.h5'; cp=project/'data'/sample/'cells.parquet'
    return {'expression_h5':[str(hp)] if hp.exists() else [], 'cells_parquet':[str(cp)] if cp.exists() else []}

def load_v10931_config(project:Path):
    p=project/'configs'/'hierarchy_v10931_ordered_merge_shard_replay.json'
    if not p.exists():
        xs=sorted((project/'configs').glob('*10931*.json'))
        if len(xs)!=1: raise RuntimeError(f'Expected one v10931 replay config; found {xs}')
        p=xs[0]
    cfg=json.loads(p.read_text())
    if 'v093_ledger_root' not in cfg: raise RuntimeError(f'{p} lacks v093_ledger_root')
    return cfg,str(p)

def _norm_label(v):
    # Label semantics are categorical. Numeric IDs remain numeric for
    # backward compatibility; arbitrary categorical IDs remain strings.
    if isinstance(v,(np.integer,int)):
        return int(v)
    if isinstance(v,(np.floating,float)) and np.isfinite(v) and float(v).is_integer():
        return int(v)
    if isinstance(v,(bytes,np.bytes_)):
        return v.decode()
    return str(v)

def _array_candidate(v,n0,target_nodes):
    # Backward-compatible helper retained for v1.2.0.1 tests and for
    # diagnostic parsing. It is not used to infer approximate hierarchy states.
    if isinstance(v,pd.Series):
        v=v.to_numpy()
    if isinstance(v,(list,tuple,np.ndarray)):
        a=np.asarray(v)
        if a.ndim==1 and len(a)==int(n0):
            vals=np.array([_norm_label(x) for x in a],dtype=object)
            if len(set(vals.tolist()))==int(target_nodes):
                return vals
    if isinstance(v,dict):
        for k,x in v.items():
            if any(t in str(k).lower() for t in ['label','partition','assignment','component','super']):
                a=_array_candidate(x,n0,target_nodes)
                if a is not None:
                    return a
    return None

def component_transition(entry_labels,center_labels):
    e=np.asarray(entry_labels); c=np.asarray(center_labels)
    if len(e)!=len(c):
        raise RuntimeError('Entry/center label lengths differ')
    en=np.array([_norm_label(x) for x in e],dtype=object)
    cn=np.array([_norm_label(x) for x in c],dtype=object)
    d=pd.DataFrame({'entry':en,'center':cn})
    nun=d.groupby('entry',sort=False)['center'].nunique()
    if int((nun>1).sum())>0:
        raise RuntimeError('Non-nested entry->center partition detected')
    x=d.drop_duplicates().groupby('center',sort=False)['entry'].apply(list)
    return {
        _norm_label(p):[_norm_label(y) for y in ch]
        for p,ch in x.items() if len(ch)>=2
    }

def component_sizes(labels):
    vals=np.array([_norm_label(x) for x in np.asarray(labels)],dtype=object)
    s=pd.Series(vals,dtype=object).value_counts(sort=False)
    return {_norm_label(k):int(v) for k,v in s.items()}

def transition_structure(entry_labels,center_labels):
    merged=component_transition(entry_labels,center_labels); es=component_sizes(entry_labels); cs=component_sizes(center_labels); rows=[]
    for parent,children in merged.items():
        masses=[es[x] for x in children]; total=sum(masses)
        rows.append({'center_component':parent,'n_entry_children':len(children),'center_mass':cs[parent],
                     'entry_child_labels':';'.join(map(str,children)),'entry_child_masses':';'.join(map(str,masses)),
                     'largest_child_fraction':max(masses)/total,'smallest_child_fraction':min(masses)/total})
    return pd.DataFrame(rows),merged

def replay_all_required_snapshots(project:Path,sample:str,lm:pd.DataFrame,n0:int):
    rcfg,rcfg_path=load_v10931_config(project)
    files,disc=discover_stream(project,sample,rcfg)
    if files is None or len(files)==0:
        return {},{'status':'HOLD','reason':'no_v093_merge_stream_discovered','replay_config':rcfg_path,'v093_ledger_root':rcfg.get('v093_ledger_root')},disc,pd.DataFrame()
    initial,ip,ik=load_initial_labels(project,sample,n0,rcfg)
    if initial is None:
        return {},{'status':'HOLD','reason':'initial_labels_unresolved','replay_config':rcfg_path,'n_discovered':len(files)},disc,pd.DataFrame()
    targets=sorted(set(lm['entry_nodes'].astype(int).tolist()+lm['center_nodes'].astype(int).tolist()),reverse=True)
    snaps,ledger,ok=replay_stream(files,initial,targets)
    meta={'status':'PASS' if ok else 'HOLD','stream_ok':bool(ok),'replay_config':rcfg_path,'v093_ledger_root':rcfg.get('v093_ledger_root'),
          'n_discovered':len(files),'initial_labels_path':ip,'initial_labels_key':ik,'initial_unique_nodes':int(len(np.unique(initial))),
          'targets_requested':targets,'targets_resolved':sorted(map(int,snaps.keys()),reverse=True),
          'targets_missing':sorted([int(x) for x in targets if int(x) not in snaps],reverse=True)}
    return snaps,meta,disc,ledger

def write_preflight(project:Path,sample:str,cfg,outdir:Path):
    lm=find_final_landmarks(project,sample,cfg); n0=infer_n0(project,sample,cfg); native=canonical_native_sources(project,sample)
    sdir=outdir/sample; sdir.mkdir(parents=True,exist_ok=True)
    report={'sample':sample,'n0':n0,'landmarks':int(len(lm)),'resolved_transitions':0,'native_sources':native,'unresolved':[]}
    if n0 is None:
        report['status']='HOLD'; report['unresolved'].append('N0'); (sdir/'merger_driver_preflight.json').write_text(json.dumps(report,indent=2)); return report
    snaps,meta,disc,ledger=replay_all_required_snapshots(project,sample,lm,n0)
    if disc is not None and len(disc): disc.to_csv(sdir/'certified_merge_shard_discovery.csv',index=False)
    if ledger is not None and len(ledger): ledger.to_csv(sdir/'certified_replay_ledger.csv',index=False)
    (sdir/'certified_replay_metadata.json').write_text(json.dumps(meta,indent=2))
    structures=[]; audits=[]; prov=[]
    for _,r in lm.iterrows():
        ee,en,ce,cn=int(r['entry_eval']),int(r['entry_nodes']),int(r['center_eval']),int(r['center_nodes'])
        el,cl=snaps.get(en),snaps.get(cn)
        if el is None or cl is None:
            report['unresolved'].append({'entry_eval':ee,'entry_nodes':en,'center_eval':ce,'center_nodes':cn,'entry_snapshot':el is not None,'center_snapshot':cl is not None}); continue
        ea=snapshot_audit(el,n0,en); ca=snapshot_audit(cl,n0,cn); nesting,nok=nested_labels(el,cl)
        if not (ea['cell_count_match'] and ea['node_count_match'] and ca['cell_count_match'] and ca['node_count_match'] and nok):
            report['unresolved'].append({'entry_nodes':en,'center_nodes':cn,'reason':'snapshot_or_nesting_audit_failed'}); continue
        st,merged=transition_structure(el,cl)
        if len(st):
            st.insert(0,'landmark_center_eval',ce); st.insert(1,'landmark_center_nodes',cn); st.insert(2,'entry_eval',ee); st.insert(3,'entry_nodes',en); structures.append(st)
        np.savez_compressed(sdir/f'labels_entry_eval_{ee:03d}_nodes_{en}.npz',labels=np.asarray(el,dtype=np.int64))
        np.savez_compressed(sdir/f'labels_center_eval_{ce:03d}_nodes_{cn}.npz',labels=np.asarray(cl,dtype=np.int64))
        nesting.to_parquet(sdir/f'nesting_entry_{en}_to_center_{cn}.parquet',index=False)
        audits.append({'entry_eval':ee,'entry_nodes':en,'center_eval':ce,'center_nodes':cn,'nested':bool(nok)})
        prov.append({'entry_eval':ee,'entry_nodes':en,'center_eval':ce,'center_nodes':cn,'n_actual_merger_parents':int(len(merged))})
        report['resolved_transitions']+=1
    if structures: pd.concat(structures,ignore_index=True).to_csv(sdir/'landmark_merger_structure.csv',index=False)
    pd.DataFrame(audits).to_csv(sdir/'landmark_partition_audit.csv',index=False)
    (sdir/'snapshot_provenance.json').write_text(json.dumps(prov,indent=2)); (sdir/'native_source_inventory.json').write_text(json.dumps(native,indent=2))
    report['replay_stream_ready']=bool(meta.get('stream_ok',False)); report['structure_ready']=report['resolved_transitions']==len(lm)
    report['expression_ready']=len(native['expression_h5'])==1 and len(native['cells_parquet'])==1
    report['status']='PASS' if report['structure_ready'] else 'HOLD'
    (sdir/'merger_driver_preflight.json').write_text(json.dumps(report,indent=2)); return report
