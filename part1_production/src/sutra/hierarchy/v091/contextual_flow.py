from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp

@dataclass(frozen=True)
class ContextualFlowConfig:
    base_hops:int=2
    max_hops:int=8
    diffusion_alpha:float=0.35
    context_mix:float=0.40
    context_reliability_gain:float=0.85
    beta_softmax:float=3.0
    uncertainty_penalty:float=0.35
    molecular_weight:float=1.0
    mechanics_weight:float=1.0
    communication_weight:float=0.75
    geometry_weight:float=1.0
    topology_weight:float=0.40
    initial_cost_quantile:float=0.35
    maximum_cost_quantile:float=0.90
    scale_growth:float=0.40
    epsilon_fraction:float=2e-4
    soft_step_1:int=1500
    soft_step_2:int=3000
    soft_step_3:int=5000
    rescue_fraction_1:float=5e-4
    rescue_fraction_2:float=1e-3
    last_resort_fraction:float=5e-3
    exact_frontier_max_edges:int=18
    checkpoint_every:int=50
    heavy_landmark_reduction_fraction:float=2.5e-3
    max_microsteps_safety:int=20000
    archive_all_candidates:bool=True
    def validate(self):
        if self.base_hops<1 or self.max_hops<self.base_hops: raise ValueError("invalid hop schedule")
        if not 0<self.diffusion_alpha<1: raise ValueError("diffusion_alpha")
        if not 0<=self.context_mix<=1: raise ValueError("context_mix")
        if not 0<=self.context_reliability_gain<=1: raise ValueError("context_reliability_gain")
        if self.beta_softmax<=0: raise ValueError("beta_softmax")
        if self.uncertainty_penalty<0: raise ValueError("uncertainty_penalty")
        if not 0<self.initial_cost_quantile<self.maximum_cost_quantile<1: raise ValueError("quantiles")
        return self

def adaptive_hops(mean_supernode_size,cfg):
    s=max(float(mean_supernode_size),1.0)
    return min(cfg.max_hops,cfg.base_hops+int(math.floor(math.log2(s))))

def _edge_index(cand,node_ids):
    idx={int(x):k for k,x in enumerate(np.asarray(node_ids,dtype=np.int64))}
    ii=np.fromiter((idx[int(x)] for x in cand.super_i),dtype=np.int64,count=len(cand))
    jj=np.fromiter((idx[int(x)] for x in cand.super_j),dtype=np.int64,count=len(cand))
    return ii,jj

def _row_stochastic_graph(cand,node_ids):
    n=len(node_ids)
    if len(cand)==0:
        return sparse.identity(n,format="csr"),np.array([],int),np.array([],int)
    ii,jj=_edge_index(cand,node_ids)
    nbe=pd.to_numeric(cand.get("n_boundary_edges",pd.Series(np.ones(len(cand)))),errors="coerce").fillna(1).to_numpy(float)
    w=np.log1p(np.maximum(nbe,1.0))
    rr=np.r_[ii,jj,np.arange(n)]; cc=np.r_[jj,ii,np.arange(n)]; dd=np.r_[w,w,np.ones(n)]
    A=sparse.coo_matrix((dd,(rr,cc)),shape=(n,n)).tocsr()
    rs=np.asarray(A.sum(axis=1)).ravel()
    P=sparse.diags(1/np.maximum(rs,1e-12))@A
    return P,ii,jj

def _normalized_context(values,reliability,cand,node_ids,P,ii,jj,hops,alpha):
    n=len(node_ids); x=np.asarray(values,float)
    r=np.clip(np.nan_to_num(np.asarray(reliability,float),nan=0.0),0,1)
    finite=np.isfinite(x); r=r*finite; x=np.where(finite,x,0.0)
    nbe=pd.to_numeric(cand.get("n_boundary_edges",pd.Series(np.ones(len(cand)))),errors="coerce").fillna(1).to_numpy(float)
    ew=np.log1p(np.maximum(nbe,1.0))
    num=np.zeros(n); den=np.zeros(n); inc=np.zeros(n)
    np.add.at(num,ii,ew*r*x); np.add.at(num,jj,ew*r*x)
    np.add.at(den,ii,ew*r); np.add.at(den,jj,ew*r)
    np.add.at(inc,ii,ew); np.add.at(inc,jj,ew)
    sup=np.divide(den,np.maximum(inc,1e-12))
    for _ in range(int(hops)):
        num=(1-alpha)*num+alpha*(P@num)
        den=(1-alpha)*den+alpha*(P@den)
        sup=(1-alpha)*sup+alpha*(P@sup)
    node=np.divide(num,np.maximum(den,1e-12))
    return 0.5*(node[ii]+node[jj]),np.clip(0.5*(sup[ii]+sup[jj]),0,1)

def contextualize_scalar(values,reliability,cand,node_ids,P,ii,jj,hops,cfg):
    local=np.asarray(values,float)
    r=np.clip(np.nan_to_num(np.asarray(reliability,float),nan=0.0),0,1)
    ctx,ctxr=_normalized_context(local,r,cand,node_ids,P,ii,jj,hops,cfg.diffusion_alpha)
    eff=np.where(np.isfinite(local),(1-cfg.context_mix)*local+cfg.context_mix*ctx,ctx)
    re=np.clip(r+(1-r)*cfg.context_reliability_gain*ctxr,0,1)
    return eff,re,ctx,ctxr

def primitive_fields(cand):
    n=len(cand)
    mol=pd.to_numeric(cand.get("molecular_distance",pd.Series(np.nan,index=cand.index)),errors="coerce").to_numpy(float)
    molr=np.isfinite(mol).astype(float)
    t=pd.to_numeric(cand.get("abs_tension_z",pd.Series(np.nan,index=cand.index)),errors="coerce").to_numpy(float)
    p=pd.to_numeric(cand.get("abs_delta_p_z",pd.Series(np.nan,index=cand.index)),errors="coerce").to_numpy(float)
    stack=np.c_[t,p]
    with np.errstate(invalid="ignore"):
        mech=np.sqrt(np.nanmean(stack*stack,axis=1))
    fc=np.isfinite(stack).sum(axis=1)/2.0
    ms=pd.to_numeric(cand.get("mechanics_support_fraction",pd.Series(np.zeros(n),index=cand.index)),errors="coerce").fillna(0).to_numpy(float)
    mechr=np.clip(ms,0,1)*fc
    comm=pd.to_numeric(cand.get("comm_support",pd.Series(np.nan,index=cand.index)),errors="coerce").to_numpy(float)
    comm=np.maximum(comm,0); commr=np.isfinite(comm).astype(float)
    geom=pd.to_numeric(cand.get("geometry_pair_cost",pd.Series(np.nan,index=cand.index)),errors="coerce").to_numpy(float)
    resolved=np.asarray(cand.get("geometry_state_resolved",pd.Series(np.isfinite(geom),index=cand.index)),dtype=bool)
    geomr=(resolved&np.isfinite(geom)).astype(float)
    nbe=pd.to_numeric(cand.get("n_boundary_edges",pd.Series(np.ones(n),index=cand.index)),errors="coerce").fillna(1).to_numpy(float)
    topo=1/np.sqrt(np.maximum(nbe,1.0)); topor=np.ones(n)
    return {"molecular":(mol,molr),"mechanics":(mech,mechr),"communication_support":(comm,commr),"geometry":(geom,geomr),"topology":(topo,topor)}

def contextual_block_table(cand,node_ids,mean_supernode_size,cfg):
    if len(cand)==0:return cand.copy()
    P,ii,jj=_row_stochastic_graph(cand,node_ids)
    hops=adaptive_hops(mean_supernode_size,cfg)
    out=cand.copy()
    for name,(x,r) in primitive_fields(cand).items():
        eff,re,ctx,cr=contextualize_scalar(x,r,cand,node_ids,P,ii,jj,hops,cfg)
        out[f"{name}_local"]=x
        out[f"{name}_context"]=ctx
        out[f"{name}_effective"]=eff
        out[f"{name}_reliability_local"]=r
        out[f"{name}_reliability_context"]=cr
        out[f"{name}_reliability_effective"]=re
    out["context_hops"]=hops
    return out

def robust_positive_scale(x):
    a=np.asarray(x,float); a=a[np.isfinite(a)&(a>=0)]
    if len(a)==0:return 1.0
    pos=a[a>0]
    if len(pos):
        q=float(np.quantile(pos,.5))
        if q>1e-12:return q
    q=float(np.quantile(a,.75))
    return q if q>1e-12 else 1.0

def pooled_scales(frames):
    out={}
    for k in ["molecular","mechanics","communication_support","geometry","topology"]:
        vals=np.concatenate([pd.to_numeric(f[f"{k}_effective"],errors="coerce").to_numpy(float) for f in frames if len(f)])
        out[k]=robust_positive_scale(vals)
    return out

def score_candidates(frame,scales,cfg):
    if len(frame)==0:
        z=frame.copy(); z["composite_merge_cost"]=np.array([],float); return z
    y=frame.copy()
    def a(k): return pd.to_numeric(y[f"{k}_effective"],errors="coerce").to_numpy(float)
    P=np.c_[
        np.log1p(np.maximum(a("molecular"),0)/max(scales["molecular"],1e-12)),
        np.log1p(np.maximum(a("mechanics"),0)/max(scales["mechanics"],1e-12)),
        np.exp(-np.maximum(a("communication_support"),0)/max(scales["communication_support"],1e-12)),
        np.log1p(np.maximum(a("geometry"),0)/max(scales["geometry"],1e-12)),
        np.log1p(np.maximum(a("topology"),0)/max(scales["topology"],1e-12)),
    ]
    rel=np.c_[y.molecular_reliability_effective,y.mechanics_reliability_effective,y.communication_support_reliability_effective,y.geometry_reliability_effective,y.topology_reliability_effective].astype(float)
    rel=np.clip(np.nan_to_num(rel,nan=0.0),0,1)
    w=np.array([cfg.molecular_weight,cfg.mechanics_weight,cfg.communication_weight,cfg.geometry_weight,cfg.topology_weight],float)
    W=rel*w[None,:]*np.isfinite(P); P=np.nan_to_num(P,nan=0,posinf=1e6)
    den=W.sum(axis=1)
    logs=np.log(np.maximum(W,1e-300))+cfg.beta_softmax*P
    agg=(logsumexp(logs,axis=1)-np.log(np.maximum(den,1e-300)))/cfg.beta_softmax
    cov=den/max(w.sum(),1e-12)
    cost=agg+cfg.uncertainty_penalty*(1-np.clip(cov,0,1)); cost[den<=1e-12]=np.inf
    y["composite_merge_cost"]=cost; y["evidence_coverage"]=cov
    for i,k in enumerate(["molecular","mechanics","communication","geometry","topology"]): y[f"{k}_penalty"]=P[:,i]
    return y

def freeze_cost_thresholds(scored_frames,cfg):
    x=np.concatenate([pd.to_numeric(f.composite_merge_cost,errors="coerce").to_numpy(float) for f in scored_frames if len(f)])
    x=x[np.isfinite(x)]
    return {"initial_threshold":float(np.quantile(x,cfg.initial_cost_quantile)),"maximum_threshold":float(np.quantile(x,cfg.maximum_cost_quantile)),"initial_quantile":cfg.initial_cost_quantile,"maximum_quantile":cfg.maximum_cost_quantile,"n_level0_boundaries":int(len(x)),"pooled_across_specimens":True,"later_level_rescaling":False}

def scale_threshold(mean_supernode_size,frozen,cfg):
    s=max(float(mean_supernode_size),1.0)
    t=float(frozen["initial_threshold"])*(1+cfg.scale_growth*math.log1p(max(0,s-1)))
    return min(float(frozen["maximum_threshold"]),t)

def apply_merge_admissibility(scored,mean_supernode_size,frozen,cfg):
    y=scored.copy(); tau=scale_threshold(mean_supernode_size,frozen,cfg)
    finite=np.isfinite(y.composite_merge_cost)
    y["merge_cost_threshold"]=tau
    y["merge_allowed"]=finite&(y.composite_merge_cost<=tau)
    if "geometry_pair_cost" in y:y["geometry_pair_cost_raw"]=y.geometry_pair_cost
    if "alpha_cost" in y:y["alpha_cost_raw"]=y.alpha_cost
    y["admissible"]=y.merge_allowed
    y["geometry_state_resolved"]=finite
    y["geometry_pair_cost"]=y.composite_merge_cost
    y["alpha_cost"]=y.composite_merge_cost
    y["ordering_merit"]=-y.composite_merge_cost
    return y

def calibration_manifest(scales,thresholds,cfg):
    return {"scheme":"normalized graph convolution + reliability-aware log-sum-exp merger cost","scales":scales,"thresholds":thresholds,"config":asdict(cfg),"hard_local_mechanics_veto":False,"hard_zero_communication_veto":False,"hard_unresolved_geometry_veto":False,"specimen_specific_scaling":False,"target_final_node_count":None}
