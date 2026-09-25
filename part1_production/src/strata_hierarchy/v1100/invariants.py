from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
import pandas as pd

def mass_counts(labels, n0=None):
    a=np.asarray(labels,dtype=np.int64)
    if n0 is None:
        n0=len(a)
    if len(a)!=int(n0):
        raise AssertionError(f"partition length {len(a)} != N0 {n0}")
    ids,c=np.unique(a,return_counts=True)
    if int(c.sum())!=int(n0):
        raise AssertionError("exact mass conservation failed")
    if np.any(c<=0):
        raise AssertionError("non-positive active mass")
    return ids.astype(np.int64),c.astype(np.int64)

def assert_disjoint_selected(selected, active_ids):
    active=set(map(int,active_ids))
    used=set()
    for r in selected.itertuples(index=False):
        u=int(r.super_i);v=int(r.super_j)
        if u==v:
            raise AssertionError("self merge")
        if u not in active or v not in active:
            raise AssertionError(f"merge endpoint not active: {u},{v}")
        if u in used or v in used:
            raise AssertionError(f"non-disjoint merge batch at {u},{v}")
        used.add(u);used.add(v)
    return True

def assert_batch_exact(before_labels, after_labels, selected, n0):
    before=np.asarray(before_labels,dtype=np.int64)
    after=np.asarray(after_labels,dtype=np.int64)
    b_ids,b_mass=mass_counts(before,n0)
    a_ids,a_mass=mass_counts(after,n0)
    assert_disjoint_selected(selected,b_ids)
    if len(a_ids) != len(b_ids)-len(selected):
        raise AssertionError(
            f"node-count drop {len(b_ids)}->{len(a_ids)} "
            f"!= selected merges {len(selected)}"
        )
    bm={int(u):int(m) for u,m in zip(b_ids,b_mass)}
    am={int(u):int(m) for u,m in zip(a_ids,a_mass)}
    touched=set()
    for r in selected.itertuples(index=False):
        u,v=sorted((int(r.super_i),int(r.super_j)))
        expected=bm[u]+bm[v]
        got=am.get(u,None)
        if got != expected:
            raise AssertionError(
                f"parent mass mismatch for {u}<-({u},{v}): {got} != {expected}"
            )
        if v in am:
            raise AssertionError(f"absorbed node {v} remains active")
        touched.add(u);touched.add(v)
    for u,m in bm.items():
        if u in touched:
            continue
        if am.get(u,None)!=m:
            raise AssertionError(f"untouched node mass changed for {u}")
    return {
        "partition_exact":True,
        "mass_exact":True,
        "nodes_before":int(len(b_ids)),
        "nodes_after":int(len(a_ids)),
        "selected_merges":int(len(selected)),
    }

def level0_components(edges,n0):
    from scipy import sparse
    from scipy.sparse.csgraph import connected_components
    i=edges.cell_i_index.to_numpy(np.int64)
    j=edges.cell_j_index.to_numpy(np.int64)
    A=sparse.coo_matrix(
        (np.ones(2*len(i),dtype=np.uint8),(np.r_[i,j],np.r_[j,i])),
        shape=(int(n0),int(n0))
    ).tocsr()
    _,lab=connected_components(A,directed=False)
    return lab.astype(np.int64)

def active_component_map(labels, level0_component):
    labels=np.asarray(labels,np.int64)
    comp=np.asarray(level0_component,np.int64)
    out={}
    for u in np.unique(labels):
        vals=np.unique(comp[labels==u])
        if len(vals)!=1:
            raise AssertionError(f"supernode {u} spans initial components {vals.tolist()}")
        out[int(u)]=int(vals[0])
    return out

def assert_merge_components(selected, component_map):
    for r in selected.itertuples(index=False):
        u=int(r.super_i);v=int(r.super_j)
        if component_map[u]!=component_map[v]:
            raise AssertionError(f"merge joins initial components: {u},{v}")
    return True

@dataclass
class LineageTracker:
    token_by_supernode: dict
    mass_by_supernode: dict

    @classmethod
    def level0(cls,n0):
        return cls(
            {i:f"L0:{i}" for i in range(int(n0))},
            {i:1 for i in range(int(n0))}
        )

    @classmethod
    def from_frame(cls,df):
        return cls(
            {int(r.supernode_id):str(r.lineage_token) for r in df.itertuples(index=False)},
            {int(r.supernode_id):int(r.mass) for r in df.itertuples(index=False)}
        )

    def active_frame(self):
        ids=sorted(self.token_by_supernode)
        return pd.DataFrame({
            "supernode_id":ids,
            "lineage_token":[self.token_by_supernode[i] for i in ids],
            "mass":[self.mass_by_supernode[i] for i in ids],
        })

    def apply(self,selected,step):
        rows=[]
        used=set()
        for k,r in enumerate(selected.itertuples(index=False)):
            u,v=sorted((int(r.super_i),int(r.super_j)))
            if u in used or v in used:
                raise AssertionError("lineage received non-disjoint batch")
            used.add(u);used.add(v)
            if u not in self.token_by_supernode or v not in self.token_by_supernode:
                raise AssertionError("lineage endpoint missing")
            tu=self.token_by_supernode[u];tv=self.token_by_supernode[v]
            mu=int(self.mass_by_supernode[u]);mv=int(self.mass_by_supernode[v])
            parent=f"M:{int(step):08d}:{k:05d}:{u}:{v}"
            pm=mu+mv
            rows.extend([
                {
                    "microstep":int(step),"parent_token":parent,"child_token":tu,
                    "parent_supernode_id":u,"child_supernode_id":u,
                    "child_role":"retained","child_mass":mu,"parent_mass":pm,
                },
                {
                    "microstep":int(step),"parent_token":parent,"child_token":tv,
                    "parent_supernode_id":u,"child_supernode_id":v,
                    "child_role":"absorbed","child_mass":mv,"parent_mass":pm,
                }
            ])
            self.token_by_supernode[u]=parent
            self.mass_by_supernode[u]=pm
            del self.token_by_supernode[v]
            del self.mass_by_supernode[v]
        return pd.DataFrame(rows)

def assert_lineage_matches_labels(tracker,labels,n0):
    ids,m=mass_counts(labels,n0)
    lm={int(u):int(x) for u,x in zip(ids,m)}
    if set(lm)!=set(tracker.mass_by_supernode):
        raise AssertionError("lineage active IDs != partition active IDs")
    for u,x in lm.items():
        if tracker.mass_by_supernode[u]!=x:
            raise AssertionError(f"lineage mass mismatch at {u}")
    if sum(tracker.mass_by_supernode.values())!=int(n0):
        raise AssertionError("lineage total mass != N0")
    return True
