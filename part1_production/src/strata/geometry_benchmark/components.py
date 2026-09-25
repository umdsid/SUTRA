from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


@dataclass(frozen=True)
class ComponentConfig:
    factors: tuple = (2.0, 3.0, 4.0, 5.0)
    q_nn: float = 0.99
    min_plateau_jaccard: float = 0.98


def _component_at_radius(ids, xy, radius):
    tree=cKDTree(xy)
    pairs=np.asarray(list(tree.query_pairs(radius)),dtype=int)
    n=len(ids)
    if len(pairs)==0:
        labels=np.arange(n,dtype=int)
    else:
        rr=np.r_[pairs[:,0],pairs[:,1]]
        cc=np.r_[pairs[:,1],pairs[:,0]]
        mat=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(n,n)).tocsr()
        _,labels=connected_components(mat,directed=False,return_labels=True)
    counts=np.bincount(labels)
    lab=int(np.argmax(counts))
    keep=np.flatnonzero(labels==lab)
    return set(ids[i] for i in keep), labels, counts


def select_primary_component(cells: pd.DataFrame, cfg=ComponentConfig()):
    c=cells.copy()
    c["cell_id"]=c.cell_id.astype(str)
    xy=c[["x_centroid","y_centroid"]].to_numpy(float)
    ids=c.cell_id.tolist()

    tree=cKDTree(xy)
    d,_=tree.query(xy,k=2)
    nn=d[:,1]
    base=float(np.quantile(nn[np.isfinite(nn)],cfg.q_nn))

    sets=[]
    records=[]
    for f in cfg.factors:
        radius=float(f*base)
        s,labels,counts=_component_at_radius(ids,xy,radius)
        sets.append(s)
        records.append({
            "factor":float(f),
            "radius":radius,
            "largest_component_cells":len(s),
            "largest_component_fraction":len(s)/max(len(ids),1),
            "n_components":int(len(counts)),
        })

    jacc=[]
    for a,b in zip(sets[:-1],sets[1:]):
        jacc.append(len(a&b)/max(len(a|b),1))

    # Choose first set on a stable adjacent-scale plateau. If none, use the
    # largest scale but explicitly fail plateau certification.
    chosen=len(sets)-1
    plateau=False
    for i,j in enumerate(jacc):
        if j>=cfg.min_plateau_jaccard:
            chosen=i+1
            plateau=True
            break

    return sets[chosen], {
        "nearest_neighbor_q99":base,
        "scale_records":records,
        "adjacent_scale_jaccards":jacc,
        "chosen_factor":float(cfg.factors[chosen]),
        "chosen_radius":float(records[chosen]["radius"]),
        "plateau_certified":bool(plateau),
        "primary_cells":len(sets[chosen]),
        "primary_fraction":len(sets[chosen])/max(len(ids),1),
        "detached_cells":len(ids)-len(sets[chosen]),
    }
