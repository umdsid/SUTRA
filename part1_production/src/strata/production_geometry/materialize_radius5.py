from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def adjacency_from_labels(labels):
    pairs=set()
    for a,b in ((labels[:,:-1],labels[:,1:]),(labels[:-1,:],labels[1:,:])):
        q=(a!=b)&(a>0)&(b>0)
        for x,y in zip(a[q].astype(np.int64),b[q].astype(np.int64)):
            if x!=y:
                pairs.add((int(min(x,y)),int(max(x,y))))
    return pairs

def component_stats(edges,n_labels):
    if n_labels<=0:
        return {"n_components":0,"largest_component_fraction":0.0}
    rr=[];cc=[]
    for a,b in edges:
        rr += [a-1,b-1]; cc += [b-1,a-1]
    if rr:
        mat=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(n_labels,n_labels)).tocsr()
        ncomp,labs=connected_components(mat,directed=False,return_labels=True)
        counts=np.bincount(labs,minlength=ncomp)
        return {
            "n_components":int(ncomp),
            "largest_component_cells":int(counts.max()),
            "largest_component_fraction":float(counts.max()/n_labels),
        }
    return {"n_components":n_labels,"largest_component_cells":1,"largest_component_fraction":1/n_labels}

def materialize(project:Path,sample:str):
    calroot=project/"results"/"tranche2_2e_radius_calibration"/sample
    cal=json.loads((calroot/"radius_calibration.json").read_text())
    if cal.get("status")!="PASS" or float(cal.get("selected_radius"))!=5.0:
        raise RuntimeError(f"{sample}: radius 5.0 not frozen in sample calibration")

    z=np.load(calroot/"selected_geometry.npz")
    labels=z["labels"].astype(np.int32)
    observed=z["observed"].astype(np.int32)

    bdf=pd.read_parquet(
        project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_cell_boundaries.parquet"
    )
    ids=sorted(bdf.cell_id.astype(str).unique())
    n=int(labels.max())
    if n!=len(ids):
        raise RuntimeError(f"{sample}: labels={n}, cell ids={len(ids)}")

    out=project/"results"/"production_geometry_r5"/sample
    out.mkdir(parents=True,exist_ok=True)

    pd.DataFrame({
        "mechanics_label":np.arange(1,n+1,dtype=np.int32),
        "cell_id":ids
    }).to_parquet(out/"label_to_cell_id.parquet",index=False)

    np.savez_compressed(out/"production_geometry.npz",labels=labels,observed_labels=observed)

    import tifffile
    tifffile.imwrite(out/"production_segmentation.tif",labels.astype(np.uint32),compression="zlib")

    pairs=adjacency_from_labels(labels)
    edges=pd.DataFrame([
        {"label_i":a,"label_j":b,"cell_i":ids[a-1],"cell_j":ids[b-1]}
        for a,b in sorted(pairs)
    ])
    edges.to_parquet(out/"production_contacts.parquet",index=False)

    truthdf=pd.read_csv(
        project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_gateB_edges.csv"
    )
    truth={tuple(sorted((str(a),str(b)))) for a,b in zip(truthdf.cell_i,truthdf.cell_j)}
    pred={tuple(sorted((str(a),str(b)))) for a,b in zip(edges.cell_i,edges.cell_j)}
    tp=len(pred&truth)
    precision=tp/max(len(pred),1); recall=tp/max(len(truth),1)
    comps=component_stats(pairs,n)

    summary={
        "sample":sample,
        "method":"constrained_nearest",
        "frozen_radius":5.0,
        "n_cells":n,
        "n_contacts":len(pred),
        "observed_pixels_preserved":bool(np.all(labels[observed>0]==observed[observed>0])),
        "gateB_contact_precision":float(precision),
        "gateB_contact_recall":float(recall),
        "gateB_contact_f1":float(2*precision*recall/max(precision+recall,1e-15)),
        "new_contact_burden":float(len(pred-truth)/max(len(pred),1)),
        "lost_gateB_fraction":float(len(truth-pred)/max(len(truth),1)),
        "remaining_unassigned_fraction":float(
            next(x["remaining_unassigned_fraction"] for x in cal["results"] if float(x["radius"])==5.0)
        ),
        **comps,
    }
    (out/"production_geometry_summary.json").write_text(json.dumps(summary,indent=2))
    return summary
