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
        aa=a[q].astype(np.int64); bb=b[q].astype(np.int64)
        for x,y in zip(aa,bb):
            if x!=y:
                pairs.add((int(min(x,y)),int(max(x,y))))
    return pairs


def component_stats(edges, n_labels):
    if n_labels<=0:
        return {"n_components":0,"largest_component_fraction":0.0}
    if not edges:
        return {"n_components":n_labels,"largest_component_fraction":1.0/n_labels}
    rr=[];cc=[]
    for a,b in edges:
        # labels are 1-based
        rr += [a-1,b-1]
        cc += [b-1,a-1]
    mat=coo_matrix((np.ones(len(rr)),(rr,cc)),shape=(n_labels,n_labels)).tocsr()
    ncomp,labs=connected_components(mat,directed=False,return_labels=True)
    counts=np.bincount(labs,minlength=ncomp)
    return {
        "n_components":int(ncomp),
        "largest_component_cells":int(counts.max()) if len(counts) else 0,
        "largest_component_fraction":float(counts.max()/n_labels) if len(counts) else 0.0,
    }


def summarize_area_change(observed, completed, n_labels):
    old=np.bincount(observed.ravel(),minlength=n_labels+1)[1:n_labels+1].astype(float)
    new=np.bincount(completed.ravel(),minlength=n_labels+1)[1:n_labels+1].astype(float)
    ratio=(new+1)/(old+1)
    return {
        "median_abs_log_area_ratio":float(np.median(np.abs(np.log(ratio)))),
        "q95_abs_log_area_ratio":float(np.quantile(np.abs(np.log(ratio)),.95)),
        "median_area_ratio":float(np.median(ratio)),
        "q05_area_ratio":float(np.quantile(ratio,.05)),
        "q95_area_ratio":float(np.quantile(ratio,.95)),
    }


def materialize_sample(project: Path, sample: str):
    src=project/"results"/"tranche2_2b_geometry_benchmark"/sample
    prep=json.loads((src/"prepare_summary.json").read_text())
    z=np.load(src/"baseline_constrained_nearest.npz")
    completed=z["labels"].astype(np.int32)
    observed=z["observed"].astype(np.int32)

    bdf=pd.read_parquet(src/"primary_cell_boundaries.parquet")
    ids=sorted(bdf.cell_id.astype(str).unique())
    n_labels=int(completed.max())
    if n_labels != len(ids):
        raise RuntimeError(
            f"{sample}: label/cell crosswalk mismatch: labels={n_labels}, ids={len(ids)}"
        )

    out=project/"results"/"production_geometry"/sample
    out.mkdir(parents=True,exist_ok=True)

    cross=pd.DataFrame({
        "mechanics_label":np.arange(1,n_labels+1,dtype=np.int32),
        "cell_id":ids,
    })
    cross.to_parquet(out/"label_to_cell_id.parquet",index=False)

    np.savez_compressed(
        out/"production_geometry.npz",
        labels=completed,
        observed_labels=observed,
    )

    # TensionMap accepts an instance-segmentation mask.
    import tifffile
    tifffile.imwrite(
        out/"production_segmentation.tif",
        completed.astype(np.uint32),
        compression="zlib",
    )

    pairs=adjacency_from_labels(completed)
    edge_rows=[
        {"label_i":a,"label_j":b,"cell_i":ids[a-1],"cell_j":ids[b-1]}
        for a,b in sorted(pairs)
    ]
    edf=pd.DataFrame(edge_rows)
    edf.to_parquet(out/"production_contacts.parquet",index=False)

    gateb=pd.read_csv(src/"primary_gateB_edges.csv")
    truth={tuple(sorted((str(a),str(b)))) for a,b in zip(gateb.cell_i,gateb.cell_j)}
    pred={tuple(sorted((r["cell_i"],r["cell_j"]))) for r in edge_rows}
    tp=len(pred&truth)
    precision=tp/max(len(pred),1)
    recall=tp/max(len(truth),1)

    occ=completed>0
    filled_fraction=float(occ.mean())
    area=summarize_area_change(observed,completed,n_labels)
    comps=component_stats(pairs,n_labels)

    summary={
        "sample":sample,
        "method":"constrained_nearest",
        "primary_component":prep["primary_component"],
        "n_cells":n_labels,
        "n_contacts":len(pred),
        "observed_pixels_preserved":bool(np.all(completed[observed>0]==observed[observed>0])),
        "filled_raster_fraction":filled_fraction,
        "gateB_contact_precision":float(precision),
        "gateB_contact_recall":float(recall),
        "gateB_contact_f1":float(2*precision*recall/max(precision+recall,1e-15)),
        "new_contact_burden":float(len(pred-truth)/max(len(pred),1)),
        "lost_gateB_fraction":float(len(truth-pred)/max(len(truth),1)),
        **area,
        **comps,
    }
    (out/"production_geometry_summary.json").write_text(json.dumps(summary,indent=2))
    return summary
