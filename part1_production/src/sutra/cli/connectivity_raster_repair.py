from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile

from sutra.geometry_benchmark.baselines import polygons_from_boundaries,make_raster
from sutra.connectivity.raster_repair import (
    remove_detached_inferred_fragments,connect_observed_polygon_raster,
    audit_disconnected
)
from sutra.production_geometry.materialize_radius5 import adjacency_from_labels,component_stats


def one(project_s,sample):
    project=Path(project_s)
    src=project/"results"/"production_geometry_r5"/sample
    z=np.load(src/"production_geometry.npz")
    labels=z["labels"].astype(np.int32)
    frozen_observed=z["observed_labels"].astype(np.int32)

    bpath=project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_cell_boundaries.parquet"
    bdf=pd.read_parquet(bpath); bdf["cell_id"]=bdf.cell_id.astype(str)
    cpath=project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_cells.csv"
    cells=pd.read_csv(cpath); cells["cell_id"]=cells.cell_id.astype(str)
    polys=polygons_from_boundaries(bdf)

    # Recreate exact raster metadata/crosswalk used by the frozen geometry.
    observed,meta=make_raster(polys,cells)
    if observed.shape!=frozen_observed.shape or not np.array_equal(observed,frozen_observed):
        raise RuntimeError(f"{sample}: regenerated observed raster differs from frozen source")

    pre= audit_disconnected(labels)

    # A. Remove completion-only detached islands. Leave vacated pixels background.
    work,n_inferred,removed=remove_detached_inferred_fragments(labels,observed)

    # B. Repair only observed raster splits by minimal paths inside measured polygons.
    work,bridge=connect_observed_polygon_raster(work,observed,polys,meta)

    post=audit_disconnected(work)

    out=project/"results"/"production_geometry_r5_connectivity"/sample
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"production_geometry_connectivity.npz",
                        labels=work,observed_labels=observed)
    tifffile.imwrite(out/"production_segmentation_connectivity.tif",
                     work.astype(np.uint32),compression="zlib")

    cross=pd.DataFrame({
        "mechanics_label":np.arange(1,int(work.max())+1,dtype=np.int32),
        "cell_id":[meta["label_to_cell"][i] for i in range(1,int(work.max())+1)]
    })
    cross.to_parquet(out/"label_to_cell_id.parquet",index=False)
    ids=dict(zip(cross.mechanics_label.astype(int),cross.cell_id.astype(str)))

    pairs=adjacency_from_labels(work)
    pred={tuple(sorted((ids[a],ids[b]))) for a,b in pairs}
    e=pd.read_csv(project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_gateB_edges.csv")
    truth={tuple(sorted((str(a),str(b)))) for a,b in zip(e.cell_i,e.cell_j)}
    tp=len(pred&truth); rec=tp/max(len(truth),1); new=len(pred-truth)/max(len(pred),1)
    comps=component_stats(pairs,len(ids))

    r={
        "sample":sample,
        "pre_disconnected_labels":len(pre),
        "inferred_fragment_labels_cleaned":n_inferred,
        "inferred_pixels_removed":removed,
        **bridge,
        "post_disconnected_labels":len(post),
        "post_disconnected_examples":post[:20],
        "gateB_contact_recall":float(rec),
        "new_contact_burden":float(new),
        **comps,
    }
    r["production_gate_pass"]=bool(
        len(post)==0 and bridge["bridge_failures"]==0 and
        rec>=.95 and new<=.35 and r["largest_component_fraction"]>=.90
    )
    (out/"connectivity_raster_repair.json").write_text(json.dumps(r,indent=2))
    return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project-root",default=".")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"];rs=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(one,str(project),s):s for s in samples}
        for f in as_completed(futs):
            r=f.result();rs.append(r)
            print(
                f"[DONE] {r['sample']}: pre={r['pre_disconnected_labels']} "
                f"inferred_cleaned={r['inferred_fragment_labels_cleaned']} "
                f"polygon_bridged={r['observed_raster_split_labels_repaired']} "
                f"bridge_fail={r['bridge_failures']} post={r['post_disconnected_labels']} "
                f"recall={100*r['gateB_contact_recall']:.2f}% "
                f"new={100*r['new_contact_burden']:.2f}% "
                f"LCC={100*r['largest_component_fraction']:.2f}% "
                f"pass={r['production_gate_pass']}"
            )
    rs.sort(key=lambda r:samples.index(r["sample"]))
    cert={"strata_version":"0.5.11","sample_reports":rs,
          "gate_status":"PASS" if all(r["production_gate_pass"] for r in rs) else "HOLD"}
    out=project/"results"/"production_geometry_r5_connectivity";out.mkdir(parents=True,exist_ok=True)
    (out/"connectivity_raster_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCONNECTIVITY-RASTER GATE: {cert['gate_status']}")
if __name__=="__main__":main()
