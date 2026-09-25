import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile
from strata.connectivity.repair import audit_and_repair,RepairConfig
from strata.production_geometry.materialize_radius5 import adjacency_from_labels,component_stats

def one(project_s,sample):
    project=Path(project_s)
    src=project/"results"/"production_geometry_r5"/sample
    z=np.load(src/"production_geometry.npz")
    labels=z["labels"].astype(np.int32); observed=z["observed_labels"].astype(np.int32)
    repaired,a=audit_and_repair(labels,observed,RepairConfig(10))
    out=project/"results"/"production_geometry_r5_connected"/sample
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/"production_geometry_connected.npz",labels=repaired,observed_labels=observed)
    tifffile.imwrite(out/"production_segmentation_connected.tif",repaired.astype(np.uint32),compression="zlib")

    cross=pd.read_parquet(src/"label_to_cell_id.parquet")
    ids=dict(zip(cross.mechanics_label.astype(int),cross.cell_id.astype(str)))
    pairs=adjacency_from_labels(repaired)
    pred={tuple(sorted((ids[x],ids[y]))) for x,y in pairs}
    e=pd.read_csv(project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_gateB_edges.csv")
    truth={tuple(sorted((str(x),str(y)))) for x,y in zip(e.cell_i,e.cell_j)}
    tp=len(pred&truth); rec=tp/max(len(truth),1); new=len(pred-truth)/max(len(pred),1)
    comps=component_stats(pairs,len(ids))
    r={"sample":sample,**a,"gateB_contact_recall":rec,"new_contact_burden":new,**comps}
    r["production_gate_pass"]=bool(r["status"]=="PASS" and rec>=.95 and new<=.35 and r["largest_component_fraction"]>=.90)
    (out/"connectivity_repair.json").write_text(json.dumps(r,indent=2))
    return r

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default="."); a=ap.parse_args()
    project=Path(a.project_root).resolve(); samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]; rs=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(one,str(project),s):s for s in samples}
        for f in as_completed(futs):
            r=f.result(); rs.append(r)
            print(f"[DONE] {r['sample']}: pre={r['completed_disconnected_labels']} inferred={r['inferred_fragment_labels']} observed={r['observed_disconnected_labels']} post={r.get('postrepair_disconnected_labels','NA')} recall={100*r['gateB_contact_recall']:.2f}% new={100*r['new_contact_burden']:.2f}% LCC={100*r['largest_component_fraction']:.2f}% pass={r['production_gate_pass']}")
    rs.sort(key=lambda r:samples.index(r["sample"]))
    cert={"strata_version":"0.5.10","sample_reports":rs,"gate_status":"PASS" if all(r["production_gate_pass"] for r in rs) else "HOLD"}
    out=project/"results"/"production_geometry_r5_connected"; out.mkdir(parents=True,exist_ok=True)
    (out/"connectivity_repair_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nCONNECTIVITY GATE: {cert['gate_status']}")
if __name__=="__main__": main()
