from __future__ import annotations
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np,pandas as pd,tifffile

from strata.connectivity.digital_bridge import repair_diagonal_only_labels
from strata.production_geometry.materialize_radius5 import adjacency_from_labels,component_stats

def one(project_s,sample):
    project=Path(project_s)
    src=project/"results"/"production_geometry_r5_connectivity"/sample
    mask=tifffile.imread(src/"production_segmentation_connectivity.tif").astype(np.int32)

    repaired,audit=repair_diagonal_only_labels(mask)

    out=project/"results"/"production_geometry_r5_digital"/sample
    out.mkdir(parents=True,exist_ok=True)
    tifffile.imwrite(out/"production_segmentation_digital.tif",
                     repaired.astype(np.uint32),compression="zlib")
    np.savez_compressed(out/"production_geometry_digital.npz",labels=repaired)

    cross=pd.read_parquet(src/"label_to_cell_id.parquet")
    ids=dict(zip(cross.mechanics_label.astype(int),cross.cell_id.astype(str)))
    pairs=adjacency_from_labels(repaired)
    pred={tuple(sorted((ids[a],ids[b]))) for a,b in pairs}

    e=pd.read_csv(project/"results"/"tranche2_2b_geometry_benchmark"/sample/"primary_gateB_edges.csv")
    truth={tuple(sorted((str(a),str(b)))) for a,b in zip(e.cell_i,e.cell_j)}
    tp=len(pred&truth)
    recall=tp/max(len(truth),1)
    new=len(pred-truth)/max(len(pred),1)
    comps=component_stats(pairs,len(ids))

    report={
        "sample":sample,
        **audit,
        "gateB_contact_recall":float(recall),
        "new_contact_burden":float(new),
        **comps,
    }
    # This tranche is an audit/repair gate, not yet VMSI readiness:
    # require all diagonal-only cases fixed and production topology preserved.
    report["diagonal_repair_gate_pass"]=bool(
        audit["diagonal_bridge_failures"]==0 and
        recall>=.95 and new<=.35 and
        report["largest_component_fraction"]>=.90
    )
    (out/"minimal_digital_connectivity.json").write_text(json.dumps(report,indent=2))
    return report

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--project-root",default=".")
    a=ap.parse_args();project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"];rs=[]
    with ProcessPoolExecutor(max_workers=3) as ex:
        futs={ex.submit(one,str(project),s):s for s in samples}
        for f in as_completed(futs):
            r=f.result();rs.append(r)
            print(
                f"[DONE] {r['sample']}: "
                f"4c {r['pre_disconnected_4']} -> {r['post_disconnected_4']} | "
                f"8c {r['pre_disconnected_8']} -> {r['post_disconnected_8']} | "
                f"diagonal={r['diagonal_only_labels']} "
                f"bridge_px={r['bridge_pixels_added']} "
                f"bridge_fail={r['diagonal_bridge_failures']} "
                f"recall={100*r['gateB_contact_recall']:.2f}% "
                f"new={100*r['new_contact_burden']:.2f}% "
                f"LCC={100*r['largest_component_fraction']:.2f}% "
                f"pass={r['diagonal_repair_gate_pass']}"
            )
    rs.sort(key=lambda x:samples.index(x["sample"]))
    out=project/"results"/"production_geometry_r5_digital";out.mkdir(parents=True,exist_ok=True)
    cert={"strata_version":"0.5.15","sample_reports":rs,
          "gate_status":"PASS" if all(r["diagonal_repair_gate_pass"] for r in rs) else "HOLD",
          "note":"Remaining post-4c failures are hard cases; VMSI remains blocked until separately adjudicated."}
    (out/"minimal_digital_connectivity_certificate.json").write_text(json.dumps(cert,indent=2))
    print(f"\nMINIMAL DIGITAL REPAIR GATE: {cert['gate_status']}")
if __name__=="__main__":main()
