from __future__ import annotations
import argparse, json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sutra.geometry_benchmark.components import ComponentConfig, select_primary_component
from sutra.geometry_benchmark.baselines import (
    polygons_from_boundaries, make_raster, constrained_nearest_completion,
    adjacency_from_labels
)


RADII=(5.0, 6.0, 7.0, 7.5, 8.0)


def _find(root, patterns):
    if isinstance(patterns, str):
        patterns=[patterns]
    for pat in patterns:
        hits=list(Path(root).rglob(pat))
        if hits:
            return sorted(hits)[0]
    return None


def evaluate(labels, observed, meta, gateb_edges):
    toid=meta["label_to_cell"]
    pred_lab=adjacency_from_labels(labels)
    pred={
        tuple(sorted((toid[a],toid[b])))
        for a,b in pred_lab
        if a in toid and b in toid
    }
    truth={tuple(sorted((str(a),str(b)))) for a,b in gateb_edges}
    tp=len(pred & truth)
    precision=tp/max(len(pred),1)
    recall=tp/max(len(truth),1)

    # graph connectivity on the completed contact graph
    ids=list(toid.values())
    adj={cid:set() for cid in ids}
    for a,b in pred:
        adj[a].add(b); adj[b].add(a)
    seen=set(); sizes=[]
    for cid in ids:
        if cid in seen: continue
        stack=[cid]; seen.add(cid); n=0
        while stack:
            u=stack.pop(); n+=1
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); stack.append(v)
        sizes.append(n)
    lcc=max(sizes) if sizes else 0

    labs=np.array(sorted(toid.keys()),dtype=int)
    old=np.bincount(observed.ravel(), minlength=int(labs.max())+1)[labs].astype(float)
    new=np.bincount(labels.ravel(), minlength=int(labs.max())+1)[labs].astype(float)
    ratio=(new+1)/(old+1)

    return {
        "n_contacts":len(pred),
        "gateB_contact_precision":float(precision),
        "gateB_contact_recall":float(recall),
        "gateB_contact_f1":float(2*precision*recall/max(precision+recall,1e-15)),
        "new_contact_burden":float(len(pred-truth)/max(len(pred),1)),
        "lost_gateB_fraction":float(len(truth-pred)/max(len(truth),1)),
        "largest_component_fraction":float(lcc/max(len(ids),1)),
        "n_components":len(sizes),
        "median_abs_log_area_ratio":float(np.median(np.abs(np.log(ratio)))),
        "q95_abs_log_area_ratio":float(np.quantile(np.abs(np.log(ratio)),.95)),
        "observed_pixels_preserved":bool(
            np.all(labels[observed>0] == observed[observed>0])
        ),
    }


def calibrate_one(project_s, sample, radii):
    project=Path(project_s)
    sroot=project/"data"/sample

    cells=pd.read_parquet(_find(sroot,"*cells.parquet"))
    cells["cell_id"]=cells.cell_id.astype(str)
    primary, comp=select_primary_component(cells, ComponentConfig())
    pcells=cells[cells.cell_id.isin(primary)].copy()

    bdf=pd.read_parquet(_find(sroot,"*cell_boundaries*.parquet"))
    bdf["cell_id"]=bdf.cell_id.astype(str)
    bdf=bdf[bdf.cell_id.isin(primary)].copy()
    polys=polygons_from_boundaries(bdf)

    epath=project/"results"/"tranche2_1_gateB"/sample/"certified_interfaces.parquet"
    edf=pd.read_parquet(epath)
    edf["cell_i"]=edf.cell_i.astype(str)
    edf["cell_j"]=edf.cell_j.astype(str)
    if "confidence_class" in edf.columns:
        edf=edf[
            edf.confidence_class.astype(str).isin(["admissible","high_confidence"])
        ].copy()
    edf=edf[
        edf.cell_i.isin(primary) & edf.cell_j.isin(primary)
    ].copy()
    edges=list(zip(edf.cell_i, edf.cell_j))

    observed, meta=make_raster(polys, pcells)

    rows=[]
    saved={}
    for radius in radii:
        labels, m = constrained_nearest_completion(
            observed, meta, edges, max_distance=float(radius)
        )
        ev=evaluate(labels, observed, meta, edges)
        ev["radius"]=float(radius)
        ev["remaining_unassigned_fraction"]=float(
            m["remaining_unassigned_fraction"]
        )
        ev["production_gate_pass"]=bool(
            ev["observed_pixels_preserved"] and
            ev["gateB_contact_recall"] >= 0.95 and
            ev["new_contact_burden"] <= 0.35 and
            ev["largest_component_fraction"] >= 0.90
        )
        rows.append(ev)
        saved[float(radius)]=labels

    # Choose the smallest radius that passes, because that minimizes inferred
    # territory extension. If none pass, do not select one.
    passing=[r for r in rows if r["production_gate_pass"]]
    chosen=min(passing, key=lambda r:r["radius"]) if passing else None

    out=project/"results"/"tranche2_2e_radius_calibration"/sample
    out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/"radius_sweep.csv",index=False)

    if chosen is not None:
        labels=saved[float(chosen["radius"])]
        np.savez_compressed(
            out/"selected_geometry.npz",
            labels=labels,
            observed=observed
        )

    report={
        "sample":sample,
        "primary_component":comp,
        "radii_tested":[float(x) for x in radii],
        "results":rows,
        "selected_radius":None if chosen is None else float(chosen["radius"]),
        "status":"PASS" if chosen is not None else "HOLD",
        "selection_rule":"smallest radius satisfying all predeclared full-geometry production gates",
    }
    (out/"radius_calibration.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()

    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    reports=[]
    with ProcessPoolExecutor(max_workers=min(a.workers,3)) as ex:
        futs={
            ex.submit(calibrate_one,str(project),s,RADII):s
            for s in samples
        }
        for fut in as_completed(futs):
            s=futs[fut]
            r=fut.result()
            reports.append(r)
            print(f"[DONE] {s}: selected_radius={r['selected_radius']} status={r['status']}")
            for q in r["results"]:
                print(
                    f"    r={q['radius']:.1f} "
                    f"recall={100*q['gateB_contact_recall']:.2f}% "
                    f"new={100*q['new_contact_burden']:.2f}% "
                    f"LCC={100*q['largest_component_fraction']:.2f}% "
                    f"pass={q['production_gate_pass']}"
                )

    reports.sort(key=lambda r:samples.index(r["sample"]))
    all_pass=all(r["status"]=="PASS" for r in reports)

    # For production reproducibility prefer one global radius if possible.
    # Determine smallest tested radius that passes in every specimen.
    common=[]
    for radius in RADII:
        ok=True
        for r in reports:
            row=next(x for x in r["results"] if x["radius"]==radius)
            ok &= bool(row["production_gate_pass"])
        if ok:
            common.append(radius)
    global_radius=min(common) if common else None

    cert={
        "strata_version":"0.5.8",
        "tranche":"2.2e",
        "method":"constrained_nearest",
        "sample_reports":reports,
        "global_selected_radius":global_radius,
        "freeze_status":"PASS" if global_radius is not None else "HOLD",
        "global_rule":"select the smallest tested radius satisfying the same production geometry gates in every specimen",
    }
    out=project/"results"/"tranche2_2e_radius_calibration"
    (out/"tranche2_2e_radius_freeze_certificate.json").write_text(
        json.dumps(cert,indent=2)
    )

    print()
    print(f"RADIUS FREEZE: {cert['freeze_status']}")
    print(f"Global selected radius: {global_radius}")
    print(f"Certificate: {out/'tranche2_2e_radius_freeze_certificate.json'}")


if __name__=="__main__":
    main()
