from __future__ import annotations
import argparse,json,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
import tifffile

from strata_native_mechanics.geometry import (
    GeometryConfig,extract_interfaces,extract_junctions,extract_cell_centroids
)
from strata_native_mechanics.patches import partition_core_cells,add_halo
from strata_native_mechanics.solver import SolverConfig,solve_patch
from strata_native_mechanics.reconcile import reconcile_patch_solutions


def run_sample(project_s,sample,patch_size,halo):
    project=Path(project_s)
    t0=time.time()

    # Prefer the frozen r=5 mechanics mask; keep gaps/background intact.
    p=project/"results"/"production_geometry_r5"/sample/"production_segmentation.tif"
    if not p.exists():
        raise FileNotFoundError(p)
    mask=tifffile.imread(p).astype(np.int32)

    out=project/"results"/"native_mechanics_v060"/sample
    out.mkdir(parents=True,exist_ok=True)

    gcfg=GeometryConfig()
    interfaces,bg,bg_lab=extract_interfaces(mask,gcfg)
    junctions=extract_junctions(mask,interfaces,bg_lab)
    cells=extract_cell_centroids(mask)

    interfaces.to_parquet(out/"interfaces.parquet",index=False)
    junctions.to_json(out/"junctions.jsonl",orient="records",lines=True)
    bg.to_parquet(out/"background_components.parquet",index=False)
    cells.to_parquet(out/"cells.parquet",index=False)

    cores=partition_core_cells(interfaces,patch_size=patch_size)
    scfg=SolverConfig()
    patch_results=[]
    patch_rows=[]
    for k,core in enumerate(cores):
        patch=add_halo(core,interfaces,hops=halo)
        r=solve_patch(patch,interfaces,junctions,cells,scfg)
        patch_results.append(r)
        ident=r.get("identifiability",{})
        patch_rows.append({
            "patch_id":k,
            "n_core_cells":len(core),
            "n_patch_cells":len(patch),
            "status":r.get("status"),
            "residual_rms":r.get("residual_rms"),
            "residual_q95":r.get("residual_q95"),
            "structural_rank_fraction":ident.get("structural_rank_fraction"),
            "structurally_identifiable":ident.get("structurally_identifiable"),
            "n_rows":ident.get("n_rows"),
            "n_variables":ident.get("n_variables"),
        })

    tdf,pdf,bdf=reconcile_patch_solutions(patch_results)
    tdf.to_parquet(out/"tensions.parquet",index=False)
    pdf.to_parquet(out/"pressures.parquet",index=False)
    bdf.to_parquet(out/"boundary_pressures.parquet",index=False)
    pd.DataFrame(patch_rows).to_csv(out/"patch_audit.csv",index=False)

    pass_frac=float(np.mean([r.get("status")=="PASS" for r in patch_results])) if patch_results else 0.0
    ident_frac=float(np.mean([r.get("identifiability",{}).get("structurally_identifiable",False)
                              for r in patch_results])) if patch_results else 0.0
    median_q95=float(np.nanmedian([r.get("residual_q95",np.nan) for r in patch_results])) if patch_results else None

    summary={
        "sample":sample,
        "method":"STRATA-native boundary-aware inverse mechanics",
        "version":"0.6.0",
        "n_cells":int(mask.max()),
        "n_interfaces":int(len(interfaces)),
        "n_junctions":int(len(junctions)),
        "n_background_components":int(len(bg)),
        "n_internal_gaps":int((bg.kind=="internal_gap").sum()) if len(bg) else 0,
        "n_exterior_components":int((bg.kind=="exterior").sum()) if len(bg) else 0,
        "n_patches":int(len(cores)),
        "patch_pass_fraction":pass_frac,
        "patch_structural_identifiability_fraction":ident_frac,
        "median_patch_residual_q95":median_q95,
        "runtime_seconds":float(time.time()-t0),
        "status":"PASS" if pass_frac>=0.9 else "HOLD",
    }
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    return summary


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--patch-size",type=int,default=300)
    ap.add_argument("--halo",type=int,default=1)
    ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]
    rs=[]
    with ProcessPoolExecutor(max_workers=min(3,a.workers)) as ex:
        futs={ex.submit(run_sample,str(project),s,a.patch_size,a.halo):s for s in samples}
        for f in as_completed(futs):
            r=f.result();rs.append(r)
            print(
                f"[DONE] {r['sample']}: cells={r['n_cells']:,} "
                f"interfaces={r['n_interfaces']:,} junctions={r['n_junctions']:,} "
                f"gaps={r['n_internal_gaps']} patches={r['n_patches']} "
                f"patch_pass={100*r['patch_pass_fraction']:.1f}% "
                f"ident={100*r['patch_structural_identifiability_fraction']:.1f}% "
                f"status={r['status']}"
            )
    rs.sort(key=lambda x:samples.index(x["sample"]))
    overall={
        "strata_version":"0.6.0",
        "stage":"native boundary-aware mechanics",
        "sample_reports":rs,
        "gate_status":"PASS" if all(r["status"]=="PASS" for r in rs) else "HOLD",
        "note":"This gate certifies numerical execution only; identifiability is reported separately and is not silently regularized away."
    }
    out=project/"results"/"native_mechanics_v060"
    (out/"native_mechanics_certificate.json").write_text(json.dumps(overall,indent=2))
    print(f"\nNATIVE MECHANICS GATE: {overall['gate_status']}")
    print(f"Certificate: {out/'native_mechanics_certificate.json'}")

if __name__=="__main__":main()
