import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
TOK=("pressure","tension","stress")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--project-root",default="."); a=ap.parse_args(); project=Path(a.project_root).resolve()
    rs=[]
    for s in ["alzheimers","gbm_reference_addon","healthy_reference"]:
        p=project/"results"/"production_mechanics_vmsi_r5_connected"/s/"vmsi_results.csv"
        if not p.exists(): rs.append({"sample":s,"status":"HOLD"}); continue
        df=pd.read_csv(p); num=df.select_dtypes(include=[np.number]); mech=[c for c in num.columns if any(t in c.lower() for t in TOK)]
        finite={c:float(np.isfinite(pd.to_numeric(df[c],errors="coerce")).mean()) for c in mech}
        std={c:float(np.nanstd(pd.to_numeric(df[c],errors="coerce"))) for c in mech}
        geom=json.loads((project/"results"/"production_geometry_r5_connected"/s/"connectivity_repair.json").read_text())
        cov=min(len(df)/max(geom.get("largest_component_cells",len(df)),1),1.0)
        ok=(cov>=.85 and mech and min(finite.values(),default=0)>=.95 and any(v>0 for v in std.values()))
        rs.append({"sample":s,"status":"PASS" if ok else "HOLD","cell_result_coverage":cov,"mechanical_columns":mech,"finite_fraction_by_column":finite,"std_by_column":std})
    overall={"strata_version":"0.5.10","sample_reports":rs,"gate_status":"PASS" if all(r["status"]=="PASS" for r in rs) else "HOLD"}
    out=project/"results"/"production_mechanics_vmsi_r5_connected";out.mkdir(parents=True,exist_ok=True)
    (out/"gateC_certificate.json").write_text(json.dumps(overall,indent=2))
    for r in rs: print(f"[{r['sample']}] {r['status']} coverage={100*r.get('cell_result_coverage',0):.1f}% fields={len(r.get('mechanical_columns',[]))}")
    print(f"\nGATE C: {overall['gate_status']}")
if __name__=="__main__": main()
