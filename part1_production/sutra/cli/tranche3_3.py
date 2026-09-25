from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
from strata.mechanics.identifiability import baseline_field_summary,structural_identifiability,perturbation_suite,certify_identifiability

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project-root',default='.'); args=ap.parse_args()
    project=Path(args.project_root).resolve(); gatec_path=project/'results'/'tranche3_2_gateC'/'gateC_certificate.json'
    if not gatec_path.exists(): raise SystemExit(f'Missing Gate C certificate: {gatec_path}')
    gatec=json.loads(gatec_path.read_text())
    if gatec.get('gateC_status')!='PASS': raise SystemExit('Gate C is not PASS; audit blocked.')
    outroot=project/'results'/'tranche3_3_identifiability'; outroot.mkdir(parents=True,exist_ok=True)
    print('STRATA 0.3.4 | Tranche 3.3 | Mechanical identifiability audit')
    reports=[]
    for k,rep in enumerate(gatec['sample_reports'],1):
        name=rep['sample']; print(f'[{k}/{len(gatec["sample_reports"])}] {name}')
        src=project/'results'/'tranche3_2_gateC'/name
        raw=pd.read_parquet(src/'raw_interface_geometry.parquet'); edges=pd.read_parquet(src/'interface_mechanics.parquet'); junctions=pd.read_parquet(src/'junction_geometry.parquet'); pressure=pd.read_parquet(src/'cell_pressure.parquet')
        ids=pressure.cell_id.astype(str).tolist(); bt=edges.tension_like.to_numpy(float); bp=pressure.pressure_like.to_numpy(float); jtol=float(rep['calibrated_junction_tolerance'])
        summary=baseline_field_summary(edges,pressure); structural=structural_identifiability(edges,ids,len(junctions)); pert=perturbation_suite(raw,edges,junctions,ids,bt,bp,jtol); audit=certify_identifiability(summary,structural,pert)
        report={'sample':name,'calibrated_junction_tolerance':jtol,'field_summary':summary,'structural_identifiability':structural,'audit':audit}
        out=outroot/name; out.mkdir(parents=True,exist_ok=True); pert.to_csv(out/'mechanical_perturbation_stability.csv',index=False); (out/'identifiability_summary.json').write_text(json.dumps(report,indent=2)); reports.append(report)
        print(f"    Neff={100*summary['tension_effective_support_fraction']:.2f}% top1mass={100*summary['tension_mass_top1pct']:.1f}% struct_rank={100*structural['structural_rank_fraction_of_unknowns']:.1f}% min_rho_tau={audit['minimum_tension_spearman']:.3f} min_rho_p={audit['minimum_pressure_spearman']:.3f} top5_overlap={audit['minimum_top5_tension_overlap']:.3f} status={audit['status']}")
    overall={'strata_version':'0.3.4','tranche':'3.3','audit':'mechanical_identifiability_and_robustness','gateC_source':str(gatec_path),'sample_reports':reports,'mechanics_freeze_status':'PASS' if all(r['audit']['status']=='PASS' for r in reports) else 'FAIL'}
    (outroot/'mechanics_freeze_certificate.json').write_text(json.dumps(overall,indent=2)); print(); print(f"Mechanics freeze audit: {overall['mechanics_freeze_status']}"); print(f"Certificate: {outroot/'mechanics_freeze_certificate.json'}")
if __name__=='__main__': main()
