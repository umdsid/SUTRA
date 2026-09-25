
import argparse,json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from sutra.hierarchy.v111.audit import analyze_sample,SAMPLES
def main():
 p=argparse.ArgumentParser();p.add_argument('--project',default='.');p.add_argument('--config',default='configs/hierarchy_v111_terminal_landscape_audit.json');a=p.parse_args();project=Path(a.project).resolve();cfg=json.loads((project/a.config).read_text());out=project/'results/hierarchy_v111_terminal_landscape_audit';out.mkdir(parents=True,exist_ok=True)
 print('STRATA 1.1.1 | Terminal landscape reconstruction');print('No new reduction. No merge decisions. No node-count target.');print('Mining completed v1.1.0 production trajectories only.');print('Principal landmarks = Pareto layer-1 persistence basins; no scalarization.');print('Mass is descriptive/supporting evidence, not an existence veto.');print(f"Parallel specimen workers: {cfg['parallel_specimens']}\n")
 reps=[]
 with ProcessPoolExecutor(max_workers=cfg['parallel_specimens']) as ex:
  fs={ex.submit(analyze_sample,project,s,cfg,out):s for s in SAMPLES}
  for f in as_completed(fs):
   r=f.result();reps.append(r);print(f"[DONE] {r['sample']}: evals={r['evaluations']} basins={r['basins']} pareto={r['pareto_landmarks']} nodes={r['pareto_nodes']} {r['status']}")
 reps.sort(key=lambda r:r['sample']);gate='PASS' if all(r['status']=='PASS' for r in reps) else 'HOLD';cert={'stage':'terminal landscape reconstruction','strata_version':'1.1.1','TERMINAL_LANDSCAPE_AUDIT_GATE':gate,'NEW_MERGES_PERFORMED':False,'NODE_COUNT_TARGET_USED':False,'PARETO_SCALARIZATION_USED':False,'MASS_IS_EXISTENCE_VETO':False,'LANDMARKS_READY':gate=='PASS','sample_reports':reps};(out/'terminal_landscape_global_certificate.json').write_text(json.dumps(cert,indent=2));print(f'\nTERMINAL LANDSCAPE AUDIT GATE: {gate}');print('NEW MERGES PERFORMED: False');print('NODE-COUNT TARGET USED: False');print(f"PARETO LANDMARKS READY: {gate=='PASS'}");print(f"Certificate: {out/'terminal_landscape_global_certificate.json'}")
if __name__=='__main__':main()
