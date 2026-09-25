#!/usr/bin/env python3
from pathlib import Path
import argparse, hashlib, json
import pandas as pd
import pyarrow as pa, pyarrow.parquet as pq
SAMPLES=("healthy_reference","nondiseased_kidney")
KEEP=["microstep","super_i","super_j","molecular_distance","mechanics_support_fraction","comm_support","molecular_effective","mechanics_effective","communication_support_effective","geometry_pair_cost","geometry_effective","topology_effective","evidence_coverage","composite_merge_cost","admissible"]
def sha(p):
 h=hashlib.sha256()
 with p.open("rb") as f:
  for x in iter(lambda:f.read(1048576),b""): h.update(x)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument("--root",required=True); a=ap.parse_args()
 root=Path(a.root).expanduser().resolve(); out=root/"results/suppfig2_fair_matched_candidate_v152"/"_parquet_bridge"; out.mkdir(parents=True,exist_ok=True); manifest={}
 for s in SAMPLES:
  src=root/"results/hierarchy_v0911_specimen_local_contextual_flow/ledger"/s/"candidate_boundaries/step_000000.parquet"
  df=pd.read_parquet(src); missing=[c for c in KEEP if c not in df.columns]
  if missing: raise RuntimeError(f"{s}: missing {missing}")
  slim=df[KEEP].copy(); dst=out/f"{s}__step_000000_bridge.parquet"
  pq.write_table(pa.Table.from_pandas(slim,preserve_index=False),dst,compression="snappy",use_dictionary=False,write_statistics=False)
  manifest[s]={"source":str(src),"source_sha256":sha(src),"source_rows":len(df),"bridge":str(dst),"bridge_sha256":sha(dst),"bridge_rows":len(slim),"columns":KEEP,"writer_pyarrow":pa.__version__}
  print("WROTE",dst,slim.shape)
 (out/"bridge_manifest.json").write_text(json.dumps(manifest,indent=2))
if __name__=="__main__": main()
