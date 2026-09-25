from __future__ import annotations
import json, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

def writepq(df,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    pq.write_table(pa.Table.from_pandas(df,preserve_index=False),tmp,compression="zstd")
    tmp.replace(path)

def writejson(obj,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,sort_keys=True)+"\n")
    tmp.replace(path)

def sha256_file(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()

class FrozenProductionLedger:
    def __init__(self,root,sample):
        self.root=Path(root)/sample
        self.root.mkdir(parents=True,exist_ok=True)
        for d in [
            "merges","ancestry","checkpoints","evaluations","states","terminal"
        ]:
            (self.root/d).mkdir(exist_ok=True)
        self.steps=self.root/"microsteps.jsonl"

    def append_step(self,row):
        with self.steps.open("a") as f:
            f.write(json.dumps(row,sort_keys=True)+"\n")
            f.flush()

    def merge_batch(self,step,selected,ancestry):
        if len(selected):
            x=selected.copy();x.insert(0,"microstep",int(step))
            writepq(x,self.root/"merges"/f"merge_{step:06d}.parquet")
        if len(ancestry):
            writepq(ancestry,self.root/"ancestry"/f"ancestry_{step:06d}.parquet")

    def checkpoint(self,step,labels,lineage_active,run_state):
        p=self.root/"checkpoints"/f"labels_{step:06d}.npz"
        tmp=p.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,labels=np.asarray(labels,np.int64))
        tmp.replace(p)
        writepq(
            lineage_active,
            self.root/"checkpoints"/f"lineage_active_{step:06d}.parquet"
        )
        writejson(run_state,self.root/"checkpoints"/f"run_state_{step:06d}.json")

    def evaluation(
        self,idx,step,labels,node_state,superedges,transport,
        block_table,terminal_state,observable_row,lineage_active,
        derivative_path,required_payload
    ):
        edir=self.root/"evaluations"/f"eval_{idx:04d}"
        edir.mkdir(parents=True,exist_ok=True)

        lp=edir/"active_labels.npz"
        tmp=lp.with_suffix(".npz.tmp")
        with tmp.open("wb") as f:
            np.savez_compressed(f,labels=np.asarray(labels,np.int64))
        tmp.replace(lp)

        writepq(node_state,edir/"node_state.parquet")
        writepq(superedges,edir/"superedges.parquet")
        writepq(transport,edir/"transport.parquet")
        writepq(block_table,edir/"block_terminal_diagnostics.parquet")
        writepq(lineage_active,edir/"active_lineage.parquet")
        writejson(observable_row,edir/"observable_state.json")
        writejson(terminal_state,edir/"terminal_rule_state.json")

        manifest={
            "evaluation_index":int(idx),
            "microstep":int(step),
            "required_payload":list(required_payload),
            "payload":{
                "active_label_vector":"active_labels.npz",
                "merge_ancestry":"../../ancestry/",
                "node_state":"node_state.parquet",
                "expression":{
                    "node_state":"node_state.parquet",
                    "observables":"observable_state.json"
                },
                "go_msigdb":{
                    "edge_state":"superedges.parquet",
                    "observables":"observable_state.json",
                    "fields":["functional_distance","functional_similarity"]
                },
                "cellchat":{
                    "edge_state":"superedges.parquet",
                    "observables":"observable_state.json"
                },
                "mechanics_confidence_provenance":{
                    "edge_state":"superedges.parquet",
                    "observables":"observable_state.json"
                },
                "pressure":{
                    "edge_state":"superedges.parquet",
                    "observables":"observable_state.json"
                },
                "topology":"observable_state.json",
                "local_geometry":{
                    "node_state":"node_state.parquet",
                    "edge_state":"superedges.parquet",
                    "observables":"observable_state.json"
                },
                "directional_state":{
                    "transport":"transport.parquet",
                    "observables":"observable_state.json"
                },
                "transport":"transport.parquet",
                "geodesics":"observable_state.json",
                "holonomy":"observable_state.json",
                "holonomy_density":"observable_state.json",
                "component_structure":{
                    "node_state":"node_state.parquet",
                    "observables":"observable_state.json"
                },
                "first_scale_differences":str(Path(derivative_path).name),
                "second_scale_differences":str(Path(derivative_path).name),
                "lineage":"active_lineage.parquet",
                "terminal_rule":"terminal_rule_state.json",
            }
        }
        writejson(manifest,edir/"payload_manifest.json")
        return edir

    def latest_checkpoint(self):
        files=sorted((self.root/"checkpoints").glob("run_state_*.json"))
        if not files:return None
        p=files[-1]
        step=int(p.stem.split("_")[-1])
        state=json.loads(p.read_text())
        labels=np.load(
            self.root/"checkpoints"/f"labels_{step:06d}.npz",
            allow_pickle=False
        )["labels"]
        lineage=pd.read_parquet(
            self.root/"checkpoints"/f"lineage_active_{step:06d}.parquet"
        )
        return step,state,labels,lineage
