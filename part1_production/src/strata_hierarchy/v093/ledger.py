from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

class ProductionLedger:
    def __init__(self,root,sample):
        self.root=Path(root)/sample
        self.root.mkdir(parents=True,exist_ok=True)
        for d in ["merges","checkpoints","landmarks","states"]:
            (self.root/d).mkdir(exist_ok=True)
        self.steps=self.root/"microsteps.jsonl"

    def append_step(self,row):
        with self.steps.open("a") as f:
            f.write(json.dumps(row,sort_keys=True)+"\n")

    def merges(self,step,df):
        if len(df):
            x=df.copy(); x.insert(0,"microstep",int(step))
            pq.write_table(pa.Table.from_pandas(x,preserve_index=False),
                           self.root/"merges"/f"merge_{step:06d}.parquet",
                           compression="zstd")

    def checkpoint(self,step,labels):
        np.savez_compressed(self.root/"checkpoints"/f"labels_{step:06d}.npz",
                            labels=np.asarray(labels,np.int64))

    def landmark(self,idx,row,superedges=None,node_state=None,transport=None):
        p=self.root/"landmarks"/f"landmark_{idx:04d}.json"
        p.write_text(json.dumps(row,indent=2)+"\n")
        if superedges is not None:
            pq.write_table(pa.Table.from_pandas(superedges,preserve_index=False),
                           self.root/"states"/f"superedges_{idx:04d}.parquet",
                           compression="zstd")
        if node_state is not None:
            pq.write_table(pa.Table.from_pandas(node_state,preserve_index=False),
                           self.root/"states"/f"nodes_{idx:04d}.parquet",
                           compression="zstd")
        if transport is not None:
            pq.write_table(pa.Table.from_pandas(transport,preserve_index=False),
                           self.root/"states"/f"transport_{idx:04d}.parquet",
                           compression="zstd")
