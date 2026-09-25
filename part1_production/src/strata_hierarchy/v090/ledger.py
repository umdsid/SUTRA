"""Streaming Flow Ledger utilities for STRATA v0.9.0."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class FlowLedger:
    def __init__(self,root,sample,archive_all_candidates=True):
        self.root=Path(root)/sample
        self.root.mkdir(parents=True,exist_ok=True)
        self.steps=self.root/"steps.jsonl"
        self.events=self.root/"merge_events"
        self.candidates=self.root/"candidate_boundaries"
        self.checkpoints=self.root/"label_checkpoints"
        self.heavy=self.root/"heavy_landmarks"
        for p in [self.events,self.candidates,self.checkpoints,self.heavy]:
            p.mkdir(parents=True,exist_ok=True)
        self.archive_all_candidates=archive_all_candidates

    def append_step(self,row):
        with self.steps.open("a") as f:
            f.write(json.dumps(row,sort_keys=True)+"\n")

    def write_merges(self,step,df):
        if len(df)==0:return
        x=df.copy()
        x.insert(0,"microstep",int(step))
        pq.write_table(
            pa.Table.from_pandas(x,preserve_index=False),
            self.events/f"step_{step:06d}.parquet",
            compression="zstd",
        )

    def write_candidates(self,step,df):
        if not self.archive_all_candidates:return
        x=df.copy()
        x.insert(0,"microstep",int(step))
        pq.write_table(
            pa.Table.from_pandas(x,preserve_index=False),
            self.candidates/f"step_{step:06d}.parquet",
            compression="zstd",
        )

    def checkpoint_labels(self,step,labels):
        np.savez_compressed(
            self.checkpoints/f"labels_{step:06d}.npz",
            labels=np.asarray(labels,dtype=np.int64),
        )

    def write_heavy_marker(self,step,row):
        (self.heavy/f"landmark_{step:06d}.json").write_text(
            json.dumps(row,indent=2)+"\n"
        )
