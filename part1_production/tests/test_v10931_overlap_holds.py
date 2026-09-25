import numpy as np,pandas as pd,tempfile
from pathlib import Path
from strata_hierarchy.v10931.replay_shards import replay_stream
def test_overlapping_batch_rejected():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/"merge_000000.parquet"
        pd.DataFrame({"super_i":[0,0],"super_j":[1,2]}).to_parquet(p)
        snaps,a,ok=replay_stream([p],np.arange(3),[1])
        assert not ok and int(a.iloc[0].batch_root_overlap)==1
