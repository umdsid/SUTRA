import numpy as np,pandas as pd,tempfile
from pathlib import Path
from strata_hierarchy.v10931.replay_shards import replay_stream
def test_two_disjoint_pairs_drop_two_nodes():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/"merge_000000.parquet"
        pd.DataFrame({"microstep":[0,0],"super_i":[0,2],"super_j":[1,3]}).to_parquet(p)
        snaps,a,ok=replay_stream([p],np.arange(4),[2])
        assert ok and 2 in snaps and int(a.iloc[0].applied)==2
