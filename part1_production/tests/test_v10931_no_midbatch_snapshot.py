import numpy as np,pandas as pd,tempfile
from pathlib import Path
from strata_hierarchy.v10931.replay_shards import replay_stream
def test_midbatch_count_is_not_synthesized():
    with tempfile.TemporaryDirectory() as t:
        p=Path(t)/"merge_000000.parquet"
        pd.DataFrame({"super_i":[0,2],"super_j":[1,3]}).to_parquet(p)
        snaps,a,ok=replay_stream([p],np.arange(4),[3,2])
        assert ok and 2 in snaps and 3 not in snaps
