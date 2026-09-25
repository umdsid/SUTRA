
import pandas as pd
from pathlib import Path
from strata_hierarchy.v120.atlas import parquet_num_rows
def test_parquet_row_count(tmp_path):
    p=tmp_path/"cells.parquet"
    pd.DataFrame({"cell_id":["a","b","c"]}).to_parquet(p,index=False)
    assert parquet_num_rows(p)==3
