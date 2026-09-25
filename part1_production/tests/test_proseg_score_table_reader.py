import pandas as pd
from pathlib import Path
from strata.cli.tranche2_2b_score_proseg import _read_table

def test_reads_csv_gz(tmp_path):
    p=tmp_path/"transcript_metadata.csv.gz"
    pd.DataFrame({"assignment":[1,2]}).to_csv(p,index=False,compression="gzip")
    x,src=_read_table(tmp_path/"transcript_metadata")
    assert len(x)==2
    assert src.endswith(".csv.gz")
