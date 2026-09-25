
from pathlib import Path
from strata_hierarchy.v120.atlas import canonical_native_sources
def test_canonical(tmp_path):
    d=tmp_path/"data"/"s"; d.mkdir(parents=True)
    (d/"cell_feature_matrix.h5").write_bytes(b"x")
    (d/"cells.parquet").write_bytes(b"x")
    x=canonical_native_sources(tmp_path,"s")
    assert len(x["expression_h5"])==1 and len(x["cells_parquet"])==1
