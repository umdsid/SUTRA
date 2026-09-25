
import ast
from pathlib import Path
def test_no_approximate_join():
    root=Path(__file__).parents[1]
    t=(root/"src/strata_hierarchy/v1113/replay.py").read_text()
    tree=ast.parse(t)
    forbidden={"merge_asof","interpolate","get_indexer"}
    for n in ast.walk(tree):
        if isinstance(n,ast.Call):
            f=n.func
            name=f.id if isinstance(f,ast.Name) else (f.attr if isinstance(f,ast.Attribute) else None)
            assert name not in forbidden
