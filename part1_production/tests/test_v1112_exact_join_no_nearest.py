import ast
from pathlib import Path

def test_no_nearest_imputation_in_source():
    root = Path(__file__).parents[1]
    p = root / "src/strata_hierarchy/v1112/recovery.py"
    src = p.read_text()
    tree = ast.parse(src)

    forbidden_calls = {
        "merge_asof",
        "interpolate",
        "reindex",
        "get_indexer",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            assert name not in forbidden_calls, f"forbidden approximate-join API used: {name}"

    assert 'on=["_eval_key","_node_key"]' in src
    assert 'on="_eval_key"' in src
    assert 'on="_node_key"' in src
