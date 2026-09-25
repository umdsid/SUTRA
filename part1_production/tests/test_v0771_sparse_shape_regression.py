import ast
from pathlib import Path

def test_cli_never_calls_len_on_sparse_covector_matrix_for_graph_size():
    p=Path(__file__).parents[1]/"src"/"strata"/"cli"/"discrete_transport_v077.py"
    s=p.read_text()
    assert "build_directed_cost_graph(geom,len(B))" not in s
    assert "build_directed_cost_graph(geom,B.shape[0])" in s

def test_patched_cli_parses():
    p=Path(__file__).parents[1]/"src"/"strata"/"cli"/"discrete_transport_v077.py"
    ast.parse(p.read_text())
