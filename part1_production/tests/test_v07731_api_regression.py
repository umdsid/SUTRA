import ast
from pathlib import Path


def test_legacy_isometry_test_uses_canonical_transport_api():
    p=Path(__file__).parents[1]/"tests"/"test_v077_g_isometry.py"
    s=p.read_text()

    assert "canonical_metric_sqrt" in s
    assert "apply_transport(H,Hinv,qi,qj,v)" in s.replace(" ","")
    assert "cholesky" not in s.lower()

    ast.parse(s)
