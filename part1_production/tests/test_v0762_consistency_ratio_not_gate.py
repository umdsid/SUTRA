from pathlib import Path

def test_consistency_ratio_is_not_used_as_a_gate_in_cli():
    p=Path(__file__).parents[1]/"src"/"strata"/"cli"/"pressure_field_finalize_v0762.py"
    s=p.read_text()
    assert '"consistency_ratio_is_gate":False' in s
