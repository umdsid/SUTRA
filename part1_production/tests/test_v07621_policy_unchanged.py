from pathlib import Path
def test_transport_policy_remains_potential_only():
    p=Path(__file__).parents[1]/"src"/"strata"/"cli"/"pressure_field_finalize_v0762.py"
    s=p.read_text()
    assert '"nonpotential_residual_for_transport":False' in s
    assert '"constraints_removed":False' in s
    assert '"outliers_clipped":False' in s
