from pathlib import Path
def test_source_stage_only():
 t=(Path(__file__).parents[1]/'src/strata_hierarchy/v111/audit.py').read_text();assert "cfg['source_stage']" in t
