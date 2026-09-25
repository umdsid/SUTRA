from pathlib import Path
def test_no_approximate():
 t=(Path(__file__).parents[1]/'src/strata_hierarchy/v120/atlas.py').read_text().lower(); assert 'nearest' not in t
