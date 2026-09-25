from pathlib import Path
def test_no_new_merges():
 t=(Path(__file__).parents[1]/'src/strata_hierarchy/v111/audit.py').read_text();assert 'merge_batch' not in t
