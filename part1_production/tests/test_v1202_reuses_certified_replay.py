from pathlib import Path
def test_reuses():
 t=(Path(__file__).parents[1]/'src/strata_hierarchy/v120/atlas.py').read_text(); assert 'v10931.replay_shards' in t and 'replay_stream' in t and 'class DSU' not in t
