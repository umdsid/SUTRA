
from pathlib import Path
def test_engine_reuse():
    root=Path(__file__).parents[1]
    t=(root/"src/strata_hierarchy/v120/atlas.py").read_text()
    assert "from strata_hierarchy.v10931.replay_shards import" in t
    assert "replay_stream" in t
    assert "class DSU" not in t
