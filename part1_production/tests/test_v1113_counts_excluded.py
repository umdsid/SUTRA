
from pathlib import Path
def test_generic_landmark_count_not_explicit_node_field():
    root=Path(__file__).parents[1]
    t=(root/"src/strata_hierarchy/v1113/replay.py").read_text()
    assert '"landmarks"' not in t.split("explicit = low in {",1)[1].split("}",1)[0]
