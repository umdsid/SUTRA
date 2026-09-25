
import json
from pathlib import Path
def test_contract():
    r=Path(__file__).parents[1]
    c=json.loads((r/"configs/brain_publication_complete_v110.json").read_text())
    x=c["contract"]
    assert x["postprocessing_only"] and x["hierarchy_frozen"]
    assert not x["new_merges"] and not x["thresholds_modified"]
    assert not x["cross_tissue_comparison"]
