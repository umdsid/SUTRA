import json
from pathlib import Path
def test_contract():
 r=Path(__file__).parents[1];c=json.loads((r/"configs/brain_publication_pipeline_v1.json").read_text());s=c["scientific_contract"];assert s["hierarchy_frozen"] and not s["new_merges"] and not s["thresholds_modified"] and not s["cross_tissue_comparison"]
