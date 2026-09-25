
import json
from pathlib import Path
def test_mass_semantics():
    root=Path(__file__).parents[1]
    c=json.loads((root/"configs/hierarchy_v112_final_supported_hierarchy_landscape.json").read_text())
    assert c["mass_is_existence_veto"] is False
    assert c["mass_is_pareto_objective"] is False
    assert "mass_support" not in c["pareto_objectives"]
