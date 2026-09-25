import json
from pathlib import Path
def test_mass_not_gate():
 c=json.loads((Path(__file__).parents[1]/'configs/hierarchy_v111_terminal_landscape_audit.json').read_text());assert 'mass' not in c['pareto_objectives']['maximize'];assert c['new_merges_performed'] is False
