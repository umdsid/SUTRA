
from pathlib import Path
def test_cli_meta_contract():
    root=Path(__file__).parents[1]
    t=(root/"src/strata/cli/exact_frozen_terminal_replay_v1113.py").read_text()
    assert 'meta.get("used_function")' in t
    assert 'meta.get("history_representation_used")' in t
    assert 'meta["used_function"]' not in t
