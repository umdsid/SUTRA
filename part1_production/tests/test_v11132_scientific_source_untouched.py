from pathlib import Path

def test_current_replay_source_is_present():
    root = Path(__file__).parents[1]
    replay = root / "src/strata_hierarchy/v1113/replay.py"
    assert replay.is_file()
    assert replay.read_text().strip()
