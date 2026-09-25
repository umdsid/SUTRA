
from pathlib import Path
def test_current_report_fields():
    root=Path(__file__).parents[1]
    t=(root/"src/strata/cli/merger_driver_atlas_v120.py").read_text()
    assert "replay_stream_ready" in t
    assert "resolved_transitions" in t
    assert ".get(" in t
