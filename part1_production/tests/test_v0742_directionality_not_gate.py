from pathlib import Path


def test_cli_does_not_use_directionality_as_admissibility_gate():
    p=Path(__file__).parents[1]/"src"/"strata"/"cli"/"hierarchy_effective_flow_v0742.py"
    s=p.read_text()
    gate=s.split('x["admissible"]=',1)[1].split(")",1)[0]
    assert "comm_directionality" not in gate
    assert "comm_reciprocity" not in gate
