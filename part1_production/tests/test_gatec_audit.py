from strata.cli.gatec_vmsi_audit import MECH_TOKENS
def test_mechanical_tokens():
    assert "pressure" in MECH_TOKENS
    assert "tension" in MECH_TOKENS
