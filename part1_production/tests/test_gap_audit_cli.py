def test_cli_import_without_matplotlib():
    from strata.cli import gap_audit
    assert callable(gap_audit.audit_sample)
