def test_parallel_cli_import():
    from strata.cli.tranche3_4_fast import _run_sample
    assert callable(_run_sample)
