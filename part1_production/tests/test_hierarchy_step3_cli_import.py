def test_step3_cli_imports_cleanly():
    # This catches missing names/imports at module load before the expensive
    # three-specimen process pool starts.
    import strata.cli.level0_materialization_v070_step3 as mod
    assert hasattr(mod, "InterfaceID")
    assert callable(mod.main)
