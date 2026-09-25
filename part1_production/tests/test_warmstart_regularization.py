def test_import():
    from strata.mechanics.warmstart_regularization import build_cache, solve_warm
    assert callable(build_cache)
    assert callable(solve_warm)
