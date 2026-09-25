def test_imports():
    from strata.mechanics.cached_regularization import build_cached_system, adaptive_calibration
    assert callable(build_cached_system)
    assert callable(adaptive_calibration)
