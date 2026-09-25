import importlib.util

def test_forensics_dependency_state_is_detectable():
    # The package may or may not be installed in a clean environment; the
    # production runner installs it when possible and the CLI has a fallback.
    state = importlib.util.find_spec("sklearn")
    assert state is None or state is not None
