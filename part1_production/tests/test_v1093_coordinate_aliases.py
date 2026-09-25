def test_centroid_aliases_are_frozen_in_source():
    from pathlib import Path
    import strata_hierarchy.v1093.replay as r
    txt=Path(r.__file__).read_text()
    assert '"x_centroid"' in txt and '"y_centroid"' in txt
