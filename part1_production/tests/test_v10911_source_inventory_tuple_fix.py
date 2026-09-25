import pandas as pd
from strata_hierarchy.v1091.semantics import classify_source_inventory

def test_itertuples_inventory_classifier_does_not_use_dict_get():
    d=pd.DataFrame([{
        "relative_path":"data/cell_feature_matrix.h5",
        "schema":["matrix/data","matrix/indices","features/name"],
        "size_bytes":123
    }])
    out=classify_source_inventory(d)
    assert int(out.iloc[0].expression_score) >= 2

def test_string_schema_is_supported():
    d=pd.DataFrame([{
        "relative_path":"data/cells.parquet",
        "schema":"cell_id centroid_x centroid_y",
        "size_bytes":123
    }])
    out=classify_source_inventory(d)
    assert int(out.iloc[0].coordinate_score) >= 2

def test_missing_schema_attribute_is_safe():
    d=pd.DataFrame([{
        "relative_path":"data/unknown.npz",
        "size_bytes":123
    }])
    out=classify_source_inventory(d)
    assert "expression_score" in out.columns
    assert "coordinate_score" in out.columns
