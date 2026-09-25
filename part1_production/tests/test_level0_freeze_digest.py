import pandas as pd

from strata_hierarchy.freeze_audit import canonical_table_digest


def test_digest_is_row_order_independent_after_canonical_sort():
    a=pd.DataFrame({"id":[2,1],"x":[3.0,4.0]})
    b=a.iloc[::-1].reset_index(drop=True)
    assert canonical_table_digest(a,["id"])==canonical_table_digest(b,["id"])
