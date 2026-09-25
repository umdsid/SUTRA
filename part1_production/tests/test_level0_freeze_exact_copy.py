import pandas as pd
import pytest

from strata_hierarchy.freeze_audit import (
    Level0FreezeError,
    assert_exact_float_match,
)


def test_exact_copy_passes():
    level=pd.DataFrame({"interface_id":[1,2],"tension":[1.0,-2.5]})
    src=pd.DataFrame({
        "interface_id":[1,2],
        "tension_production":[1.0,-2.5],
    })
    r=assert_exact_float_match(
        level,src,"interface_id","tension","tension_production",name="tau"
    )
    assert r["exact_match"]


def test_exact_copy_rejects_even_small_changed_value():
    level=pd.DataFrame({"interface_id":[1],"tension":[1.0+1e-12]})
    src=pd.DataFrame({"interface_id":[1],"tension_production":[1.0]})
    with pytest.raises(Level0FreezeError):
        assert_exact_float_match(
            level,src,"interface_id","tension","tension_production",name="tau"
        )
