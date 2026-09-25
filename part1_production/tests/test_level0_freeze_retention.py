import pandas as pd
import pytest

from sutra.hierarchy.freeze_audit import (
    Level0FreezeError,
    validate_retention_has_no_mechanics,
)


def test_retention_patch_has_no_mechanics():
    cells=pd.DataFrame({"patch_id":[0,-1,-1]})
    interfaces=pd.DataFrame({
        "owner_patch_id":[0,-1],
        "tension":[1.0,None],
        "delta_pressure":[0.5,None],
    })
    r=validate_retention_has_no_mechanics(cells,interfaces)
    assert r["n_retention_cells"]==2
    assert r["n_geometry_only_interfaces"]==1


def test_retention_mechanics_is_rejected():
    cells=pd.DataFrame({"patch_id":[-1]})
    interfaces=pd.DataFrame({
        "owner_patch_id":[-1],
        "tension":[1.0],
        "delta_pressure":[None],
    })
    with pytest.raises(Level0FreezeError):
        validate_retention_has_no_mechanics(cells,interfaces)
