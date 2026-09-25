import pytest
from scipy import sparse
from strata_hierarchy.v075.metric_core import SymmetricBaseConfig,build_symmetric_base_metric
def test_nonpositive_weight_rejected():
    with pytest.raises(ValueError):
        build_symmetric_base_metric(sparse.eye(2),1.0,2,SymmetricBaseConfig(mechanics_tension_weight=0.0))
