import numpy as np
import pandas as pd
from strata.mechanics.dispersion_regularization import (
    confidence_weights, neff_fraction, top_mass
)


def test_confidence_weights():
    e = pd.DataFrame({
        "confidence_class":["high_confidence","admissible"]
    })
    w = confidence_weights(e)
    assert np.allclose(w, [0.75,1.0])


def test_neff_uniform_is_one():
    t = np.ones(100)
    assert abs(neff_fraction(t)-1.0) < 1e-12


def test_topmass_uniform():
    t = np.ones(100)
    assert abs(top_mass(t,0.01)-0.01) < 1e-12
