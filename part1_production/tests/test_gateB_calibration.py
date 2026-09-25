import pandas as pd
from shapely.geometry import Polygon
from strata.topology.calibration import (
    paired_boundary_metrics,
    candidate_pairs,
    CalibrationConfig,
    calibrate_sample,
)


def sq(x0,y0,x1,y1):
    return Polygon([(x0,y0),(x1,y0),(x1,y1),(x0,y1)])


def test_boundary_gap_exact_touch():
    a = sq(0,0,1,1)
    b = sq(1,0,2,1)
    m = paired_boundary_metrics(a,b)
    assert abs(m["min_gap"]) < 1e-12


def test_boundary_gap_known_separation():
    a = sq(0,0,1,1)
    b = sq(1.1,0,2.1,1)
    m = paired_boundary_metrics(a,b)
    assert abs(m["min_gap"] - 0.1) < 1e-8


def test_candidate_pairs_respect_max_epsilon():
    p = {"A":sq(0,0,1,1), "B":sq(1.1,0,2.1,1), "C":sq(3,0,4,1)}
    pairs = candidate_pairs(p, 0.2)
    ids = {(a,b) for a,b,*_ in pairs}
    assert ("A","B") in ids
    assert all("C" not in pair for pair in ids)


def test_calibration_materializes_sweep():
    p = {
        "A":sq(0,0,1,1),
        "B":sq(1.05,0,2.05,1),
        "C":sq(2.10,0,3.10,1),
    }
    cfg = CalibrationConfig(
        epsilons=(0.0,0.025,0.05,0.075,0.10,0.15,0.20),
        min_shared_length=0.25,
        plateau_required_steps=1,
    )
    sweep, pairs, conf, cert = calibrate_sample(p,cfg)
    assert len(sweep) == len(cfg.epsilons)
    assert "n_edges" in sweep.columns
    assert "persistence_fraction" in pairs.columns
    assert cert["chosen_epsilon"] in cfg.epsilons
