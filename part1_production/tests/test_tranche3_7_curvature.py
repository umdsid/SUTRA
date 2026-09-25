import numpy as np
from shapely.geometry import LineString
from strata.mechanics.curvature_constraints import fit_circle_pratt_like,signed_curvature_from_arc,CurvatureConfig


def test_circle_fit_unit_circle():
    th=np.linspace(0,np.pi/2,50)
    xy=np.c_[np.cos(th),np.sin(th)]
    fit=fit_circle_pratt_like(xy)
    assert fit is not None
    cx,cy,R,rmse=fit
    assert abs(R-1.0)<1e-3
    assert rmse<1e-3


def test_curved_arc_is_valid():
    th=np.linspace(-0.5,0.5,50)
    line=LineString(np.c_[5*np.cos(th),5*np.sin(th)])
    out=signed_curvature_from_arc(
        line,[1.0,0.0],
        CurvatureConfig(min_radius=1.0,max_radius=100.0,max_rel_circle_rmse=0.05,min_turning_angle_rad=0.05)
    )
    assert out["curvature_valid"]
    assert abs(abs(out["curvature"])-0.2)<0.02


def test_straight_arc_rejected():
    line=LineString([(0,0),(1,0),(2,0),(3,0)])
    out=signed_curvature_from_arc(line,[0,1])
    assert not out["curvature_valid"]
