from strata_hierarchy.v1100.metrics import block_speed
def test_symmetric_relative_speed_is_scale_free():
    a={"x":10.0,"y":20.0};b={"x":11.0,"y":18.0}
    c={"x":1000.0,"y":2000.0};d={"x":1100.0,"y":1800.0}
    assert abs(block_speed(a,b,["x","y"])-block_speed(c,d,["x","y"]))<1e-12
