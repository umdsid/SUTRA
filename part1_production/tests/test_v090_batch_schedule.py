from strata_hierarchy.v090.slow_flow import SlowFlowConfig,batch_fraction

def test_larger_steps_are_last_resort():
    c=SlowFlowConfig().validate()
    a=batch_fraction(0,c)
    b=batch_fraction(c.soft_step_1,c)
    d=batch_fraction(c.soft_step_2,c)
    e=batch_fraction(c.soft_step_3,c)
    assert a[0] < b[0] <= d[0] <= e[0]
    assert a[1]=="epsilon"
    assert e[1]=="last_resort"
