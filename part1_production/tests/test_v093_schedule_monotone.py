from strata_hierarchy.v093.flow import schedule_fraction
def test_schedule_only_accelerates_with_reduction():
    cfg={"adaptive_schedule":[
        {"name":"ultraslow","until_removed_fraction":.25,"fraction":.0002},
        {"name":"slow","until_removed_fraction":.5,"fraction":.0005},
        {"name":"late","until_removed_fraction":1.01,"fraction":.005},
    ]}
    a=schedule_fraction(.1,cfg)[0]
    b=schedule_fraction(.3,cfg)[0]
    c=schedule_fraction(.9,cfg)[0]
    assert a<b<c
