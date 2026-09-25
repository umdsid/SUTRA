from strata.preflight.cross_sample import cross_sample_audit


def test_gene_panel_mismatch_fails():
    r=[
        {"sample":"A","_feature_names":["g1","g2"],"_feature_ids":["1","2"],
         "cells":{"coordinates":{"available":False}},"files":{},"morphology_files":[]},
        {"sample":"B","_feature_names":["g2","g1"],"_feature_ids":["2","1"],
         "cells":{"coordinates":{"available":False}},"files":{},"morphology_files":[]},
    ]
    x=cross_sample_audit(r)
    assert x["gene_panel"]["status"]=="FAIL"
