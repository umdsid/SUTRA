from strata.geometry_adjudication.adjudicate import (
    evaluate_contact_set,conservative_selection_score
)

def test_contact_metrics():
    truth={("A","B"),("B","C")}
    pred={("A","B"),("A","C")}
    r=evaluate_contact_set(pred,truth)
    assert abs(r["contact_precision_vs_gateB"]-0.5)<1e-12
    assert abs(r["contact_recall_vs_gateB"]-0.5)<1e-12

def test_score_prefers_preservation():
    a={"contact_f1_vs_gateB":0.9,"transcript_capture_fraction":0.8,
       "new_contact_burden":0.1,"median_abs_log_area_ratio":0.02}
    b={"contact_f1_vs_gateB":0.6,"transcript_capture_fraction":0.9,
       "new_contact_burden":0.05,"median_abs_log_area_ratio":0.02}
    assert conservative_selection_score(a)>conservative_selection_score(b)
