def gate(recall,new,lcc,preserved=True):
    return preserved and recall>=0.95 and new<=0.35 and lcc>=0.90

def test_gate_boundary():
    assert gate(.95,.35,.90)
    assert not gate(.95,.351,.90)
