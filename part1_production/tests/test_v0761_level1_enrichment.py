import pandas as pd
from sutra.hierarchy.v076.pressure_tail_audit import level1_selection_enrichment
def test_selected_pair_enrichment_maps_cell_pairs():
    E=pd.DataFrame({"cell_i_index":[0,1],"cell_j_index":[1,2],"extreme_tail":[True,False]})
    B=pd.DataFrame({"super_i":[0,1],"super_j":[1,2],"selected_geometry":[True,False]})
    r=level1_selection_enrichment(E,B)
    assert r["n_pressure_edges_in_selected_pairs"]==1
    assert r["selected_edge_extreme_fraction"]==1.0
