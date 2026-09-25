from __future__ import annotations
from collections import defaultdict
import numpy as np
import pandas as pd


def reconcile_patch_solutions(patch_results):
    tv=defaultdict(list)
    pv=defaultdict(list)
    bv=defaultdict(list)
    for r in patch_results:
        if r.get("status")!="PASS":
            continue
        for k,v in r.get("tensions",{}).items(): tv[int(k)].append(float(v))
        for k,v in r.get("pressures",{}).items(): pv[int(k)].append(float(v))
        for k,v in r.get("boundary_pressures",{}).items(): bv[int(k)].append(float(v))

    trows=[{"interface_id":k,"tension":float(np.median(v)),
            "n_patch_estimates":len(v),"patch_sd":float(np.std(v))}
           for k,v in tv.items()]
    prows=[{"cell_label":k,"pressure":float(np.median(v)),
            "n_patch_estimates":len(v),"patch_sd":float(np.std(v))}
           for k,v in pv.items()]
    brows=[{"background_component":k,"boundary_pressure":float(np.median(v)),
            "n_patch_estimates":len(v),"patch_sd":float(np.std(v))}
           for k,v in bv.items()]
    return pd.DataFrame(trows),pd.DataFrame(prows),pd.DataFrame(brows)
