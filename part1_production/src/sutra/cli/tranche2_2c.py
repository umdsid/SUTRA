from __future__ import annotations
import argparse,json,gzip
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import shape

from sutra.geometry_adjudication.adjudicate import (
    polygons_from_boundaries,correct_assignment_fraction,hungarian_match,
    proseg_contacts_with_epsilon,evaluate_contact_set,
    conservative_selection_score,winner_margin
)


def read_geojson_gz(path):
    with gzip.open(path,"rt") as f:
        obj=json.load(f)
    return [{"geometry":shape(feat["geometry"]),
             "properties":feat.get("properties",{}) or {}}
            for feat in obj.get("features",[])]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project-root",default=".")
    ap.add_argument("--epsilon",type=float,default=0.35)
    a=ap.parse_args()
    project=Path(a.project_root).resolve()
    root=project/"results"/"tranche2_2b_geometry_benchmark"
    samples=["alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"]

    reports=[]
    for s in samples:
        out=root/s
        prep=json.loads((out/"prepare_summary.json").read_text())
        oldcert=json.loads((out/"geometry_benchmark_certificate.json").read_text())

        cells=pd.read_csv(out/"primary_cells.csv")
        cells["cell_id"]=cells.cell_id.astype(str)
        bdf=pd.read_parquet(out/"primary_cell_boundaries.parquet")
        bdf["cell_id"]=bdf.cell_id.astype(str)
        orig_polys=polygons_from_boundaries(bdf)

        edges=pd.read_csv(out/"primary_gateB_edges.csv")
        truth={tuple(sorted((str(x),str(y)))) for x,y in zip(edges.cell_i,edges.cell_j)}

        polyfile=out/"proseg"/"cell_polygons.geojson.gz"
        proseg_rows=read_geojson_gz(polyfile)
        mapping,costs=hungarian_match(
            proseg_rows,orig_polys,cells,
            max_centroid_um=20.0,
            overlap_weight=0.75,
            distance_weight=0.25
        )
        pred=proseg_contacts_with_epsilon(proseg_rows,mapping,a.epsilon)
        pscore=evaluate_contact_set(pred,truth)

        txfile=out/"proseg"/"transcript_metadata.csv.gz"
        corr_assign=correct_assignment_fraction(txfile) if txfile.exists() else None

        # centroid displacement on the improved matching
        cdict=cells.set_index("cell_id")[["x_centroid","y_centroid"]]
        disps=[]
        for i,cid in mapping.items():
            if cid not in cdict.index: continue
            p=proseg_rows[i]["geometry"]
            old=cdict.loc[cid].to_numpy(float)
            new=np.array([p.centroid.x,p.centroid.y],float)
            disps.append(float(np.linalg.norm(new-old)))

        pscore.update({
            "method":"proseg",
            "matched_fraction":len(mapping)/max(len(cells),1),
            "matching_median_cost":float(np.median(list(costs.values()))) if costs else None,
            "median_centroid_displacement":float(np.median(disps)) if disps else None,
            "q95_centroid_displacement":float(np.quantile(disps,.95)) if disps else None,
            "corrected_assigned_fraction":corr_assign,
            "contact_epsilon":a.epsilon,
            "matching":"hungarian_overlap_plus_centroid",
        })

        # Start from the current deterministic scores; those were already scored
        # at the production raster-contact rule.
        methods={}
        for name in ("constrained_nearest","transcript_watershed"):
            m=prep["methods"].get(name)
            if m and "score" in m:
                methods[name]=m["score"]
        methods["proseg"]=pscore

        ranking=[]
        for name,score in methods.items():
            ranking.append((conservative_selection_score(score),name))
        ranking.sort(reverse=True)

        margin=winner_margin(ranking)
        winner=ranking[0][1] if ranking else None
        # freeze only with a non-trivial margin and no catastrophic Gate-B loss.
        freeze_ok = (
            winner is not None and
            margin is not None and margin >= 0.03 and
            methods[winner].get("lost_gateB_fraction",1.0) <= 0.15 and
            methods[winner].get("new_contact_burden",1.0) <= 0.35
        )

        rep={
            "sample":s,
            "epsilon":a.epsilon,
            "methods":methods,
            "ranking":[{"method":name,"score":float(val)} for val,name in ranking],
            "winner":winner,
            "winner_margin":margin,
            "freeze_status":"PASS" if freeze_ok else "HOLD",
            "freeze_criteria":{
                "minimum_score_margin":0.03,
                "maximum_lost_gateB_fraction":0.15,
                "maximum_new_contact_burden":0.35,
            },
        }
        (out/"geometry_adjudication_certificate.json").write_text(json.dumps(rep,indent=2))
        reports.append(rep)

        print(f"[{s}] winner={winner} margin={margin:.4f} freeze={rep['freeze_status']}")
        if "proseg" in methods:
            q=methods["proseg"]
            print(
                f"    Proseg corrected: F1={q['contact_f1_vs_gateB']:.3f} "
                f"prec={q['contact_precision_vs_gateB']:.3f} "
                f"recall={q['contact_recall_vs_gateB']:.3f} "
                f"new={q['new_contact_burden']:.1%} "
                f"assigned={q['corrected_assigned_fraction']:.1%}"
            )

    winners={r["winner"] for r in reports}
    all_pass=all(r["freeze_status"]=="PASS" for r in reports)
    same_winner=(len(winners)==1)

    overall={
        "strata_version":"0.5.6",
        "tranche":"2.2c",
        "purpose":"fair geometry adjudication and production freeze",
        "same_winner_all_samples":same_winner,
        "sample_reports":reports,
        "production_geometry":next(iter(winners)) if (same_winner and all_pass) else None,
        "freeze_status":"PASS" if (same_winner and all_pass) else "HOLD",
        "rule":"Freeze requires the same method to win all specimens, each with >=0.03 score margin, <=15% lost Gate-B edges, and <=35% new-contact burden.",
    }
    (root/"tranche2_2c_freeze_certificate.json").write_text(json.dumps(overall,indent=2))
    print()
    print(f"GEOMETRY FREEZE: {overall['freeze_status']}")
    print(f"Production geometry: {overall['production_geometry']}")
    print(f"Certificate: {root/'tranche2_2c_freeze_certificate.json'}")

if __name__=="__main__":main()
