from __future__ import annotations
import pandas as pd
import numpy as np

def topology_diagnostics(poly_diag: pd.DataFrame, edges: pd.DataFrame, nodes: pd.DataFrame):
    n_poly = int(len(poly_diag))
    n_invalid = int((poly_diag["status"] != "PASS").sum()) if n_poly else 0
    n_edges = int(len(edges))
    isolated = int((nodes["degree"] == 0).sum()) if len(nodes) else 0
    return {
        "n_polygons": n_poly,
        "n_invalid_polygons": n_invalid,
        "fraction_invalid_polygons": float(n_invalid/n_poly) if n_poly else None,
        "n_observed_interfaces": n_edges,
        "n_topology_nodes": int(len(nodes)),
        "n_isolated_cells": isolated,
        "fraction_isolated_cells": float(isolated/len(nodes)) if len(nodes) else None,
        "mean_degree": float(nodes["degree"].mean()) if len(nodes) else None,
        "median_degree": float(nodes["degree"].median()) if len(nodes) else None,
        "mean_shared_length": float(edges["shared_length"].mean()) if n_edges else None,
        "median_shared_length": float(edges["shared_length"].median()) if n_edges else None,
        "exact_interface_fraction": float((edges["support_method"]=="exact_boundary_intersection").mean()) if n_edges else None,
    }

def reciprocity_audit(edges: pd.DataFrame):
    # Edges are stored canonically once; reciprocity means no duplicate reverse pairs.
    if edges.empty:
        return {"duplicate_unordered_pairs":0, "status":"PASS"}
    keys = edges.apply(lambda r: tuple(sorted((str(r.cell_i), str(r.cell_j)))), axis=1)
    d = int(keys.duplicated().sum())
    return {"duplicate_unordered_pairs": d, "status": "PASS" if d == 0 else "FAIL"}
