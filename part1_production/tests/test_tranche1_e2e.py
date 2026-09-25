from pathlib import Path
import json

import h5py
import numpy as np
import pandas as pd

from strata.cli.tranche1 import main


def _write_matrix(path: Path):
    with h5py.File(path, "w") as h:
        g = h.create_group("matrix")
        # features x cells = 3 x 2, CSC
        g.create_dataset("barcodes", data=np.array([b"cell_A", b"cell_B"]))
        g.create_dataset("data", data=np.array([2, 1, 3], dtype=np.int32))
        g.create_dataset("indices", data=np.array([0, 2, 1], dtype=np.int32))
        g.create_dataset("indptr", data=np.array([0, 2, 3], dtype=np.int32))
        g.create_dataset("shape", data=np.array([3, 2], dtype=np.int64))
        f = g.create_group("features")
        f.create_dataset("id", data=np.array([b"g1", b"g2", b"ctrl1"]))
        f.create_dataset("name", data=np.array([b"GENE1", b"GENE2", b"NegativeControlCodeword_1"]))
        f.create_dataset("feature_type", data=np.array([b"Gene Expression", b"Gene Expression", b"Negative Control Codeword"]))


def test_tranche1_end_to_end(tmp_path: Path):
    project = tmp_path / "STRATA"
    data = project / "data" / "sample_alpha"
    data.mkdir(parents=True)
    (project / "configs").mkdir(parents=True)
    (project / "configs" / "tranche1.json").write_text(json.dumps({
        "data_root": "data",
        "results_root": "results/tranche1_level0",
        "hash_inputs": True,
        "audit_sparse_counts": True,
        "require_cells_table": True,
        "strict_barcode_alignment": True
    }))
    _write_matrix(data / "sample_alpha_cell_feature_matrix.h5")
    pd.DataFrame({
        "cell_id": ["cell_A", "cell_B"],
        "x_centroid": [1.0, 2.0],
        "y_centroid": [3.0, 4.0],
    }).to_csv(data / "sample_alpha_cells.csv", index=False)
    # Current discovery correctly requires measured cell-boundary assets.
    # Tranche 1 hashes/provenances this file but does not parse its geometry.
    pd.DataFrame({
        "cell_id": ["cell_A", "cell_B"],
        "vertex_x": [0.0, 1.0],
        "vertex_y": [0.0, 1.0],
    }).to_parquet(data / "sample_alpha_cell_boundaries.parquet", index=False)

    rc = main(["--project-root", str(project)])
    assert rc == 0
    summary = json.loads((project / "results/tranche1_level0/tranche1_summary.json").read_text())
    assert summary["overall_status"] == "PASS"
    assert summary["n_samples"] == 1
    audit = summary["samples"][0]
    assert audit["n_cells_matrix"] == 2
    assert audit["n_features"] == 3
    assert audit["exact_cell_order_match"] is True
