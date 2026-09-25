from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json

import numpy as np
import pandas as pd

from sutra.io.xenium import MatrixMetadata, identify_cell_id_column


@dataclass
class Level0Audit:
    sample_id: str
    n_cells_matrix: int
    n_cells_table: int
    n_features: int
    nnz: int
    exact_cell_set_match: bool
    exact_cell_order_match: bool
    duplicate_matrix_barcodes: int
    duplicate_table_cell_ids: int
    duplicate_feature_ids: int
    duplicate_feature_names: int
    zero_count_cells: int | None
    zero_count_features: int | None
    feature_type_counts: dict[str, int]
    measured_cell_columns: list[str]
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def audit_level0(sample_id: str, matrix: MatrixMetadata, cells: pd.DataFrame, strict_alignment: bool = True) -> Level0Audit:
    cid_col = identify_cell_id_column(cells)
    table_ids = cells.index.astype(str).tolist() if cells.index.name == cid_col else cells[cid_col].astype(str).tolist()
    matrix_ids = [str(x) for x in matrix.barcodes]

    mset = set(matrix_ids)
    tset = set(table_ids)
    exact_set = mset == tset and len(matrix_ids) == len(table_ids)
    exact_order = matrix_ids == table_ids

    duplicate_matrix = len(matrix_ids) - len(mset)
    duplicate_table = len(table_ids) - len(tset)
    duplicate_feature_ids = len(matrix.feature_ids) - len(set(matrix.feature_ids))
    duplicate_feature_names = len(matrix.feature_names) - len(set(matrix.feature_names))

    zero_cells = int(np.sum(matrix.cell_totals == 0)) if matrix.cell_totals is not None else None
    zero_features = int(np.sum(matrix.feature_totals == 0)) if matrix.feature_totals is not None else None
    type_counts = pd.Series(matrix.feature_types, dtype="string").value_counts(dropna=False).to_dict()
    type_counts = {str(k): int(v) for k, v in type_counts.items()}

    failures: list[str] = []
    if duplicate_matrix:
        failures.append("duplicate_matrix_barcodes")
    if duplicate_table:
        failures.append("duplicate_table_cell_ids")
    if duplicate_feature_ids:
        failures.append("duplicate_feature_ids")
    if not exact_set:
        failures.append("cell_set_mismatch")
    if strict_alignment and not exact_order:
        failures.append("cell_order_mismatch")

    return Level0Audit(
        sample_id=sample_id,
        n_cells_matrix=matrix.shape[1],
        n_cells_table=len(table_ids),
        n_features=matrix.shape[0],
        nnz=matrix.nnz,
        exact_cell_set_match=exact_set,
        exact_cell_order_match=exact_order,
        duplicate_matrix_barcodes=duplicate_matrix,
        duplicate_table_cell_ids=duplicate_table,
        duplicate_feature_ids=duplicate_feature_ids,
        duplicate_feature_names=duplicate_feature_names,
        zero_count_cells=zero_cells,
        zero_count_features=zero_features,
        feature_type_counts=type_counts,
        measured_cell_columns=[str(c) for c in cells.columns],
        status="PASS" if not failures else "FAIL:" + ",".join(failures),
    )


def write_level0_tables(outdir: str | Path, matrix: MatrixMetadata, cells: pd.DataFrame) -> None:
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cell_id": matrix.barcodes}).to_csv(out / "barcodes.tsv", sep="\t", index=False)
    pd.DataFrame({
        "feature_id": matrix.feature_ids,
        "feature_name": matrix.feature_names,
        "feature_type": matrix.feature_types,
    }).to_csv(out / "features.tsv", sep="\t", index=False)
    schema = {
        "columns": [
            {"name": str(c), "dtype": str(cells[c].dtype)} for c in cells.columns
        ],
        "index_name": None if cells.index.name is None else str(cells.index.name),
        "n_rows": int(len(cells)),
    }
    (out / "measured_metadata_schema.json").write_text(json.dumps(schema, indent=2) + "\n")
