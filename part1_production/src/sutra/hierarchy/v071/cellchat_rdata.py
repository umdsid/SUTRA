"""CellChat native R-data intake for STRATA v0.7.1.2.1.

Extraction precedence:
  1. pure-Python rdata package;
  2. Rscript fallback when available.

No CellChat inference routine is called.
"""

from __future__ import annotations

from pathlib import Path
import importlib
import shutil
import subprocess
from typing import Any

import numpy as np
import pandas as pd

from .resource_intake import (
    CELLCHAT_INTERACTION,
    _mdfind_name,
    _recursive_name_search,
    _unique_existing,
    sha256_file,
)

CELLCHAT_RDATA_NAMES = (
    "CellChatDB.human.rda",
    "CellChatDB.human.RData",
    "CellChatDB.human.rds",
)


def locate_cellchat_database(search_roots) -> tuple[str, Path] | None:
    roots = _unique_existing(search_roots)

    hits = _recursive_name_search(roots, CELLCHAT_INTERACTION)
    if hits:
        return "interaction_csv", sorted(hits)[0]

    for name in CELLCHAT_RDATA_NAMES:
        hits = _recursive_name_search(roots, name)
        if hits:
            p = sorted(hits)[0]
            return ("human_rds" if p.suffix.lower() == ".rds" else "human_rda", p)

    hits = _mdfind_name(CELLCHAT_INTERACTION)
    if hits:
        return "interaction_csv", sorted(hits)[0]

    ambient = []
    for name in CELLCHAT_RDATA_NAMES:
        ambient.extend(_mdfind_name(name))
    ambient = _unique_existing(ambient)
    if ambient:
        p = sorted(ambient)[0]
        return ("human_rds" if p.suffix.lower() == ".rds" else "human_rda", p)

    return None


def _as_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()

    # xarray DataArray may appear for some R arrays.
    if hasattr(obj, "to_pandas"):
        q = obj.to_pandas()
        if isinstance(q, pd.DataFrame):
            return q

    if isinstance(obj, dict):
        # A converted R dataframe may appear as a mapping of equal-length columns.
        lengths = []
        for v in obj.values():
            try:
                lengths.append(len(v))
            except TypeError:
                lengths.append(None)
        valid = [x for x in lengths if x is not None]
        if valid and len(set(valid)) == 1 and len(valid) == len(lengths):
            return pd.DataFrame(obj)

    raise TypeError(
        f"interaction object cannot be converted to DataFrame: {type(obj).__name__}"
    )


def _find_interaction(obj: Any) -> pd.DataFrame | None:
    """Recursively locate a named interaction member in converted R objects."""
    if isinstance(obj, dict):
        # Prefer exact CellChat list field.
        if "interaction" in obj:
            return _as_dataframe(obj["interaction"])

        # Common top-level RDA object.
        if "CellChatDB.human" in obj:
            found = _find_interaction(obj["CellChatDB.human"])
            if found is not None:
                return found

        for value in obj.values():
            try:
                found = _find_interaction(value)
            except (TypeError, ValueError):
                found = None
            if found is not None:
                return found

    if isinstance(obj, (list, tuple)):
        for value in obj:
            try:
                found = _find_interaction(value)
            except (TypeError, ValueError):
                found = None
            if found is not None:
                return found

    return None


def export_cellchat_interaction_with_python(
    source: str | Path,
    source_kind: str,
    out_csv: str | Path,
) -> dict:
    source = Path(source).resolve()
    out_csv = Path(out_csv).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        rdata = importlib.import_module("rdata")
    except ImportError as exc:
        raise RuntimeError("Python package 'rdata' is not installed") from exc

    if source_kind == "human_rds":
        converted = rdata.read_rds(source)
    elif source_kind == "human_rda":
        converted = rdata.read_rda(source)
    else:
        raise ValueError(f"unsupported source kind: {source_kind}")

    interaction = _find_interaction(converted)
    if interaction is None:
        top = (
            list(converted.keys())
            if isinstance(converted, dict)
            else [type(converted).__name__]
        )
        raise RuntimeError(
            "pure-Python R-data parser could not locate CellChat interaction "
            f"table; top-level objects={top}"
        )

    # Normalize index into a normal CSV column only when it is meaningful.
    interaction = interaction.copy()
    if interaction.index.name is not None:
        interaction = interaction.reset_index()

    interaction.to_csv(out_csv, index=False)

    if not out_csv.exists() or out_csv.stat().st_size == 0:
        raise RuntimeError("pure-Python CellChat export produced no CSV")

    return {
        "source_kind": source_kind,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "export_path": str(out_csv),
        "export_sha256": sha256_file(out_csv),
        "export_method": "python_rdata",
        "rdata_version": getattr(rdata, "__version__", "unknown"),
        "n_exported_rows": int(len(interaction)),
        "export_columns": [str(c) for c in interaction.columns],
    }


def export_cellchat_interaction_with_r(
    source: str | Path,
    source_kind: str,
    out_csv: str | Path,
    extractor_script: str | Path,
) -> dict:
    source = Path(source).resolve()
    out_csv = Path(out_csv).resolve()
    extractor_script = Path(extractor_script).resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError("Rscript is not available")

    proc = subprocess.run(
        [rscript, str(extractor_script), str(source), source_kind, str(out_csv)],
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Rscript CellChat export failed: "
            + (proc.stderr.strip() or proc.stdout.strip())
        )

    if not out_csv.exists() or out_csv.stat().st_size == 0:
        raise RuntimeError("Rscript reported success but wrote no CSV")

    return {
        "source_kind": source_kind,
        "source_path": str(source),
        "source_sha256": sha256_file(source),
        "export_path": str(out_csv),
        "export_sha256": sha256_file(out_csv),
        "export_method": "Rscript",
        "Rscript": rscript,
        "extractor_script": str(extractor_script),
        "extractor_sha256": sha256_file(extractor_script),
    }


def export_cellchat_interaction(
    source: str | Path,
    source_kind: str,
    out_csv: str | Path,
    extractor_script: str | Path | None = None,
) -> dict:
    """
    Robust extraction with Python first, Rscript only as fallback.

    Failure from both routes is reported verbatim.
    """
    py_error = None
    try:
        return export_cellchat_interaction_with_python(
            source, source_kind, out_csv
        )
    except Exception as exc:
        py_error = f"{type(exc).__name__}: {exc}"

    if extractor_script is not None and shutil.which("Rscript") is not None:
        try:
            meta = export_cellchat_interaction_with_r(
                source, source_kind, out_csv, extractor_script
            )
            meta["python_rdata_error"] = py_error
            return meta
        except Exception as exc:
            r_error = f"{type(exc).__name__}: {exc}"
    else:
        r_error = "Rscript unavailable"

    raise RuntimeError(
        "CellChat R-data extraction failed. "
        f"python_rdata=[{py_error}] ; Rscript=[{r_error}]"
    )


__all__ = [
    "CELLCHAT_RDATA_NAMES",
    "locate_cellchat_database",
    "export_cellchat_interaction_with_python",
    "export_cellchat_interaction_with_r",
    "export_cellchat_interaction",
]
