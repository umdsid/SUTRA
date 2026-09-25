
from pathlib import Path
import ast

P = Path(__file__).parents[1] / "scripts" / "vmsi_interface_preflight_fast.py"
SRC = P.read_text()
TREE = ast.parse(SRC)

def _imports():
    out = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            out.append(node.module)
    return out

def _called_attributes():
    out = []
    for node in ast.walk(TREE):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                out.append(fn.id)
            elif isinstance(fn, ast.Attribute):
                out.append(fn.attr)
    return out

def test_no_tensionmap_segmenter_import_or_call():
    imports = _imports()
    calls = _called_attributes()
    assert not any(
        name == "src.segment" or name.endswith(".segment")
        for name in imports
    )
    assert "Segmenter" not in calls
    assert "process_segmented_image" not in calls

def test_vectorized_mapping():
    assert "exact_mapping_vectorized" in SRC
    assert "old * base + new" in SRC

def test_readme_transform_present():
    assert 'find_boundaries(cell_mask, mode="subpixel")' in SRC
    assert 'label(1 - boundary, connectivity=1)' in SRC

def test_island_safe_fill_only():
    assert "binary_fill_holes" in SRC
    assert "convex_hull_image" not in SRC
