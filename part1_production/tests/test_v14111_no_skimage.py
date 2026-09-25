
import ast
from pathlib import Path
import numpy as np
import strata.cli.mechanics_admissibility_domain_v0515 as m

def test_module_has_no_skimage_import():
    src_path = Path(m.__file__)
    tree = ast.parse(src_path.read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.append(node.module)
    assert not any(name == "skimage" or name.startswith("skimage.") for name in imported)

def test_sequentialize_keeps_cell_identity_separate():
    x=np.array([[0,9,9,0,4,4],[0,9,9,0,4,4]],dtype=np.int32)
    y,rows=m.sequentialize_connected_labels(x)
    assert rows==[(4,1),(9,2)]
    assert set(np.unique(y))=={0,1,2}
    assert np.all(y[x==4]==1)
    assert np.all(y[x==9]==2)

def test_disconnected_guard():
    x=np.zeros((5,5),np.int32)
    x[0,0]=7;x[4,4]=7
    assert m.disconnected_labels(x)==[7]
