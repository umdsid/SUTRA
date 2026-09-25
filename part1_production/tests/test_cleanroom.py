from pathlib import Path
import ast


def test_no_external_project_imports():
    root = Path(__file__).resolve().parents[1] / "src"
    bad = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.lower().startswith("sohl"):
                        bad.append((path, alias.name))
            elif isinstance(node, ast.ImportFrom):
                if (node.module or "").lower().startswith("sohl"):
                    bad.append((path, node.module))
    assert not bad, f"External project imports detected: {bad}"
