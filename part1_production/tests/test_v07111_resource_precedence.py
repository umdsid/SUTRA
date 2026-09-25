from pathlib import Path

import strata_hierarchy.v071.resource_intake as ri


def test_explicit_root_outranks_spotlight(monkeypatch,tmp_path):
    explicit=tmp_path/"explicit"
    ambient=tmp_path/"ambient"
    explicit.mkdir(); ambient.mkdir()

    name="h.all.v2025.1.Hs.symbols.gmt"
    local=explicit/name
    other=ambient/name
    local.write_text("LOCAL\\td\\tA\\tB\\n")
    other.write_text("AMBIENT\\td\\tX\\tY\\n")

    monkeypatch.setattr(
        ri,
        "_mdfind_name",
        lambda queried: [other] if queried==name else [],
    )

    chosen=ri.locate_functional_resources([explicit])
    assert local.resolve() in set(chosen.values())
    assert other.resolve() not in set(chosen.values())


def test_spotlight_is_used_when_explicit_root_lacks_collection(monkeypatch,tmp_path):
    ambient=tmp_path/"ambient"
    ambient.mkdir()
    name="h.all.v2025.1.Hs.symbols.gmt"
    other=ambient/name
    other.write_text("AMBIENT\\td\\tX\\tY\\n")

    monkeypatch.setattr(
        ri,
        "_mdfind_name",
        lambda queried: [other] if queried==name else [],
    )

    chosen=ri.locate_functional_resources([tmp_path/"does_not_exist"])
    assert other.resolve() in set(chosen.values())
