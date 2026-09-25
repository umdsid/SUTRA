from strata_hierarchy.v071.resource_intake import (
    locate_functional_resources,
    copy_functional_snapshot,
    sha256_file,
)


def test_functional_resource_is_copied_exactly(monkeypatch,tmp_path):
    src=tmp_path/"src";src.mkdir()
    g=src/"h.all.v2025.1.Hs.symbols.gmt"
    g.write_text("S\\tdesc\\tA\\tB\\n")

    # Unit test must be hermetic: do not let ambient machine resources enter.
    import strata_hierarchy.v071.resource_intake as ri
    monkeypatch.setattr(ri,"_mdfind_name",lambda name: [])

    chosen=locate_functional_resources([src])
    assert chosen["hallmark"]==g.resolve()

    dest=tmp_path/"out"
    copied,m=copy_functional_snapshot(chosen,dest)
    hallmark=[r for r in m if r["role"]=="hallmark"][0]
    assert hallmark["source_sha256"]==sha256_file(g)
    assert hallmark["snapshot_sha256"]==sha256_file(g)
    assert sha256_file(hallmark["snapshot_path"])==sha256_file(g)
