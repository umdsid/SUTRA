import pandas as pd

from strata_hierarchy.v071.block_preflight import audit_signaling_resources


def test_untyped_signaling_table_holds(tmp_path):
    d=tmp_path/"cellchat";d.mkdir()
    pd.DataFrame({
        "sender_gene":["L"],
        "receiver_gene":["R"],
    }).to_csv(d/"cellchat_interactions.csv",index=False)
    audit,reg=audit_signaling_resources(tmp_path,{"L","R"})
    assert audit["status"]=="HOLD"
    assert audit["tables"][0]["status"]=="UNTYPED_SPATIAL_SUPPORT"


def test_typed_registry_deduplicates_evidence(tmp_path):
    d=tmp_path/"signaling";d.mkdir()
    pd.DataFrame({
        "sender_gene":["L","L"],
        "receiver_gene":["R","R"],
        "kind":["contact","contact"],
        "database":["A","B"],
    }).to_csv(d/"ligand_receptor.csv",index=False)
    audit,reg=audit_signaling_resources(tmp_path,{"L","R"})
    assert audit["status"]=="PASS"
    assert len(reg)==1
    assert int(reg.iloc[0].evidence_rows)==2
