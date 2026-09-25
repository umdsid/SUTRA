
from pathlib import Path
import json
from strata_hierarchy.v1112.recovery import isolated_historical_landmarks
def test_global_json_does_not_cross_samples(tmp_path):
    stage=tmp_path/"results"/"hist"; stage.mkdir(parents=True)
    (stage/"global.json").write_text(json.dumps({
      "healthy_reference":{"landmark_nodes":[100,50]},
      "alzheimers":{"landmark_nodes":[999,888]}}))
    cfg={"historical_stages":["hist"]}
    r=isolated_historical_landmarks(tmp_path,"healthy_reference",cfg)
    assert r[0]["nodes"]==[100,50]
