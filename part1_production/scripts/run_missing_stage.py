
from __future__ import annotations
import argparse, importlib.util, json, os, re, subprocess, sys
from pathlib import Path

STAGES = {
    "h071": {
        "modules":["sutra.cli.hierarchy_active_blocks_v071"],
        "config_tokens":["v071","active_blocks"],
    },
    "h0741": {
        "modules":["sutra.cli.hierarchy_effective_flow_v0741"],
        "config_tokens":["v0741","effective_flow"],
    },
    "h0742": {
        "modules":["sutra.cli.hierarchy_effective_flow_v0742"],
        "config_tokens":["v0742","effective_flow"],
    },
    "h075": {
        "modules":[
            "sutra.cli.hierarchy_local_geometry_v075_step1",
            "sutra.cli.hierarchy_local_geometry_v075_step2",
            "sutra.cli.hierarchy_local_geometry_v075_step3",
            "sutra.cli.hierarchy_local_geometry_v075_step4",
        ],
        "config_tokens":["v075","local_geometry"],
    },
    "h092": {
        "modules":["sutra.cli.step_spectrum_consistency_v092"],
        "config_tokens":["v092","step_spectrum"],
    },
}

PROJECT_FLAGS=("--project-root","--project_root","--project")
CONFIG_FLAGS=("--config","--config-path","--config_path")

def module_source(module):
    sp=importlib.util.find_spec(module)
    if sp is None or not sp.origin:
        return None,None
    p=Path(sp.origin)
    return p,p.read_text(errors="ignore")

def help_text(py,module,project):
    env=os.environ.copy()
    env["PYTHONPATH"]=str(project/"src")+os.pathsep+env.get("PYTHONPATH","")
    cp=subprocess.run([str(py),"-m",module,"--help"],cwd=project,env=env,
                      stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    return cp.returncode,cp.stdout

def pick_flag(help_txt, flags):
    for f in flags:
        if f in help_txt:return f
    return None

def source_default_config(src,project):
    if not src:return None
    # explicit configs/foo.json source literals
    hits=re.findall(r'configs/[A-Za-z0-9_.\-/]+\.json',src)
    for h in hits:
        p=project/h
        if p.exists():return p
    return None

def score_config(p,tokens):
    n=p.name.lower()
    score=0
    for t in tokens:
        t=t.lower()
        if t in n:score+=10
        compact=t.replace("_","")
        if compact and compact in n.replace("_",""):score+=4
    return score

def find_config(project,tokens,src):
    p=source_default_config(src,project)
    if p is not None:return p,"source_literal"
    configs=sorted((project/"configs").glob("*.json"))
    cand=sorted([(score_config(p,tokens),p) for p in configs],reverse=True,key=lambda x:x[0])
    if cand and cand[0][0]>0:return cand[0][1],"filename_match"
    return None,None

def build_command(py,module,project,tokens):
    sp,src=module_source(module)
    if sp is None:
        raise RuntimeError(f"module not found: {module}")
    rc,ht=help_text(py,module,project)
    # argparse --help normally exits 0. If not, retain output for diagnosis.
    if rc not in (0,):
        raise RuntimeError(f"{module} --help failed ({rc}):\n{ht[-3000:]}")
    cmd=[str(py),"-m",module]
    pf=pick_flag(ht,PROJECT_FLAGS)
    if pf:cmd += [pf,str(project)]
    cf=pick_flag(ht,CONFIG_FLAGS)
    config=None;source=None
    if cf:
        config,source=find_config(project,tokens,src)
        if config is None:
            raise RuntimeError(f"{module}: accepts {cf} but no matching config was resolved")
        cmd += [cf,str(config)]
    return {
        "module":module,"source":str(sp),"project_flag":pf,"config_flag":cf,
        "config":str(config) if config else None,"config_resolution":source,
        "command":cmd,"help_head":"\n".join(ht.splitlines()[:20])
    }

def plan_stage(project,stage):
    if stage not in STAGES:raise KeyError(stage)
    py=project/".venv/bin/python"
    spec=STAGES[stage];plans=[]
    for m in spec["modules"]:
        plans.append(build_command(py,m,project,spec["config_tokens"]))
    return plans

def execute(plans,project):
    env=os.environ.copy()
    env["PYTHONPATH"]=str(project/"src")+os.pathsep+env.get("PYTHONPATH","")
    for p in plans:
        print(">>>"," ".join(p["command"]),flush=True)
        rc=subprocess.call(p["command"],cwd=project,env=env)
        if rc!=0:raise SystemExit(rc)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--project",required=True)
    ap.add_argument("--stage",required=True,choices=sorted(STAGES))
    ap.add_argument("--probe",action="store_true")
    a=ap.parse_args()
    project=Path(a.project).resolve()
    plans=plan_stage(project,a.stage)
    out=project/"manifests"/"missing_runner_adapters"
    out.mkdir(parents=True,exist_ok=True)
    q=out/f"{a.stage}.json";q.write_text(json.dumps(plans,indent=2))
    print(f"Adapter plan {a.stage}:")
    for p in plans:
        print(" module:",p["module"])
        print(" source:",p["source"])
        print(" project flag:",p["project_flag"])
        print(" config:",p["config"])
        print(" command:"," ".join(p["command"]))
    print("Plan:",q)
    if not a.probe:execute(plans,project)

if __name__=="__main__":main()
