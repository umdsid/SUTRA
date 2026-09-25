#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/Desktop/SUTRA}"
ROOT="$(cd "$ROOT" && pwd)"

unset PYTHONPATH || true
export PYTHONPATH="$ROOT/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1
export STRATA_LEVEL0_WORKERS="${STRATA_LEVEL0_WORKERS:-3}"
export STRATA_HIERARCHY_WORKERS="${STRATA_HIERARCHY_WORKERS:-3}"
export STRATA_WORKERS="${STRATA_WORKERS:-3}"

STATE="$ROOT/manifests/sutra_five_v0911_full"
LOGROOT="$ROOT/logs/sutra_five_v0911_full"
mkdir -p "$STATE" "$LOGROOT"

cd "$ROOT"

python - <<'PY'
from pathlib import Path
import strata, sys
root=Path.cwd().resolve()
src=(root/"src").resolve()
mod=Path(strata.__file__).resolve()
print("SUTRA ownership preflight")
print(" python:", sys.executable)
print(" strata:", mod)
if src not in mod.parents:
    raise SystemExit(f"ERROR: strata import is outside SUTRA/src: {mod}")
for name in ("data","resources"):
    p=root/name
    if not p.is_dir() or p.is_symlink():
        raise SystemExit(f"ERROR: {name} must be a real local directory: {p}")
print(" ownership: PASS")
PY

run_stage() {
    local name="$1"
    shift
    local marker="$STATE/${name}.complete"
    local log="$LOGROOT/${name}.log"

    if [[ -f "$marker" ]]; then
        echo "[SKIP COMPLETE] $name"
        return 0
    fi

    echo
    echo "===== START $name $(date) ====="
    echo "CMD: $*"

    (
        cd "$ROOT"
        unset PYTHONPATH || true
        export PYTHONPATH="$ROOT/src"
        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export VECLIB_MAXIMUM_THREADS=1
        export MKL_NUM_THREADS=1
        export STRATA_LEVEL0_WORKERS="${STRATA_LEVEL0_WORKERS:-3}"
        export STRATA_HIERARCHY_WORKERS="${STRATA_HIERARCHY_WORKERS:-3}"
        export STRATA_WORKERS="${STRATA_WORKERS:-3}"
        "$@"
    ) 2>&1 | tee "$log"

    touch "$marker"
    echo "[PASS] $name"
}

require_file() {
    [[ -s "$1" ]] || {
        echo "ERROR missing required file: $1" >&2
        exit 20
    }
}

echo
echo "================================================================"
echo "SUTRA FIVE-SPECIMEN CLEAN PRODUCTION"
echo "native mechanics -> Level 0 -> v0.7.9 -> specimen-local v0.9.1.1"
echo "Old pooled v0.9.0/v0.9.1 production is intentionally NOT run."
echo "================================================================"

run_stage gateB_v021 bash ./run_tranche2_1.sh

run_stage geometry_primary_prepare bash -lc '
set -euo pipefail
ROOT="$(pwd)"
PY="$ROOT/.venv/bin/python"
SAMPLES=(alzheimers gbm_reference_addon healthy_reference nondiseased_kidney prcc)
pids=()
for S in "${SAMPLES[@]}"; do
  "$PY" -m strata.cli.tranche2_2b_prepare \
    --project-root "$ROOT" \
    --sample "$S" \
    --max-cells 0 \
    > "$ROOT/logs/sutra_five_v0911_full/geometry_prepare_${S}.log" 2>&1 &
  pids+=("$!")
done
fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "ERROR: geometry preparation failed: ${SAMPLES[$i]}"
    tail -n 80 "$ROOT/logs/sutra_five_v0911_full/geometry_prepare_${SAMPLES[$i]}.log" || true
    fail=1
  else
    echo "[DONE] geometry prepare ${SAMPLES[$i]}"
  fi
done
[[ "$fail" -eq 0 ]]
'

run_stage radius_calibration_v058 bash ./run_tranche2_2e.sh

run_stage radius5_materialize bash ./run_radius5_materialize_v0916.sh

run_stage native_mechanics_v061 bash ./run_native_mechanics_fast.sh
run_stage native_identifiability_v062 bash ./run_native_identifiability_v062.sh
run_stage native_corrections_v064 bash ./run_native_corrections_v064.sh
run_stage native_junction_recovery_v065 bash ./run_native_junction_recovery_v065.sh
run_stage native_interface_completion_v066 bash ./run_native_interface_completion_v066.sh
run_stage native_observability_v067 bash ./run_native_observability_v067.sh
run_stage native_rowspace_v068 bash ./run_native_rowspace_observability_v068.sh
run_stage native_production_v069 bash ./run_native_production_observable_v069.sh

for s in healthy_reference alzheimers gbm_reference_addon nondiseased_kidney prcc; do
    require_file "$ROOT/results/production_mechanics_v069/$s/production_core_tensions.parquet"
    require_file "$ROOT/results/production_mechanics_v069/$s/production_core_pressure_contrasts.parquet"
    require_file "$ROOT/results/production_mechanics_v069/$s/patch_solver_diagnostics.parquet"
done
echo "[PASS] five-sample production mechanics v0.6.9 contract"

run_stage mechanics_finalize_v0691 bash ./run_finalize_mechanics_v0691.sh
run_stage level0_v070 env STRATA_LEVEL0_WORKERS="${STRATA_LEVEL0_WORKERS:-3}" ./run_level0_step3_v070.sh

run_stage level0_freeze_v070 bash ./run_level0_step4_freeze_v070.sh

run_stage h071 bash ./run_hierarchy_v071_active_blocks.sh
run_stage h072 bash ./run_hierarchy_v072_short_pilot.sh
run_stage h073 bash ./run_hierarchy_v073_full.sh
run_stage h074 bash ./run_hierarchy_v074_effective_flow.sh
run_stage h0741 bash ./run_hierarchy_v0741_effective_flow.sh
run_stage h0742 bash ./run_hierarchy_v0742_effective_flow.sh
run_stage h0743 bash ./run_hierarchy_v0743_channel_direction_audit.sh
run_stage h075 bash ./run_hierarchy_v075_local_geometry.sh
run_stage h076 bash ./run_hierarchy_v076_geometry_aware.sh
run_stage h0761 bash ./run_hierarchy_v0761_pressure_tail_audit.sh
run_stage h0762 bash ./run_hierarchy_v0762_pressure_field_finalize.sh
run_stage h077 bash ./run_hierarchy_v077_discrete_transport.sh
run_stage h078 bash ./run_hierarchy_v078_transport_holonomy.sh
run_stage h079 bash ./run_hierarchy_v079_loop_normalized_geometry.sh

require_file "$ROOT/results/hierarchy_v079_loop_normalized_geometry/loop_normalized_geometry_global_certificate.json"

run_stage h0911_calibration bash ./run_hierarchy_v0911_calibrate_only.sh

python - <<'PY'
from pathlib import Path
import json, math
import pandas as pd

root=Path.cwd()
out=root/"results"/"hierarchy_v0911_specimen_local_contextual_flow"
j=json.loads((out/"contextual_calibration.json").read_text())
df=pd.read_csv(out/"specimen_local_calibration_audit.csv")

expected={"alzheimers","gbm_reference_addon","healthy_reference","nondiseased_kidney","prcc"}
seen=set(df["sample"].astype(str))
if seen != expected:
    raise SystemExit(f"ERROR: calibration sample set mismatch: {seen}")

policy=j["cross_specimen_policy"]
assert policy["pooled_component_scaling"] is False
assert policy["pooled_threshold_calibration"] is False
assert policy["forced_equal_merge_fraction"] is False
assert policy["comparison_after_independent_hierarchy"] is True

scale_cols=[c for c in df.columns if c.startswith("scale_")]
for _,r in df.iterrows():
    s=r["sample"]
    t0=float(r["initial_threshold"])
    tm=float(r["maximum_threshold"])
    a0=float(r["allowed_at_initial_fraction"])
    am=float(r["allowed_at_cap_fraction"])
    if not (math.isfinite(t0) and math.isfinite(tm) and t0 < tm):
        raise SystemExit(f"ERROR: invalid threshold schedule for {s}: {t0}, {tm}")
    if not (0.33 <= a0 <= 0.37):
        raise SystemExit(f"ERROR: {s} initial admitted fraction is not q35-like: {a0}")
    if not (0.88 <= am <= 0.92):
        raise SystemExit(f"ERROR: {s} cap admitted fraction is not q90-like: {am}")
    for c in scale_cols:
        v=float(r[c])
        if not (math.isfinite(v) and v > 0):
            raise SystemExit(f"ERROR: {s} invalid specimen-local scale {c}={v}")

print()
print("===== SPECIMEN-LOCAL CALIBRATION VALIDATION: PASS =====")
cols=["sample","initial_threshold","maximum_threshold",
      "allowed_at_initial_fraction","allowed_at_cap_fraction"]
print(df[cols].to_string(index=False))
print()
print("pooled component scaling:   FALSE")
print("pooled threshold calibration: FALSE")
print("forced equal merge fraction: FALSE")
PY

run_stage h0911_full bash ./run_hierarchy_v0911_specimen_local_contextual_flow.sh

python - <<'PY'
from pathlib import Path
import json

root=Path.cwd()
p=root/"results"/"hierarchy_v0911_specimen_local_contextual_flow"/"contextual_flow_global_certificate.json"
if not p.is_file():
    raise SystemExit(f"ERROR: missing v0.9.1.1 global certificate: {p}")
j=json.loads(p.read_text())
print()
print("===== SUTRA v0.9.1.1 GLOBAL RESULT =====")
print("CONTEXTUAL_FLOW_GATE:", j.get("CONTEXTUAL_FLOW_GATE"))
print("FLOW_LEDGER_COMPLETE:", j.get("FLOW_LEDGER_COMPLETE"))
for r in j.get("sample_reports",[]):
    print(
        f"{r['sample']:24s} "
        f"{r['level0_cells']:8,d} -> {r['final_nodes']:8,d} "
        f"removed={100*r['removed_fraction']:6.2f}% "
        f"steps={r['microsteps']:5,d} "
        f"status={r['status']} "
        f"stop={r['stop_reason']}"
    )
print("Certificate:", p)
PY

echo
echo "================================================================"
echo "SUTRA FIVE-SPECIMEN v0.9.1.1 RUN COMPLETE $(date)"
echo "================================================================"
