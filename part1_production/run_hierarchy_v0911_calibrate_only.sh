#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec "$ROOT/run_hierarchy_v0911_specimen_local_contextual_flow.sh" --calibrate-only "$@"
