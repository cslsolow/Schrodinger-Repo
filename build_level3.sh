#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/output/verified_perturbations/level3}"
FILTER_FILE="${FILTER_FILE:-${PROJECT_DIR}/instances_verified500.txt}"
BASELINE_TRAJECTORIES="${BASELINE_TRAJECTORIES:-/data/swebench/silinchen/Silin-SWE-Bench/v2_baseline/mini-swe-agent/trajectories_baseline_gpt54mini}"
REPO_ROOT="${REPO_ROOT:-/data/swebench/workspace_henglian/SWE-Search/tmp/repos}"
LOG_DIR="${PROJECT_DIR}/tmp/verified_level3_builder"
LOG_FILE="${LOG_DIR}/verified500_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

if [[ ! -f "${FILTER_FILE}" ]]; then
  echo "Error: filter file not found: ${FILTER_FILE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
PYTHONPATH="src:.:${PYTHONPATH:-}" python3 glasses/build_verified_level3_store.py \
  --filter-file "${FILTER_FILE}" \
  --baseline-trajectories "${BASELINE_TRAJECTORIES}" \
  --repo-root "${REPO_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --tmp-root "${PROJECT_DIR}/tmp/verified_level3_builder" \
  "$@" 2>&1 | tee "${LOG_FILE}"

echo "log: ${LOG_FILE}"
