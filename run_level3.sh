#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"

FILTER_FILE="${FILTER_FILE:-${PROJECT_DIR}/instances_verified500.txt}"
if [[ ! -f "${FILTER_FILE}" ]]; then
  echo "Error: filter file not found: ${FILTER_FILE}" >&2
  exit 1
fi

LEVEL3_TRAJ_DIR="${LEVEL3_TRAJ_DIR:-${PROJECT_DIR}/../v2_baseline/mini-swe-agent/trajectories_baseline_gpt54mini}"
if [[ ! -d "${LEVEL3_TRAJ_DIR}" ]]; then
  LEVEL3_TRAJ_DIR=""
fi

LEVEL3_REPO_ROOT="${LEVEL3_REPO_ROOT:-${PROJECT_DIR}/repos}"
if [[ ! -d "${LEVEL3_REPO_ROOT}" ]]; then
  echo "Error: Level 3 repo root not found: ${LEVEL3_REPO_ROOT}" >&2
  exit 1
fi

LEVEL3_VERIFIED_STORE="${LEVEL3_VERIFIED_STORE:-${PROJECT_DIR}/output/verified_perturbations/level3}"
if [[ ! -d "${LEVEL3_VERIFIED_STORE}" ]]; then
  echo "Error: Level 3 verified store not found: ${LEVEL3_VERIFIED_STORE}" >&2
  exit 1
fi

OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/trajectories_level3}"
mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export MSWEA_DATASETS_LOCAL_ONLY="${MSWEA_DATASETS_LOCAL_ONLY:-1}"
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"
echo "Full run log: ${RUN_LOG}" >&2
if [[ "${MAPPED_RUN_TEE:-0}" != "1" ]]; then
  echo "Benchmark stdout/stderr -> log only. Watch: tail -f \"${RUN_LOG}\"   (set MAPPED_RUN_TEE=1 to also print here)" >&2
fi
echo "Run mode: nohup background (set MAPPED_RUN_BACKGROUND=0 for foreground)." >&2

_run() {
  local -a level3_traj_args=()
  if [[ -d "${LEVEL3_TRAJ_DIR}" ]]; then
    level3_traj_args=(--level3-trajectories "${LEVEL3_TRAJ_DIR}")
  fi
  echo "=== started $(date -Iseconds) ==="
  echo "Command: $0 $*"
  echo "filter-file: ${FILTER_FILE}"
  if [[ -d "${LEVEL3_TRAJ_DIR}" ]]; then
    echo "level3-trajectories: ${LEVEL3_TRAJ_DIR}"
  else
    echo "level3-trajectories: <not provided; using verified store only>"
  fi
  echo "level3-repo-root: ${LEVEL3_REPO_ROOT}"
  echo "level3-verified-store: ${LEVEL3_VERIFIED_STORE}"
  echo "output: ${OUTPUT_DIR}"
  PYTHONPATH="src:.:${PYTHONPATH:-}" python3 -m minisweagent.run.benchmarks.swebench_mapped \
    -c swebench.yaml \
    -c swebench_openai_gpt54mini.yaml \
    --subset verified \
    --split test \
    --workers 24 \
    --filter-file "${FILTER_FILE}" \
    --level3-mode intra_file_reorder \
    --level3-repo-root "${LEVEL3_REPO_ROOT}" \
    --level3-verified-store "${LEVEL3_VERIFIED_STORE}" \
    --level3-seed 42 \
    --level3-submission-mode keep \
    "${level3_traj_args[@]}" \
    --output "${OUTPUT_DIR}"
  echo "=== finished $(date -Iseconds) ==="
}

export FILTER_FILE LEVEL3_TRAJ_DIR LEVEL3_REPO_ROOT LEVEL3_VERIFIED_STORE OUTPUT_DIR PROJECT_DIR TMPDIR

if [[ "${MAPPED_RUN_BACKGROUND:-1}" == "1" ]]; then
  if [[ "${MAPPED_RUN_TEE:-0}" == "1" ]]; then
    echo "MAPPED_RUN_BACKGROUND ignores MAPPED_RUN_TEE (log file only)." >&2
  fi
  export -f _run
  nohup bash -c '_run' >> "${RUN_LOG}" 2>&1 &
  echo "$!" > "${OUTPUT_DIR}/latest_run.pid"
  echo "Started in background PID $!, log: ${RUN_LOG}" >&2
  echo "PID file: ${OUTPUT_DIR}/latest_run.pid" >&2
  echo "Watch: tail -f \"${RUN_LOG}\"" >&2
  exit 0
fi

if [[ "${MAPPED_RUN_TEE:-0}" == "1" ]]; then
  _run 2>&1 | tee -a "${RUN_LOG}"
else
  _run >> "${RUN_LOG}" 2>&1
fi
