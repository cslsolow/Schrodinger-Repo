#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}"
OUTPUT_ROOT="${PROJECT_DIR}/output/verified_perturbations/level4b"
FILTER_FILE="${FILTER_FILE:-${PROJECT_DIR}/instances_verified500.txt}"
LOG_DIR="${PROJECT_DIR}/tmp/verified_level4b_builder"
LOG_FILE="${LOG_DIR}/verified500_$(date +%Y%m%d_%H%M%S).log"
LEVEL4B_BUILD_WORKERS="${LEVEL4B_BUILD_WORKERS:-4}"
LEVEL4B_MODEL_NAME="${LEVEL4B_MODEL_NAME:-openai/gpt-5.4-mini}"
LEVEL4B_MODEL_CONFIG="${LEVEL4B_MODEL_CONFIG:-${PROJECT_DIR}/src/minisweagent/config/benchmarks/swebench_openai_gpt54mini.yaml}"
LEVEL4B_REPO_ROOT="${LEVEL4B_REPO_ROOT:-${PROJECT_DIR}/repos}"
FORWARD_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --output-root=*)
      OUTPUT_ROOT="${1#*=}"
      shift
      ;;
    --filter-file)
      FILTER_FILE="$2"
      shift 2
      ;;
    --filter-file=*)
      FILTER_FILE="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_ROOT}"

if [[ ! -f "${FILTER_FILE}" ]]; then
  echo "Error: filter file not found: ${FILTER_FILE}" >&2
  exit 1
fi

cd "${PROJECT_DIR}"
export PYTHONUNBUFFERED=1
export TMPDIR="${PROJECT_DIR}/tmp"
mkdir -p "${TMPDIR}"

_run() {
  echo "=== Level 4 build started $(date -Iseconds) ==="
  echo "filter-file: ${FILTER_FILE}"
  echo "output-root: ${OUTPUT_ROOT}"
  echo "tmp-root: ${PROJECT_DIR}/tmp/verified_level4b_builder"
  echo "workers: ${LEVEL4B_BUILD_WORKERS}"
  echo "model-name: ${LEVEL4B_MODEL_NAME}"
  echo "model-config: ${LEVEL4B_MODEL_CONFIG}"
  echo "repo-root: ${LEVEL4B_REPO_ROOT}"
  echo "log: ${LOG_FILE}"
  PYTHONPATH="src:.:${PYTHONPATH:-}" python3 glasses/build_verified_level4b_store.py \
    --filter-file "${FILTER_FILE}" \
    --repo-root "${LEVEL4B_REPO_ROOT}" \
    --output-root "${OUTPUT_ROOT}" \
    --tmp-root "${PROJECT_DIR}/tmp/verified_level4b_builder" \
    --target-variants 3 \
    --variant-count 12 \
    --workers "${LEVEL4B_BUILD_WORKERS}" \
    --model-name "${LEVEL4B_MODEL_NAME}" \
    --model-config "${LEVEL4B_MODEL_CONFIG}" \
    "$@"
  echo "=== Level 4 build finished $(date -Iseconds) ==="
}

if [[ "${LEVEL4B_BUILD_BACKGROUND:-1}" == "1" ]]; then
  export PROJECT_DIR FILTER_FILE OUTPUT_ROOT LOG_FILE LEVEL4B_BUILD_WORKERS LEVEL4B_MODEL_NAME LEVEL4B_MODEL_CONFIG LEVEL4B_REPO_ROOT PYTHONUNBUFFERED TMPDIR
  export -f _run
  nohup bash -c '_run "$@"' bash "${FORWARD_ARGS[@]}" >> "${LOG_FILE}" 2>&1 &
  echo "$!" > "${OUTPUT_ROOT}/latest_build.pid"
  echo "Started Level 4 builder in background PID $!, log: ${LOG_FILE}" >&2
  echo "PID file: ${OUTPUT_ROOT}/latest_build.pid" >&2
  echo "Watch: tail -f \"${LOG_FILE}\"" >&2
  exit 0
fi

if [[ "${LEVEL4B_BUILD_TEE:-0}" == "1" ]]; then
  _run "${FORWARD_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
else
  _run "${FORWARD_ARGS[@]}" >> "${LOG_FILE}" 2>&1
fi

echo "log: ${LOG_FILE}"
